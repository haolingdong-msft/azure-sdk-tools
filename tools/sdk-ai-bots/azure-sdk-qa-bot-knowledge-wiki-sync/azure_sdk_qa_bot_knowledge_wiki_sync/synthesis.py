"""Bottom-up synthesis: fill node summaries and rolled-up wiki pages.

Walks the ToC tree **bottom-up** (children before parents) and fills:

* **leaf sections** — a one-line ``summary`` (the raw text stays in
  ``section_text`` as citable evidence);
* **internal nodes** — a ``page`` rolled up from their children's summaries, and
  a ``summary`` derived from that page.

Two synthesizers are used so cost can be targeted:

* ``summarizer`` — writes every node's ``summary`` (and, for internal *sections*,
  their extractive roll-up ``page``). Kept cheap (extractive) by default so the
  embedding-entry text is stable.
* ``page_writer`` — writes the **document-level** overview ``page`` (the
  WeKnora-style cross-document distillation that retrieval surfaces as a
  ``(overview)`` synthesis reference). This is where an LLM adds the most value,
  so ``page_writer`` may be an LLM while ``summarizer`` stays extractive — a clean
  A/B that changes only the surfaced overview pages.

The tree hierarchy *is* the topic clustering, and the per-level roll-up *is* the
hierarchical equivalent of GraphRAG's community reports — without entity
extraction or community detection. Independent nodes are synthesised
concurrently (bottom-up waves) so an LLM ``page_writer`` stays fast.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from .llm import Synthesizer, _first_sentences
from .models import KIND_DOC, KIND_SECTION, WikiTree

logger = logging.getLogger(__name__)


def synthesize_tree(
    tree: WikiTree,
    summarizer: Synthesizer,
    page_writer: Synthesizer | None = None,
    *,
    doc_pages_only: bool = True,
    max_children_briefs: int = 40,
    max_workers: int | None = None,
) -> None:
    """Populate ``summary`` and ``page`` on every node, in place, bottom-up.

    ``page_writer`` (defaults to ``summarizer``) writes document overview pages;
    when ``doc_pages_only`` is True only ``doc`` nodes use it (internal sections
    fall back to the cheap ``summarizer`` roll-up).
    """
    page_writer = page_writer or summarizer
    if max_workers is None:
        max_workers = int(os.environ.get("WIKI_SYNTH_MAX_WORKERS", "16"))

    counts = {"leaf": 0, "doc_page": 0, "section_page": 0, "other": 0}

    def process(nid: str) -> str:
        node = tree.nodes[nid]
        children = tree.children_of(nid)
        if not children:
            node.summary = summarizer.summarize(node.title, node.section_text)
            return "leaf"

        child_briefs = [
            f"{c.title}: {c.summary}".strip().rstrip(":").strip()
            for c in children[:max_children_briefs]
            if (c.summary or c.title)
        ]

        if node.kind == KIND_DOC:
            node.page = page_writer.roll_up(node.title, child_briefs, node.section_text)
            node.summary = _first_sentences(node.page, n=1, limit=300) or node.title
            return "doc_page"
        if node.kind == KIND_SECTION:
            writer = summarizer if doc_pages_only else page_writer
            node.page = writer.roll_up(node.title, child_briefs, node.section_text)
            node.summary = _first_sentences(node.page, n=1, limit=300) or node.title
            return "section_page"
        # folder / root: cheap extractive topic list, no page.
        node.summary = (
            f"{node.title}: covers " + ", ".join(c.title for c in children[:12])
        )[:300]
        return "other"

    # Bottom-up waves: a node is ready once all its children are processed.
    pending = {nid: len(n.children) for nid, n in tree.nodes.items()}
    ready = [nid for nid, c in pending.items() if c == 0]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while ready:
            for kind in executor.map(process, ready):
                counts[kind] += 1
            next_ready: list[str] = []
            for nid in ready:
                parent = tree.nodes[nid].parent
                if parent is not None and parent in pending:
                    pending[parent] -= 1
                    if pending[parent] == 0:
                        next_ready.append(parent)
            ready = next_ready

    tree.stats["synthesized"] = counts
    logger.info(
        "synthesize_tree: %d leaf summaries, %d doc pages, %d section pages "
        "(doc_pages_only=%s, workers=%d)",
        counts["leaf"],
        counts["doc_page"],
        counts["section_page"],
        doc_pages_only,
        max_workers,
    )
