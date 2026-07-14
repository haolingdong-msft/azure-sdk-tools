"""Bottom-up synthesis: fill node summaries, knowledge cards, and domain digests.

Two synthesis styles, selected by ``mode``:

* ``overview`` (legacy) — internal nodes get a *navigation* roll-up page written
  from their children's summaries. Good for "where to look", weak on facts.
* ``knowledge`` (default for the knowledge path) — the artifact the agent should
  reason FROM:
    - **document** nodes get a **knowledge card**: dense, declarative facts /
      rules / exact API names extracted by an LLM reading the document's *full
      text* (not child summaries), so no information is lost to summary-of-
      summaries;
    - **folder** nodes get a **domain-knowledge digest** rolled up from their
      documents' knowledge cards — the compact "what an expert knows about this
      area" surface used for pre-loaded domain knowledge.
  Leaf sections keep a cheap extractive ``summary`` (the embedding-entry text).

Independent nodes at each level are processed concurrently.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from .llm import Synthesizer, _first_sentences
from .models import KIND_DOC, KIND_FOLDER, KIND_ROOT, KIND_SECTION, WikiTree

logger = logging.getLogger(__name__)

_MAX_DOC_TEXT_CHARS = 9000


def _doc_full_text(tree: WikiTree, doc_id: str) -> str:
    """Reconstruct a document's text from its section subtree, in reading order."""
    parts: list[str] = []

    def walk(nid: str) -> None:
        node = tree.nodes.get(nid)
        if node is None:
            return
        if node.header_path:
            depth = min(len(node.header_path), 6)
            parts.append(f"{'#' * depth} {node.title}")
        if node.section_text:
            parts.append(node.section_text)
        for child in node.children:
            walk(child)

    doc = tree.nodes.get(doc_id)
    if doc is None:
        return ""
    if doc.section_text:
        parts.append(doc.section_text)
    for child in doc.children:
        walk(child)
    return "\n\n".join(parts)[:_MAX_DOC_TEXT_CHARS]


def synthesize_tree(
    tree: WikiTree,
    summarizer: Synthesizer,
    page_writer: Synthesizer | None = None,
    *,
    mode: str = "knowledge",
    max_children_briefs: int = 40,
    max_workers: int | None = None,
) -> None:
    """Populate ``summary`` (all nodes) and ``page`` (internal nodes), bottom-up.

    ``mode='knowledge'`` makes document ``page``s knowledge cards (from full doc
    text) and folder ``page``s domain digests; ``mode='overview'`` keeps the
    legacy navigation roll-ups. ``page_writer`` (defaults to ``summarizer``)
    performs the LLM-heavy page synthesis.
    """
    page_writer = page_writer or summarizer
    if max_workers is None:
        max_workers = int(os.environ.get("WIKI_SYNTH_MAX_WORKERS", "16"))
    knowledge = mode == "knowledge"

    counts = {"leaf": 0, "doc_page": 0, "section_page": 0, "folder_digest": 0, "other": 0}

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
            if knowledge:
                full_text = _doc_full_text(tree, nid)
                node.page = page_writer.extract_knowledge(node.title, full_text)
            else:
                node.page = page_writer.roll_up(node.title, child_briefs, node.section_text)
            node.summary = _first_sentences(node.page, n=1, limit=300) or node.title
            return "doc_page"

        if node.kind == KIND_SECTION:
            if knowledge:
                # Section pages stay cheap; the doc card is the knowledge unit.
                node.page = ExtractiveKnowledge(summarizer).roll(node, children)
                node.summary = _first_sentences(node.page, n=1, limit=300) or node.title
            else:
                node.page = summarizer.roll_up(node.title, child_briefs, node.section_text)
                node.summary = _first_sentences(node.page, n=1, limit=300) or node.title
            return "section_page"

        if node.kind == KIND_FOLDER and knowledge:
            doc_cards = [
                c.page for c in children if c.kind == KIND_DOC and c.page
            ][:max_children_briefs]
            if doc_cards:
                node.page = page_writer.digest_knowledge(node.title, doc_cards)
            node.summary = (
                f"{node.title}: " + ", ".join(c.title for c in children[:12])
            )[:300]
            return "folder_digest"

        node.summary = (
            f"{node.title}: covers " + ", ".join(c.title for c in children[:12])
        )[:300]
        return "other"

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
        "synthesize_tree(mode=%s): %d leaf, %d doc cards, %d section pages, "
        "%d folder digests (workers=%d)",
        mode,
        counts["leaf"],
        counts["doc_page"],
        counts["section_page"],
        counts["folder_digest"],
        max_workers,
    )


class ExtractiveKnowledge:
    """Cheap section-level page: concatenate child knowledge briefs (no LLM)."""

    def __init__(self, summarizer: Synthesizer):
        self._s = summarizer

    def roll(self, node, children) -> str:
        bullets = []
        if node.section_text:
            lead = _first_sentences(node.section_text, n=2, limit=300)
            if lead:
                bullets.append(lead)
        for c in children:
            if c.summary:
                bullets.append(f"- {c.summary}")
        return "\n".join(bullets).strip()
