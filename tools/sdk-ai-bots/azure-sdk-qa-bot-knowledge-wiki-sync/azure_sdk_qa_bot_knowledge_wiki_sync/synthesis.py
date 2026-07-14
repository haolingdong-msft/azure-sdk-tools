"""Bottom-up synthesis: fill node summaries and rolled-up wiki pages.

Walks the ToC tree in **post-order** (children before parents) and, using the
pluggable :class:`~.llm.Synthesizer`:

* **leaf sections** get a one-line ``summary`` distilled from their own text
  (the raw text stays in ``section_text`` as citable evidence);
* **internal nodes** (sections with children, and documents) get a ``page`` —
  a cross-document roll-up synthesised from their children's summaries — and a
  ``summary`` derived from that page's first sentence (no extra model call).

The tree hierarchy *is* the topic clustering, and the per-level roll-up *is* the
hierarchical equivalent of GraphRAG's community reports — obtained without
entity extraction or community detection.
"""

from __future__ import annotations

import logging

from .llm import Synthesizer, _first_sentences
from .models import KIND_DOC, KIND_FOLDER, KIND_ROOT, KIND_SECTION, WikiTree

logger = logging.getLogger(__name__)


def _post_order(tree: WikiTree) -> list[str]:
    """Return node ids in post-order (each node after all its descendants)."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(nid: str) -> None:
        if nid in seen or nid not in tree.nodes:
            return
        seen.add(nid)
        for child in tree.nodes[nid].children:
            visit(child)
        order.append(nid)

    for root in tree.roots:
        visit(root)
    return order


def synthesize_tree(
    tree: WikiTree,
    synthesizer: Synthesizer,
    *,
    max_children_briefs: int = 40,
) -> None:
    """Populate ``summary`` and ``page`` on every node, in place, bottom-up."""
    order = _post_order(tree)
    leaves = internal = 0

    for nid in order:
        node = tree.nodes[nid]
        children = tree.children_of(nid)

        if not children:
            # Leaf: summarise own text; evidence stays in section_text.
            node.summary = synthesizer.summarize(node.title, node.section_text)
            leaves += 1
            continue

        child_briefs = [
            f"{c.title}: {c.summary}".strip().rstrip(":").strip()
            for c in children[:max_children_briefs]
            if (c.summary or c.title)
        ]

        if node.kind in (KIND_SECTION, KIND_DOC):
            node.page = synthesizer.roll_up(node.title, child_briefs, node.section_text)
            node.summary = _first_sentences(node.page, n=1, limit=300) or node.title
            internal += 1
        else:
            # Folder / root: cheap extractive summary (a topic list); no page.
            node.summary = (
                f"{node.title}: covers "
                + ", ".join(c.title for c in children[:12])
            )[:300]

    tree.stats["synthesized"] = {"leaves": leaves, "internal_pages": internal}
    logger.info(
        "synthesize_tree: %d leaf summaries, %d internal roll-up pages",
        leaves,
        internal,
    )
