"""Cross-link discovery — the lightweight associative graph.

Instead of extracting entities and their co-occurrence (GraphRAG's heavy step),
we draw a few ``related`` "see also" edges between synthesised nodes that live in
**different documents** but are semantically close: one embedding per node + a
blocked nearest-neighbour pass (numpy matmul, bounded memory). This gives the
associative recall a pure hierarchy cannot express, at a fraction of the cost.

The embeddings are returned as an :class:`~.embeddings.EmbeddingIndex` for reuse
as the retrieval entry index, so each node is embedded exactly once.
"""

from __future__ import annotations

import logging

import numpy as np

from .embeddings import EmbeddingIndex
from .llm import Embedder
from .models import KIND_DOC, KIND_SECTION, WikiTree

logger = logging.getLogger(__name__)

_BLOCK = 512  # rows per similarity block (bounds peak memory)


def _embed_text(node) -> str:
    parts = [node.title_path(), node.summary]
    return " — ".join(p for p in parts if p).strip() or node.title


def add_cross_links(
    tree: WikiTree,
    embedder: Embedder,
    *,
    top_k: int = 3,
    min_sim: float = 0.55,
) -> EmbeddingIndex:
    """Embed content nodes, wire ``related`` cross-links, return the index.

    Only ``doc`` / ``section`` nodes participate. Edges never connect two nodes
    from the same document (already tree-linked), so ``related`` captures
    genuine cross-document association.
    """
    node_ids = [
        n.id for n in tree.nodes.values() if n.kind in (KIND_SECTION, KIND_DOC)
    ]
    if not node_ids:
        return EmbeddingIndex(ids=[], matrix=np.zeros((0, 1), dtype=np.float32))

    texts = [_embed_text(tree.nodes[nid]) for nid in node_ids]
    index = EmbeddingIndex.from_rows(node_ids, embedder.embed(texts))
    matrix = index.matrix
    n = len(node_ids)

    # Same-document mask key per row (edges within a doc are skipped).
    doc_of = np.array([tree.nodes[nid].source_path for nid in node_ids])

    total_edges = 0
    if top_k > 0:
        for start in range(0, n, _BLOCK):
            end = min(start + _BLOCK, n)
            sims = matrix[start:end] @ matrix.T  # [block, N]
            for local_i, gi in enumerate(range(start, end)):
                row = sims[local_i]
                row[gi] = -1.0  # exclude self
                row[doc_of == doc_of[gi]] = -1.0  # exclude same-document
                k = min(top_k, n)
                cand = np.argpartition(row, -k)[-k:]
                cand = cand[np.argsort(row[cand])[::-1]]
                related = [node_ids[j] for j in cand if row[j] >= min_sim]
                tree.nodes[node_ids[gi]].related = related
                total_edges += len(related)

    tree.stats["cross_links"] = {"nodes": n, "edges": total_edges}
    logger.info(
        "add_cross_links: %d content nodes, %d related edges (top_k=%d, min_sim=%.2f)",
        n,
        total_edges,
        top_k,
        min_sim,
    )
    return index
