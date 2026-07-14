"""Wiki-tree retrieval — the query-side navigation.

One traversal, three moves (see the design):

1. **Entry** — embed the query and rank content nodes by cosine similarity
   (numpy matmul over the :class:`~.embeddings.EmbeddingIndex`), optionally
   scoped to a tenant's source folders.
2. **Expansion** — from the top entry nodes, take the node's own section
   evidence and follow 1 hop of ``related`` cross-links for associative recall.
3. **Synthesis** — surface the most relevant ancestor-document ``page`` (the
   rolled-up cross-document overview), playing the role GraphRAG's community
   report does — but pre-computed and human-legible.

Results are returned as :class:`WikiReference` objects, relevance-ranked and
capped at ``top_k``, mirroring the KB/graph reference contract. Azure-free: it
takes an ``embed_query`` callable, so the same code runs in the offline eval and
in the backend service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .embeddings import EmbeddingIndex
from .models import KIND_DOC, WikiTree

logger = logging.getLogger(__name__)


@dataclass
class WikiReference:
    """A retrieval hit, shaped like the KB/graph ``Reference``."""

    title: str
    source: str
    rel_title: str  # folder-relative "#"-encoded path; backend resolves the link
    content: str
    score: float
    kind: str  # "section" | "synthesis"
    source_path: str = ""


class WikiRetriever:
    """Loads a wiki-tree snapshot and answers queries over it."""

    def __init__(self, tree: WikiTree, index: EmbeddingIndex):
        self._tree = tree
        self._index = index
        # Per-row source folder, for fast tenant scoping via boolean mask.
        self._row_source = np.array(
            [tree.nodes[nid].source if nid in tree.nodes else "" for nid in index.ids]
        )

    def search(
        self,
        query: str,
        embed_query: Callable[[str], list[float]],
        *,
        allowed_sources: set[str] | None = None,
        top_k: int = 8,
        entry_k: int = 6,
        synthesis_k: int = 2,
        min_sim: float = 0.2,
    ) -> list[WikiReference]:
        """Retrieve ranked source-section refs + synthesis-page refs for *query*."""
        query = (query or "").strip()
        if not query or not self._index.ids:
            return []

        qv = np.asarray(embed_query(query), dtype=np.float32)
        scores = self._index.cosine_all(qv)  # [N]

        if allowed_sources is not None:
            mask = np.isin(self._row_source, list(allowed_sources))
            scores = np.where(mask, scores, -1.0)

        # Top entry nodes by score (descending), above threshold.
        order = np.argsort(scores)[::-1]
        entry: list[tuple[float, str]] = []
        for pos in order[: max(entry_k * 3, entry_k)]:
            s = float(scores[pos])
            if s < min_sim:
                break
            entry.append((s, self._index.ids[pos]))
            if len(entry) >= entry_k:
                break
        if not entry:
            return []
        best = entry[0][0] or 1.0

        # --- 2. Expansion: entry nodes + 1-hop related ------------------
        picked: dict[str, float] = {}
        for score, nid in entry:
            picked[nid] = max(picked.get(nid, 0.0), score)
            for rid in self._tree.nodes.get(nid).related if nid in self._tree.nodes else []:
                if rid in self._tree.nodes and (
                    allowed_sources is None
                    or self._tree.nodes[rid].source in allowed_sources
                ):
                    picked[rid] = max(picked.get(rid, 0.0), score * 0.85)

        section_refs: list[WikiReference] = []
        seen_sections: set[tuple[str, str]] = set()
        for nid, score in sorted(picked.items(), key=lambda kv: -kv[1]):
            node = self._tree.nodes[nid]
            evidence = node.section_text or node.page
            if not evidence:
                continue
            dedupe_key = (node.source_path, node.title_path())
            if dedupe_key in seen_sections:
                continue
            seen_sections.add(dedupe_key)
            section_refs.append(
                WikiReference(
                    title=node.title_path(),
                    source=node.source,
                    rel_title=node.rel_title,
                    content=evidence,
                    score=round(score / best, 4),
                    kind="section",
                    source_path=node.source_path,
                )
            )
            if len(section_refs) >= top_k:
                break

        # --- 3. Synthesis: best ancestor-doc pages ----------------------
        synthesis_refs = self._synthesis_refs([nid for _, nid in entry], synthesis_k)

        logger.info(
            "wiki search: query=%r sections=%d synthesis=%d",
            query[:80],
            len(section_refs),
            len(synthesis_refs),
        )
        return section_refs + synthesis_refs

    def _synthesis_refs(self, entry_ids: list[str], k: int) -> list[WikiReference]:
        if k <= 0:
            return []
        out: list[WikiReference] = []
        seen: set[str] = set()
        for nid in entry_ids:
            doc = self._ancestor_doc(nid)
            if doc is None or doc.id in seen or not doc.page:
                continue
            seen.add(doc.id)
            out.append(
                WikiReference(
                    title=f"{doc.title} (overview)",
                    source=doc.source,
                    rel_title=doc.rel_title,
                    content=doc.page,
                    score=1.0,
                    kind="synthesis",
                    source_path=doc.source_path,
                )
            )
            if len(out) >= k:
                break
        return out

    def _ancestor_doc(self, nid: str):
        cur = self._tree.nodes.get(nid)
        while cur is not None:
            if cur.kind == KIND_DOC:
                return cur
            cur = self._tree.nodes.get(cur.parent) if cur.parent else None
        return None
