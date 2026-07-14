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

    # ------------------------------------------------------------------ #
    # Navigation API (PageIndex-style: map the tree, then open nodes)
    # ------------------------------------------------------------------ #
    def map(
        self,
        query: str,
        embed_query: Callable[[str], list[float]],
        *,
        allowed_sources: set[str] | None = None,
        source_path_filters: dict[str, list[str]] | None = None,
        entry_k: int = 12,
        min_sim: float = 0.15,
    ) -> list[dict]:
        """Return a ranked MAP of relevant nodes (title path + summary, no body).

        This is the "where to look" step: the agent reads the map and decides
        which node ids to :meth:`open`. Bodies are deliberately omitted to keep
        the map compact and force a reasoning step over structure, not cosine.
        """
        query = (query or "").strip()
        if not query or not self._index.ids:
            return []
        qv = np.asarray(embed_query(query), dtype=np.float32)
        scores = self._index.cosine_all(qv)
        if allowed_sources is not None:
            mask = np.isin(self._row_source, list(allowed_sources))
            scores = np.where(mask, scores, -1.0)

        order = np.argsort(scores)[::-1]
        best = float(scores[order[0]]) if len(order) else 1.0
        best = best or 1.0
        out: list[dict] = []
        for pos in order[: entry_k * 3]:
            s = float(scores[pos])
            if s < min_sim:
                break
            nid = self._index.ids[pos]
            node = self._tree.nodes.get(nid)
            if node is None:
                continue
            if not _passes_title_filter(node, source_path_filters):
                continue
            doc = self._ancestor_doc(nid)
            out.append(
                {
                    "id": nid,
                    "title_path": node.title_path(),
                    "summary": node.summary,
                    "source": node.source,
                    "kind": node.kind,
                    "has_children": bool(node.children),
                    "doc_id": doc.id if doc else "",
                    "doc_title": doc.title if doc else "",
                    "score": round(s / best, 4),
                }
            )
            if len(out) >= entry_k:
                break
        return out

    def open(
        self,
        node_ids: list[str],
        *,
        allowed_sources: set[str] | None = None,
        source_path_filters: dict[str, list[str]] | None = None,
        max_children: int = 15,
        max_related: int = 8,
    ) -> list[dict]:
        """Return full node payloads: page + evidence + children/related handles.

        This is the "read + decide where next" step. ``page`` is the distilled
        overview (the authoritative answer surface); ``content`` is the raw
        section evidence for citation; ``children``/``related`` are handles the
        agent can open to drill down or follow cross-document links.
        """
        out: list[dict] = []
        seen: set[str] = set()
        for nid in node_ids:
            if nid in seen:
                continue
            seen.add(nid)
            node = self._tree.nodes.get(nid)
            if node is None:
                continue
            if allowed_sources is not None and node.source and node.source not in allowed_sources:
                continue
            children = [
                {"id": c.id, "title": c.title, "summary": c.summary, "source": c.source}
                for c in self._tree.children_of(nid)[:max_children]
            ]
            related: list[dict] = []
            for rid in node.related[:max_related]:
                r = self._tree.nodes.get(rid)
                if r is None:
                    continue
                if allowed_sources is not None and r.source and r.source not in allowed_sources:
                    continue
                if not _passes_title_filter(r, source_path_filters):
                    continue
                related.append(
                    {"id": r.id, "title": r.title_path(), "summary": r.summary, "source": r.source}
                )
            out.append(
                {
                    "id": nid,
                    "title_path": node.title_path(),
                    "source": node.source,
                    "rel_title": node.rel_title,
                    "page": node.page,
                    "content": node.section_text,
                    "children": children,
                    "related": related,
                }
            )
        return out


def _passes_title_filter(node, source_path_filters: dict[str, list[str]] | None) -> bool:
    """True unless the node's source has title terms none of which match rel_title."""
    if not source_path_filters:
        return True
    terms = source_path_filters.get(node.source)
    if not terms:
        return True
    hay = (node.rel_title or "").lower()
    return any(t.lower() in hay for t in terms)
