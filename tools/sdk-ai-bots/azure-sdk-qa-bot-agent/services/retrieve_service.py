"""Coarse-to-fine retrieval service — the tree-routed hybrid (Approach A).

Fuses the KB path and the wiki tree into a **single server-side pipeline** so the
chat agent calls one tool instead of orchestrating two competing retrievers:

1. **Route (PageIndex).** The wiki tree ``map`` reasons over the table-of-contents
   to pick the source folders / documents the answer lives in — structural
   routing, not similarity.
2. **Recall (KB).** ``search_knowledge_base`` runs its wide dense/agentic recall,
   optionally **scoped to the routed folders** so it stays wide *within* the
   right scope but drops cross-source noise.
3. **Synthesise (WeKnora).** The routed documents' distilled overview ``page``s
   are attached as synthesis references — the connective, cross-document tissue
   the flat KB chunks cannot produce.

The three reinforce instead of compete: the tree says *where*, KB fills it in
*wide*, the page *connects* it. Everything is merged server-side into one
relevance-ranked :class:`Reference` list.
"""

from __future__ import annotations

import logging

import config.app_config as app_config
from models.knowledge import Reference
from tools.knowledge_tools import KnowledgeTools
from utils.knowledge_wiki import get_wiki_service

logger = logging.getLogger(__name__)


class RetrieveService:
    """Server-side coarse-to-fine retrieval (tree route → KB recall → page synthesis)."""

    def __init__(self) -> None:
        self._kb = KnowledgeTools()

    @staticmethod
    def _route_kb_enabled() -> bool:
        return app_config.get("WIKI_ROUTE_KB", "true").lower() == "true"

    @staticmethod
    def _max_route_folders() -> int:
        try:
            return int(app_config.get("WIKI_ROUTE_MAX_FOLDERS", "6"))
        except (TypeError, ValueError):
            return 6

    @staticmethod
    def _overview_top() -> int:
        try:
            return int(app_config.get("WIKI_ROUTE_OVERVIEW_TOP", "2"))
        except (TypeError, ValueError):
            return 2

    @staticmethod
    def _domain_top() -> int:
        try:
            return int(app_config.get("WIKI_ROUTE_DOMAIN_TOP", "1"))
        except (TypeError, ValueError):
            return 1

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        *,
        allowed_source_folders: set[str] | None = None,
        source_path_filters: dict[str, list[str]] | None = None,
    ) -> list[Reference]:
        """Run the coarse-to-fine pipeline and return one merged reference set."""
        query = (query or "").strip()
        if not query:
            return []

        wiki = get_wiki_service()

        # -- 1. Route: map the tree to relevant folders + overview doc handles --
        routed_folders: list[str] = []
        overview_doc_ids: list[str] = []
        if wiki.enabled:
            try:
                entries = await wiki.map_query(
                    query,
                    allowed_source_folders=allowed_source_folders,
                    source_path_filters=source_path_filters,
                    entry_k=12,
                )
            except Exception:
                logger.warning("retrieve: wiki route failed; KB-only fallback", exc_info=True)
                entries = []
            seen_folders: list[str] = []
            for e in entries:
                src = e.get("source") or ""
                if src and src not in seen_folders:
                    seen_folders.append(src)
                doc_id = e.get("doc_id") or ""
                if doc_id and doc_id not in overview_doc_ids:
                    overview_doc_ids.append(doc_id)
            routed_folders = seen_folders[: self._max_route_folders()]

        # -- 2. Recall: KB search, scoped to routed folders when confident -----
        kb_sources: list[str] | None = None
        if self._route_kb_enabled() and routed_folders:
            # Only route when the tree resolved to a *subset* of the tenant's
            # sources — otherwise routing adds nothing and risks over-narrowing.
            kb_sources = routed_folders
        kb_refs: list[Reference] = []
        try:
            kb_result = await self._kb.search_knowledge_base(
                queries=[query],
                tenant_id=tenant_id,
                sources=kb_sources,
                service_type=None,
                search_mode="quick",
            )
            kb_refs = list(kb_result.results or [])
        except Exception:
            logger.warning("retrieve: KB search failed", exc_info=True)
            # If a scoped KB search failed, retry unscoped once.
            if kb_sources is not None:
                try:
                    kb_result = await self._kb.search_knowledge_base(
                        queries=[query], tenant_id=tenant_id, sources=None,
                        service_type=None, search_mode="quick",
                    )
                    kb_refs = list(kb_result.results or [])
                except Exception:
                    logger.exception("retrieve: unscoped KB retry failed")

        # -- 3. Synthesise: domain digest + routed docs' knowledge cards -------
        knowledge_refs: list[Reference] = []

        # 3a. Domain knowledge — the top routed folder's rolled-up digest, so the
        # agent gets broad "what an expert knows about this area" background.
        if wiki.enabled and routed_folders and self._domain_top() > 0:
            try:
                digests = await wiki.domain_digests(routed_folders[: self._domain_top()])
            except Exception:
                logger.warning("retrieve: domain digest failed", exc_info=True)
                digests = {}
            for folder in routed_folders[: self._domain_top()]:
                page = (digests.get(folder) or "").strip()
                if page:
                    knowledge_refs.append(
                        Reference(
                            title=f"Domain knowledge: {folder}",
                            source=folder,
                            link="",
                            content=page,
                            score=1.0,
                        )
                    )

        # 3b. Document knowledge cards — dense facts for the routed documents.
        if wiki.enabled and overview_doc_ids:
            try:
                opened = await wiki.open_nodes(
                    overview_doc_ids[: self._overview_top() * 2],
                    allowed_source_folders=allowed_source_folders,
                    source_path_filters=source_path_filters,
                )
            except Exception:
                logger.warning("retrieve: knowledge card open failed", exc_info=True)
                opened = []
            card_count = 0
            for node in opened:
                page = (node.get("page") or "").strip()
                if not page:
                    continue
                title = node.get("title_path") or "Knowledge"
                knowledge_refs.append(
                    Reference(
                        title=f"{title} (knowledge)",
                        source=node.get("source") or "wiki",
                        link=node.get("link") or "",
                        content=page,
                        score=1.0,
                    )
                )
                card_count += 1
                if card_count >= self._overview_top():
                    break

        merged = _merge(kb_refs, knowledge_refs)
        logger.info(
            "retrieve: routed_folders=%s kb=%d knowledge=%d merged=%d (query=%r)",
            routed_folders,
            len(kb_refs),
            len(knowledge_refs),
            len(merged),
            query[:80],
        )
        return merged


def _merge(kb_refs: list[Reference], knowledge_refs: list[Reference]) -> list[Reference]:
    """Knowledge first (the agent reasons FROM it), then KB evidence.

    Leads with the distilled knowledge (domain digest + document knowledge
    cards) so the agent grounds on understanding, then supplies the KB chunks as
    exact-wording evidence. KB refs are de-duplicated by link; knowledge refs are
    always kept (distinct distilled content) but de-duplicated among themselves.
    """
    out: list[Reference] = []
    seen_knowledge: set[str] = set()
    for ref in knowledge_refs:
        key = (ref.title or "").strip()
        if key in seen_knowledge:
            continue
        seen_knowledge.add(key)
        out.append(ref)
    seen_links: set[str] = set()
    for ref in kb_refs:
        key = (ref.link or ref.title or "").strip()
        if key and key in seen_links:
            continue
        if key:
            seen_links.add(key)
        out.append(ref)
    return out
