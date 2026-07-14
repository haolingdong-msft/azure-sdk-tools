"""Warm wiki-tree retrieval service (backend singleton).

Loads the current wiki-tree snapshot (``tree.json`` + ``embeddings.npy``) from
the wiki blob container **once per pod** and answers ``/wiki/query`` requests
over it — so each short-lived chat-agent sandbox pays no cold-load cost. Per
query it embeds the query (Azure OpenAI), ranks nodes with one matmul, expands
the tree + cross-links, and resolves each hit back to a KB-style
:class:`Reference` (same link resolution as ``search_knowledge_base``).

Mirrors the design of the GraphRAG ``KnowledgeGraphService`` but is far lighter:
no parquet load, no LocalSearch context builder, no per-report embedding preload.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import config.app_config as app_config
from azure.identity.aio import get_bearer_token_provider
from models.knowledge import Reference
from utils.azure_credential import get_credential
from utils.knowledge_wiki.embeddings import EmbeddingIndex
from utils.knowledge_wiki.models import WikiTree
from utils.knowledge_wiki.retrieval import WikiReference, WikiRetriever

logger = logging.getLogger(__name__)

_MANIFEST_BLOB = "latest.json"
_TREE_BLOB = "tree.json"
_EMB_NPY = "embeddings.npy"
_EMB_IDS = "embedding_ids.json"
_SNIPPET_MAX_CHARS = 3000
_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"


class WikiTreeService:
    """Process-wide singleton owning the loaded wiki-tree snapshot."""

    def __init__(self) -> None:
        self._tree: WikiTree | None = None
        self._retriever: WikiRetriever | None = None
        self._build_id: str | None = None
        self._aoai = None  # lazy AsyncAzureOpenAI

    # -- configuration ---------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(app_config.get("STORAGE_WIKI_OUTPUT_CONTAINER", "").strip())

    def _account_url(self) -> str:
        return app_config.get("STORAGE_BASE_URL", "").strip()

    def _container(self) -> str:
        return app_config.get("STORAGE_WIKI_OUTPUT_CONTAINER", "wiki").strip()

    def _embed_endpoint(self) -> str:
        return (
            app_config.get("WIKI_EMBEDDING_ENDPOINT", "")
            or app_config.get("AOAI_CHAT_COMPLETIONS_ENDPOINT", "")
        ).strip()

    def _embed_deployment(self) -> str:
        return app_config.get("WIKI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small").strip()

    # -- lifecycle -------------------------------------------------------
    def _aoai_client(self):
        if self._aoai is None:
            from openai import AsyncAzureOpenAI

            token_provider = get_bearer_token_provider(get_credential(), _COGNITIVE_SCOPE)
            self._aoai = AsyncAzureOpenAI(
                azure_endpoint=self._embed_endpoint(),
                azure_ad_token_provider=token_provider,
                api_version=app_config.get("WIKI_EMBEDDING_API_VERSION", "2024-10-21"),
            )
        return self._aoai

    async def _container_client(self):
        from azure.storage.blob.aio import ContainerClient

        return ContainerClient(
            self._account_url(), self._container(), credential=get_credential()
        )

    async def _read_blob(self, container, name: str) -> bytes:
        downloader = await container.download_blob(name)
        return await downloader.readall()

    async def reload(self) -> dict[str, Any]:
        """Load the current snapshot from blob, atomically swapping it in."""
        if not self.enabled:
            return {"loaded": False, "reason": "disabled"}
        start = time.monotonic()
        container = await self._container_client()
        try:
            async with container:
                manifest = json.loads(await self._read_blob(container, _MANIFEST_BLOB))
                build_id = manifest["build_id"]
                prefix = manifest.get("prefix", f"snapshots/{build_id}")
                tree = WikiTree.from_json(
                    (await self._read_blob(container, f"{prefix}/{_TREE_BLOB}")).decode("utf-8")
                )
                ids = json.loads(await self._read_blob(container, f"{prefix}/{_EMB_IDS}"))
                npy = await self._read_blob(container, f"{prefix}/{_EMB_NPY}")
        except Exception:
            logger.exception("Wiki snapshot load failed; keeping previous snapshot")
            return {"loaded": self._tree is not None, "reason": "load-error"}

        index = EmbeddingIndex.from_bytes(npy, ids)
        retriever = WikiRetriever(tree, index)

        # Atomic swap.
        self._tree, self._retriever, self._build_id = tree, retriever, build_id
        status = {
            "loaded": True,
            "build_id": build_id,
            "row_counts": {"nodes": len(tree.nodes), "embeddings": index.matrix.shape[0]},
            "elapsed_s": round(time.monotonic() - start, 2),
        }
        logger.info("Wiki snapshot loaded: %s", status)
        return status

    async def reload_if_changed(self) -> dict[str, Any]:
        if not self.enabled:
            return {"loaded": False, "reason": "disabled"}
        container = await self._container_client()
        try:
            async with container:
                manifest = json.loads(await self._read_blob(container, _MANIFEST_BLOB))
        except Exception:
            logger.warning("Wiki manifest poll failed", exc_info=True)
            return {"loaded": self._tree is not None, "reason": "manifest-error"}
        if manifest.get("build_id") == self._build_id:
            return {"loaded": True, "reason": "unchanged", "build_id": self._build_id}
        return await self.reload()

    # -- query -----------------------------------------------------------
    async def _embed_query(self, query: str) -> list[float]:
        client = self._aoai_client()
        resp = await client.embeddings.create(
            model=self._embed_deployment(), input=[query[:8000]]
        )
        return resp.data[0].embedding

    async def search(
        self,
        query: str,
        *,
        allowed_source_folders: set[str] | None = None,
        source_path_filters: dict[str, list[str]] | None = None,
        top_k: int = 8,
    ) -> list[Reference]:
        """Retrieve wiki-tree references for *query*, resolved to KB-style links."""
        if self._retriever is None:
            status = await self.reload()  # lazy load on first query
            if not status.get("loaded"):
                return []

        qvec = await self._embed_query(query)
        wrefs = self._retriever.search(
            query,
            embed_query=lambda _q: qvec,
            allowed_sources=allowed_source_folders,
            top_k=top_k,
        )
        wrefs = _apply_source_path_filters(wrefs, source_path_filters)
        return [_to_reference(w) for w in wrefs]

    # ------------------------------------------------------------------ #
    # Navigation API (PageIndex-style map / open)
    # ------------------------------------------------------------------ #
    async def map_query(
        self,
        query: str,
        *,
        allowed_source_folders: set[str] | None = None,
        source_path_filters: dict[str, list[str]] | None = None,
        entry_k: int = 12,
    ) -> list[dict]:
        """Return a ranked map of relevant nodes (no body text)."""
        if self._retriever is None:
            status = await self.reload()
            if not status.get("loaded"):
                return []
        qvec = await self._embed_query(query)
        return self._retriever.map(
            query,
            embed_query=lambda _q: qvec,
            allowed_sources=allowed_source_folders,
            source_path_filters=source_path_filters,
            entry_k=entry_k,
        )

    async def open_nodes(
        self,
        node_ids: list[str],
        *,
        allowed_source_folders: set[str] | None = None,
        source_path_filters: dict[str, list[str]] | None = None,
    ) -> list[dict]:
        """Open nodes: return distilled page + evidence + children/related, links resolved."""
        if self._retriever is None:
            status = await self.reload()
            if not status.get("loaded"):
                return []
        raw = self._retriever.open(
            node_ids,
            allowed_sources=allowed_source_folders,
            source_path_filters=source_path_filters,
        )
        for node in raw:
            node["link"] = _resolve_link(node.get("source", ""), node.get("rel_title", ""))
            content = node.get("content") or ""
            if len(content) > _SNIPPET_MAX_CHARS:
                node["content"] = content[:_SNIPPET_MAX_CHARS] + "\n… [truncated]"
            page = node.get("page") or ""
            if len(page) > _SNIPPET_MAX_CHARS:
                node["page"] = page[:_SNIPPET_MAX_CHARS] + "\n… [truncated]"
        return raw

    async def domain_digests(self, sources: list[str]) -> dict[str, str]:
        """Return ``{source_folder: domain-knowledge digest}`` for the folders."""
        if self._retriever is None:
            status = await self.reload()
            if not status.get("loaded"):
                return {}
        try:
            return self._retriever.folder_pages(sources)
        except Exception:
            logger.warning("domain_digests failed", exc_info=True)
            return {}


def _resolve_link(source: str, rel_title: str) -> str:
    """Resolve a source + rel_title into a KB-style document URL."""
    if not rel_title:
        return ""
    try:
        from config.tenant_config import get_knowledge_source

        ks = get_knowledge_source(source)
        if ks is not None:
            return ks.get_link(rel_title)
    except Exception:
        logger.debug("link resolution failed for source=%s", source, exc_info=True)
    return rel_title


def _apply_source_path_filters(
    wrefs: list[WikiReference], filters: dict[str, list[str]] | None
) -> list[WikiReference]:
    """Drop refs whose source doesn't satisfy its per-source title terms.

    Mirrors the KB tool's per-source ``search.ismatch(...,'title')`` narrowing:
    for a source with terms, keep a ref only if its ``rel_title`` contains at
    least one term (case-insensitive). Sources without terms are unaffected.
    """
    if not filters:
        return wrefs
    out: list[WikiReference] = []
    for w in wrefs:
        terms = filters.get(w.source)
        if not terms:
            out.append(w)
            continue
        hay = w.rel_title.lower()
        if any(t.lower() in hay for t in terms):
            out.append(w)
    return out


def _to_reference(w: WikiReference) -> Reference:
    """Resolve a :class:`WikiReference` into a KB-style :class:`Reference`."""
    link = w.rel_title
    try:
        from config.tenant_config import get_knowledge_source

        ks = get_knowledge_source(w.source)
        if ks is not None and w.rel_title:
            link = ks.get_link(w.rel_title)
    except Exception:
        logger.debug("link resolution failed for source=%s", w.source, exc_info=True)

    content = w.content or ""
    if len(content) > _SNIPPET_MAX_CHARS:
        content = content[:_SNIPPET_MAX_CHARS] + "\n… [truncated]"

    title = w.title
    if w.kind == "synthesis" and not title.endswith("(overview)"):
        title = f"{title} (overview)"

    return Reference(
        title=title,
        source=w.source or "wiki",
        link=link,
        content=content,
        score=w.score,
    )


_service: WikiTreeService | None = None


def get_wiki_service() -> WikiTreeService:
    """Return the process-wide :class:`WikiTreeService` singleton."""
    global _service
    if _service is None:
        _service = WikiTreeService()
    return _service
