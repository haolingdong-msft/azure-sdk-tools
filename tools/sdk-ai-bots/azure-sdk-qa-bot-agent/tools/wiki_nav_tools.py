"""Wiki-tree navigation tools (HTTP client → backend server).

Exposes the **reasoning-based** retrieval path (PageIndex-style) as two tools:

* :meth:`WikiNavTools.wiki_map` — returns a lightweight MAP of the most relevant
  wiki-tree nodes (title path + one-line summary, no body) so the agent can
  reason about *where* the answer lives.
* :meth:`WikiNavTools.wiki_open` — opens the node ids the agent chose and returns
  each node's distilled overview ``page`` + raw ``content`` evidence + resolved
  source ``link`` + ``children``/``related`` handles to drill further.

Unlike a flat chunk search, this lets the agent navigate the document structure
and read whole distilled pages (a more reliable answer surface) while every hop
stays traceable. Both tools delegate to the backend's warm ``WikiTreeService``
and fail soft.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import httpx

from config.app_config import get as cfg
from models.knowledge import WikiMapResult, WikiOpenResult
from tools import tool
from utils.azure_credential import get_credential

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 60.0


async def _post(endpoint: str, audience: str, payload: dict[str, Any], timeout: float) -> Any:
    """POST *payload* to *endpoint* with an AAD bearer token; return parsed JSON or None."""
    scope = f"{audience}/.default"
    try:
        access_token = await get_credential().get_token(scope)
    except Exception:
        logger.exception("wiki nav: failed to acquire AAD token for scope=%s", scope)
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {access_token.token}"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        logger.warning("wiki nav HTTP call to %s failed: %s", endpoint, exc)
        return None
    except Exception:
        logger.exception("wiki nav: unexpected error calling %s", endpoint)
        return None


def _timeout() -> float:
    try:
        return float(cfg("WIKI_QUERY_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS)))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_SECONDS


class WikiNavTools:
    """Wiki-tree navigation tools backed by the backend server."""

    @tool
    async def wiki_map(
        self,
        *,
        query: Annotated[
            str,
            "A natural-language QUESTION or topic. Returns a MAP of the most "
            "relevant wiki-tree nodes — each with an opaque `id`, its heading "
            "path, and a one-line summary (NO body text). Read this map to "
            "decide WHICH nodes to open with `wiki_open`. Prefer opening the "
            "node whose title/summary best matches the question, and its "
            "document-level node (`doc_id`) when you want the cross-document "
            "overview. This is the FIRST step of the wiki path — always follow "
            "it with `wiki_open`.",
        ],
        tenant_id: Annotated[
            str,
            "Tenant identifier from the active skill's [skill_tenant_id] line. "
            "Scopes the map to that tenant's knowledge sources. Empty string = "
            "unscoped.",
        ] = "",
    ) -> WikiMapResult:
        """Map the wiki tree for ``query``: return relevant node handles + summaries."""
        normalised_query = (query or "").strip()
        if not normalised_query:
            return WikiMapResult(entries=[], query="")

        endpoint = cfg("WIKI_MAP_URL", "").strip()
        audience = cfg("WIKI_QUERY_AUDIENCE", "").strip()
        if not endpoint or not audience:
            logger.warning("wiki_map: WIKI_MAP_URL / WIKI_QUERY_AUDIENCE not configured")
            return WikiMapResult(entries=[], query=normalised_query)

        payload: dict[str, Any] = {"query": normalised_query}
        if (tenant_id or "").strip():
            payload["tenant_id"] = tenant_id.strip()

        body = await _post(endpoint, audience, payload, _timeout())
        if body is None:
            return WikiMapResult(entries=[], query=normalised_query)
        try:
            result = WikiMapResult.model_validate(body)
        except Exception:
            logger.exception("wiki_map: backend payload failed validation: %r", body)
            return WikiMapResult(entries=[], query=normalised_query)
        if not result.query:
            result.query = normalised_query
        logger.info(
            "=========Wiki Map========= entries=%d query=%r",
            len(result.entries),
            normalised_query,
        )
        return result

    @tool
    async def wiki_open(
        self,
        *,
        node_ids: Annotated[
            list[str],
            "1–5 node `id` values to open, taken from a previous `wiki_map` "
            "result (or from the `children`/`related`/`doc_id` handles of an "
            "already-opened node). Returns for each node: its distilled "
            "overview `page` (the authoritative, cross-document answer surface), "
            "the raw source `content` (citable evidence), the resolved source "
            "`link`, and `children`/`related` handles you can open next to drill "
            "down or follow cross-document links. Base your answer on the "
            "`page` + `content`, and cite the `link`.",
        ],
        tenant_id: Annotated[
            str,
            "Tenant identifier from the active skill's [skill_tenant_id] line. "
            "Empty string = unscoped.",
        ] = "",
    ) -> WikiOpenResult:
        """Open wiki-tree nodes: distilled pages + evidence + navigation handles."""
        ids = [n for n in (node_ids or []) if n and n.strip()][:5]
        if not ids:
            return WikiOpenResult(nodes=[])

        endpoint = cfg("WIKI_OPEN_URL", "").strip()
        audience = cfg("WIKI_QUERY_AUDIENCE", "").strip()
        if not endpoint or not audience:
            logger.warning("wiki_open: WIKI_OPEN_URL / WIKI_QUERY_AUDIENCE not configured")
            return WikiOpenResult(nodes=[])

        payload: dict[str, Any] = {"node_ids": ids}
        if (tenant_id or "").strip():
            payload["tenant_id"] = tenant_id.strip()

        body = await _post(endpoint, audience, payload, _timeout())
        if body is None:
            return WikiOpenResult(nodes=[])
        try:
            result = WikiOpenResult.model_validate(body)
        except Exception:
            logger.exception("wiki_open: backend payload failed validation: %r", body)
            return WikiOpenResult(nodes=[])
        logger.info("=========Wiki Open========= nodes=%d", len(result.nodes))
        for i, node in enumerate(result.nodes):
            logger.info(
                "Wiki Open [%d] title=%s link=%s page_len=%d children=%d related=%d",
                i + 1,
                node.title_path,
                node.link,
                len(node.page or ""),
                len(node.children),
                len(node.related),
            )
        return result
