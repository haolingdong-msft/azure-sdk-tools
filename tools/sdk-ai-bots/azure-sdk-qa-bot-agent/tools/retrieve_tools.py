"""Coarse-to-fine retrieval tool (HTTP client → backend server).

Exposes a single :meth:`RetrieveTools.retrieve` tool that delegates to the
backend's ``/retrieve`` endpoint, where the tree-routed hybrid pipeline runs
server-side: the wiki tree routes to the relevant documents, KB search fills
them in with wide recall, and the routed documents' distilled overview pages are
attached as synthesis. The agent gets one clean, wide + structured reference set
and does not orchestrate two competing retrievers. Fails soft.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import httpx

from config.app_config import get as cfg
from models.knowledge import WikiSearchResult
from tools import tool
from utils.azure_credential import get_credential

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 90.0


class RetrieveTools:
    """Single-call coarse-to-fine retrieval tool backed by the backend server."""

    @tool
    async def retrieve(
        self,
        *,
        query: Annotated[
            str,
            "A full-sentence QUESTION restating what the user needs (resolve "
            "pronouns, keep concrete names/versions/error text). Returns one "
            "merged, relevance-ranked reference set that combines: (a) wide "
            "knowledge-base recall focused on the documents most relevant to the "
            "question, and (b) rolled-up cross-document 'overview' pages (titled "
            "'… (overview)') that synthesise how the pieces relate. This is your "
            "PRIMARY grounding tool — call it once per domain question. Base your "
            "answer on the returned `content` and cite each reference's `link`; "
            "lead with the direct answer and cover every specific fact the "
            "question needs.",
        ],
        tenant_id: Annotated[
            str,
            "Tenant identifier from the active skill's [skill_tenant_id] line. "
            "Scopes retrieval to that tenant's knowledge sources. Empty string = "
            "unscoped.",
        ] = "",
    ) -> WikiSearchResult:
        """Retrieve a merged, tree-routed hybrid reference set for ``query``."""
        normalised_query = (query or "").strip()
        if not normalised_query:
            return WikiSearchResult(references=[], query="")

        endpoint = cfg("RETRIEVE_URL", "").strip()
        audience = cfg("WIKI_QUERY_AUDIENCE", "").strip()
        if not endpoint or not audience:
            logger.warning("retrieve: RETRIEVE_URL / WIKI_QUERY_AUDIENCE not configured")
            return WikiSearchResult(references=[], query=normalised_query)

        try:
            timeout = float(cfg("RETRIEVE_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS)))
        except (TypeError, ValueError):
            timeout = _DEFAULT_TIMEOUT_SECONDS

        scope = f"{audience}/.default"
        try:
            access_token = await get_credential().get_token(scope)
        except Exception:
            logger.exception("retrieve: failed to acquire AAD token for scope=%s", scope)
            return WikiSearchResult(references=[], query=normalised_query)

        payload: dict[str, Any] = {"query": normalised_query}
        if (tenant_id or "").strip():
            payload["tenant_id"] = tenant_id.strip()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {access_token.token}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            logger.warning("retrieve HTTP call failed for query=%r: %s", normalised_query, exc)
            return WikiSearchResult(references=[], query=normalised_query)
        except Exception:
            logger.exception("retrieve: unexpected error for query=%r", normalised_query)
            return WikiSearchResult(references=[], query=normalised_query)

        try:
            result = WikiSearchResult.model_validate(body)
        except Exception:
            logger.exception("retrieve: backend payload failed validation: %r", body)
            return WikiSearchResult(references=[], query=normalised_query)

        if not result.query:
            result.query = normalised_query
        logger.info(
            "=========Retrieve Result========= references=%d query=%r",
            len(result.references),
            normalised_query,
        )
        for i, ref in enumerate(result.references):
            logger.info("Retrieve Reference [%d] title=%s, link=%s", i + 1, ref.title, ref.link)
        return result
