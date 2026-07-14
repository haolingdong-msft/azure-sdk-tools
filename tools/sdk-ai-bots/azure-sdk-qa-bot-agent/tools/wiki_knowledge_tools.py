"""Wiki-tree retrieval tool (HTTP client → backend server).

Exposes a single tool, :meth:`WikiKnowledgeTools.search_wiki`, that retrieves
wiki-tree-grounded references for a natural-language query by delegating to the
backend's warm ``/wiki/query`` endpoint (so the agent sandbox never loads the
snapshot itself). Output mirrors ``search_knowledge_base`` so the two reference
sets fuse uniformly.
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

_DEFAULT_TIMEOUT_SECONDS = 60.0


class WikiKnowledgeTools:
    """Wiki-tree retrieval tools backed by the backend server."""

    @tool
    async def search_wiki(
        self,
        *,
        query: Annotated[
            str,
            "A single natural-language QUERY or topic to ground. Returns a list "
            "of references (title, link, snippet): the most relevant source "
            "sections navigated from a table-of-contents tree, PLUS a rolled-up "
            "cross-document 'overview' page that synthesises how the pieces "
            "relate (titled '… (overview)'). Output shape mirrors "
            "search_knowledge_base. Phrase the input as a question or topic, not "
            "a keyword list. Use in parallel with search_knowledge_base — the "
            "wiki path adds structural navigation and cross-document synthesis "
            "the flat KB chunks cannot.",
        ],
        tenant_id: Annotated[
            str,
            "Optional tenant identifier (e.g. 'typespec_channel_qa_bot'). Read it "
            "from the active skill's [skill_tenant_id] line. When set to a known "
            "tenant the backend restricts retrieval to that tenant's "
            "KnowledgeSource folders — same scoping as search_knowledge_base. "
            "Pass an empty string for unscoped retrieval.",
        ] = "",
    ) -> WikiSearchResult:
        """Retrieve wiki-tree references for ``query`` via the backend.

        Fails soft: returns an empty ``references`` list when the query is
        blank, the endpoint is not configured, the HTTP call fails, or nothing
        matched — the chat agent falls back to other tools in that case.
        """
        normalised_query = (query or "").strip()
        if not normalised_query:
            return WikiSearchResult(references=[], query="")

        endpoint = cfg("WIKI_QUERY_URL", "").strip()
        audience = cfg("WIKI_QUERY_AUDIENCE", "").strip()
        if not endpoint or not audience:
            logger.warning(
                "search_wiki: WIKI_QUERY_URL / WIKI_QUERY_AUDIENCE not configured — "
                "returning empty result"
            )
            return WikiSearchResult(references=[], query=normalised_query)

        try:
            timeout = float(cfg("WIKI_QUERY_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS)))
        except (TypeError, ValueError):
            timeout = _DEFAULT_TIMEOUT_SECONDS

        scope = f"{audience}/.default"
        try:
            access_token = await get_credential().get_token(scope)
        except Exception:
            logger.exception("search_wiki: failed to acquire AAD token for scope=%s", scope)
            return WikiSearchResult(references=[], query=normalised_query)

        normalised_tenant = (tenant_id or "").strip()
        logger.info(
            "Posting wiki query to %s (timeout=%.1fs, tenant_id=%r)",
            endpoint,
            timeout,
            normalised_tenant,
        )

        payload: dict[str, Any] = {"query": normalised_query}
        if normalised_tenant:
            payload["tenant_id"] = normalised_tenant

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
            logger.warning("search_wiki HTTP call failed for query=%r: %s", normalised_query, exc)
            return WikiSearchResult(references=[], query=normalised_query)
        except Exception:
            logger.exception("search_wiki: unexpected error for query=%r", normalised_query)
            return WikiSearchResult(references=[], query=normalised_query)

        try:
            result = WikiSearchResult.model_validate(body)
        except Exception:
            logger.exception("search_wiki: backend payload failed validation: %r", body)
            return WikiSearchResult(references=[], query=normalised_query)

        if not result.query:
            result.query = normalised_query

        logger.info("=========Wiki Result========= references=%d", len(result.references))
        for i, ref in enumerate(result.references):
            logger.info("Wiki Reference [%d] title=%s, link=%s", i + 1, ref.title, ref.link)
        logger.info("===================================== query=%r", normalised_query)

        return result
