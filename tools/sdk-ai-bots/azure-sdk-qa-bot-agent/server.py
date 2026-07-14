"""Azure SDK QA Bot — Backend Server.

The backend server that the Teams App communicates with.
Calls the hosted Chat Agent via the Azure AI Foundry SDK (azure-ai-projects)
and handles feedback through a local workflow.
"""

import asyncio
import contextlib
import logging
import os
import sys
import time
from contextvars import ContextVar
from contextlib import asynccontextmanager
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv(override=False)

from fastapi import FastAPI, Request
from models.chat import ChatRequest, ChatResponse
from models.conversation import ConversationMessage, SaveConversationMessageResponse
from models.feedback import FeedbackRequest, FeedbackResponse
from models.intention import IntentionRequest, IntentionResponse
from models.knowledge import Reference, WikiQueryRequest, WikiSearchResult
from models.knowledge_retrieve import KnowledgeRetrieveResponse, KnowledgeRetrieveRequest
from services.chat_service import ChatService
from services.conversation_service import ConversationService
from services.feedback_service import FeedbackService
from services.intention_service import IntentionService
from services.knowledge_service import KnowledgeService
from services.thread_memory_service import ThreadMemoryService
from utils.azure_ai_foundry import close_clients
from utils.azure_cosmosdb import close_cosmos_client
from _version import VERSION
from utils.azure_credential import close_credential
from utils.azure_storage import close_storage_client
from utils.azure_monitor import (
    configure_metrics,
    record_chat_request,
    record_chat_duration,
)
from utils.background_tasks import BackgroundTaskTracker
import config.app_config as app_config
from config.tenant_config import TenantID
import uvicorn

_request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="system")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx_var.get() or "system"
        return True


def _configure_logging() -> None:
    """Configure process-wide logging for backend debug and local runs."""
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [RequestID: %(request_id)s] %(name)s: %(message)s"
    )
    request_id_filter = _RequestIdFilter()

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.addFilter(request_id_filter)
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    else:
        root.setLevel(logging.INFO)
        for handler in root.handlers:
            handler.setFormatter(formatter)
            handler.addFilter(request_id_filter)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(logger_name)
        for handler in uvicorn_logger.handlers:
            handler.setFormatter(formatter)
            handler.addFilter(request_id_filter)

    # Suppress noisy Azure SDK HTTP / credential / telemetry loggers
    for noisy in (
        "azure.core.pipeline.policies.http_logging_policy",
        "azure.cosmos",
        "azure.monitor.opentelemetry",
        "azure.monitor.opentelemetry.exporter",
        "azure.monitor.opentelemetry.exporter.export",
        "azure.monitor.opentelemetry.exporter.export._base",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger(__name__)
configure_metrics()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle for the FastAPI app."""
    logger.info("Backend server starting up")
    await app_config.init()

    # Pre-warm the wiki-tree retrieval snapshot (tree.json + embeddings) off
    # the request path so the first /wiki/query doesn't pay the cold-load tax.
    # Tolerates failure — queries fall back to a lazy load.
    async def _warm_wiki():
        try:
            from utils.knowledge_wiki import get_wiki_service

            service = get_wiki_service()
            if not service.enabled:
                logger.info("Wiki service disabled (STORAGE_WIKI_OUTPUT_CONTAINER unset)")
                return
            logger.info("Pre-warming wiki-tree snapshot at startup")
            status = await service.reload()
            if not status.get("loaded"):
                logger.error("Wiki pre-warm did not load a snapshot: %s", status)
            else:
                logger.info("Wiki pre-warm complete: %s", status)
        except Exception:
            logger.error("Wiki pre-warm failed; first query will lazy load", exc_info=True)

    warm_task = asyncio.create_task(_warm_wiki())

    # Periodic poll: hot-swap when latest.json build_id changes.
    async def _poll_wiki_manifest():
        try:
            interval = float(os.environ.get("WIKI_RELOAD_POLL_SECONDS", "86400"))
        except (TypeError, ValueError):
            interval = 86400.0
        if interval <= 0:
            return
        from utils.knowledge_wiki import get_wiki_service

        service = get_wiki_service()
        if not service.enabled:
            return
        while True:
            try:
                await asyncio.sleep(interval)
                await service.reload_if_changed()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Wiki scheduled manifest poll failed")

    poll_task = asyncio.create_task(_poll_wiki_manifest())

    try:
        yield
    finally:
        for task in (warm_task, poll_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        # Cleanup SDK clients on shutdown
        logger.info("Backend server shutting down")
        await BackgroundTaskTracker.instance().shutdown()
        await close_clients()
        await close_cosmos_client()
        await close_storage_client()
        await close_credential()


app = FastAPI(title="Azure SDK QA Bot Backend", version=VERSION, lifespan=lifespan)


@app.get("/ping")
async def ping():
    """Health check endpoint used by App Service and the deploy pipeline."""
    return {"status": "ok", "version": VERSION}


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid4())
    token = _request_id_ctx_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        _request_id_ctx_var.reset(token)
    response.headers["x-request-id"] = request_id
    return response


_chat_service = ChatService()
_conversation_service = ConversationService()
_feedback_service = FeedbackService()
_intention_service = IntentionService()
_knowledge_service = KnowledgeService()
_thread_memory_service = ThreadMemoryService()


@app.post(
    "/completion", response_model=ChatResponse
)  # backwards compatibility for old endpoint
@app.post("/agent/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest):
    """Process a chat request through the chat service."""
    # backwards azure-sdk-qa-bot tenant ID
    if req.tenant_id == TenantID.AZURE_SDK_QA_BOT:
        req.tenant_id = TenantID.TYPESPEC_CHANNEL_QA_BOT
    tenant = req.tenant_id.value
    logger.info(
        "Chat request: tenant=%s, conversation=%s, user=%s, message=%s",
        tenant,
        req.conversation_id,
        req.message.user_name or req.message.user_id,
        req.message.content[:200],
    )
    record_chat_request(tenant)
    start = time.perf_counter()
    try:
        resp = await _chat_service.chat(req)
        elapsed = time.perf_counter() - start
        record_chat_duration(tenant, elapsed, success=True)
        logger.info(
            "Chat response: %s",
            resp.model_dump_json(exclude={"full_context"}),
        )
        return resp
    except Exception:
        elapsed = time.perf_counter() - start
        record_chat_duration(tenant, elapsed, success=False)
        logger.error(
            "Chat failed: tenant=%s, conversation=%s, elapsed=%.2fs",
            tenant,
            req.conversation_id,
            elapsed,
            exc_info=True,
        )
        raise


@app.post(
    "/feedback", response_model=FeedbackResponse
)  # backwards compatibility for old endpoint
@app.post("/agent/feedback", response_model=FeedbackResponse)
async def handle_feedback(req: FeedbackRequest):
    """Process user feedback through the feedback workflow."""
    logger.info(
        "Feedback request: tenant=%s, link=%s, reaction=%s",
        req.tenant_id,
        req.link,
        req.reaction,
    )
    return await _feedback_service.process(req)


@app.post("/message/intention", response_model=IntentionResponse)
async def handle_intention(req: IntentionRequest):
    """Classify whether the bot should auto-reply to a message."""
    logger.info(
        "Intention request: conversation=%s, message=%s",
        req.conversation_id,
        req.message.content[:200],
    )
    return await _intention_service.classify(req)


@app.post("/conversation/save", response_model=SaveConversationMessageResponse)
async def save_conversation(req: ConversationMessage):
    """Save a conversation message and trigger background tenant memory update."""
    logger.info(
        "Save conversation request: tenant=%s, conversation=%s, message=%s",
        req.tenant_id,
        req.conversation_id,
        req.content[:200],
    )
    await _conversation_service.save_conversation(req)
    # Fire-and-forget background task to feed the thread to tenant memory
    BackgroundTaskTracker.instance().track(
        asyncio.create_task(_update_thread_memory(req))
    )
    return SaveConversationMessageResponse()

@app.post("/knowledge/retrieve", response_model=KnowledgeRetrieveResponse)
async def retrieve_knowledge(req: KnowledgeRetrieveRequest):
    """Retrieve knowledge for a request using search_knowledge_base tool."""
    logger.info(
        "Retrieve knowledge request: tenant=%s, message=%s",
        req.tenant_id,
        req.query[:200],
    )
    try:
        resp = await _knowledge_service.retrieve(req)
        logger.info(
            "Knowledge retrieval completed: tenant=%s, knowledge=%d",
            req.tenant_id,
            len(resp.knowledge_list) if resp.knowledge_list else 0,
        )
        return resp
    except Exception:
        logger.error(
            "Knowledge retrieval failed: tenant=%s",
            req.tenant_id,
            exc_info=True,
        )
        raise


async def _update_thread_memory(message: ConversationMessage) -> None:
    """Background task: query full thread and update tenant memory store."""
    try:
        thread_messages = await _conversation_service.get_thread_messages(message)
        await _thread_memory_service.process_thread_update(message, thread_messages)
    except Exception:
        logger.warning(
            "Background thread memory update failed for message=%s",
            message.id,
            exc_info=True,
        )


# --------------------------------------------------------------------------- #
# Wiki-tree query endpoint (called by chat_agent's search_wiki tool)
# --------------------------------------------------------------------------- #
# The chat agent runs in a fresh Foundry sandbox per session; loading the
# wiki snapshot + node embeddings there each time would be wasteful. Instead
# the agent POSTs here and the backend's lifespan pre-warms a single
# WikiTreeService for the pod, so each call resolves in ~1s (one AOAI
# embedding + one matmul + tree/link expansion). Authentication is delegated
# to App Service EasyAuth at the ingress (same as other backend endpoints).


@app.post("/wiki/query", response_model=WikiSearchResult)
async def wiki_query(req: WikiQueryRequest) -> WikiSearchResult:
    """Run wiki-tree retrieval and return KB-style references.

    Scopes to the tenant's ``KnowledgeSource`` folders (and per-source title
    filters) exactly like ``search_knowledge_base``. Never raises 5xx for
    query-side failures so the chat agent can degrade gracefully.
    """
    from config.tenant_config import TenantID, get_tenant_config
    from utils.knowledge_wiki import get_wiki_service

    normalised_query = (req.query or "").strip()
    if not normalised_query:
        return WikiSearchResult(references=[], query="")

    service = get_wiki_service()
    if not service.enabled:
        return WikiSearchResult(references=[], query=normalised_query)

    # Resolve tenant → allowed source folders + per-source title terms.
    allowed_source_folders: set[str] | None = None
    source_path_filters: dict[str, list[str]] | None = None
    tenant_id_raw = (req.tenant_id or "").strip()
    if tenant_id_raw:
        try:
            tenant_enum = TenantID(tenant_id_raw)
        except ValueError:
            logger.warning("Wiki query: unknown tenant_id %r — unscoped", tenant_id_raw)
            tenant_enum = None
        if tenant_enum is not None:
            tenant_config = get_tenant_config(tenant_enum)
            if tenant_config and tenant_config.sources:
                allowed_source_folders = {
                    src.name for src in tenant_config.sources if src.name
                }
                source_path_filters = {
                    name: _split_title_terms(odata)
                    for name, odata in tenant_config.source_filter.items()
                    if _split_title_terms(odata)
                } or None

    try:
        refs = await service.search(
            normalised_query,
            allowed_source_folders=allowed_source_folders,
            source_path_filters=source_path_filters,
        )
    except Exception:
        logger.exception("Wiki query failed for %r", normalised_query)
        return WikiSearchResult(references=[], query=normalised_query)

    return WikiSearchResult(references=refs, query=normalised_query)


def _split_title_terms(odata: str) -> list[str]:
    """Best-effort extraction of match terms from a KB ``source_filter`` clause.

    Tenant ``source_filter`` values are OData ``search.ismatch(...,'title')``
    clauses; we only need the quoted match terms to narrow wiki refs by their
    ``rel_title``. Returns the single-quoted tokens found in *odata*.
    """
    import re

    if not odata:
        return []
    return [m for m in re.findall(r"'([^']+)'", odata) if m and m != "title"]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089)
