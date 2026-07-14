"""Azure SDK QA Bot — Hosted Chat Agent.

Self-contained entrypoint for the hosted agent container.
Runs as an HTTP server on port 8088 using the Responses protocol.
Deployed to Microsoft Foundry as a container agent.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# sys.path — add the project root so top-level packages
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Environment
load_dotenv(override=False)
os.environ.setdefault("ENABLE_SENSITIVE_DATA", "true")

from agent_framework import Agent
from agent_framework import CompactionProvider
from agent_framework import SkillsProvider
from agent_framework import ToolResultCompactionStrategy
from agent_framework_foundry_hosting import ResponsesHostServer

import config.app_config as app_config
from config.app_config import get as cfg
from tools.knowledge_tools import KnowledgeTools
from tools.wiki_nav_tools import WikiNavTools
from tools.retrieve_tools import RetrieveTools
from tools.web_tools import WebTools
from tools.ado_mcp_tools import create_ado_mcp_tool
from tools.github_mcp_tools import create_github_mcp_tool
from tools.pipeline_tools import PipelineTools
from skills.tenant_skills import create_tenant_skills
from utils.azure_ai_foundry import (
    get_agent_client,
    get_project_client,
)
from utils.azure_memory_store import (
    ensure_user_memory_store,
)
from utils.memory_context_provider import MemoryContextProvider

logger = logging.getLogger(__name__)

# -- Agent configuration constants ----------------------------------------
# Navigation (wiki_map → wiki_open → optional drill-down) needs a few tool-call
# rounds, so allow a slightly higher iteration ceiling than the KB-only path.
MAX_TOOL_CALL_ITERATIONS = 6
MAX_TOOL_CALLS_PER_TURN = 10

# Strong refs to fire-and-forget startup tasks so they aren't GC'd mid-flight.
_background_tasks: set[asyncio.Task] = set()


def _load_instructions(file_path: Path) -> str:
    """Load agent instructions from the instructions markdown file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Agent instructions file not found: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


# Prepended to the base instruction when KNOWLEDGE_MODE=wiki_only, so the agent
# grounds only on the wiki navigation path (the KB tool is not registered).
_WIKI_ONLY_DIRECTIVE = """# KNOWLEDGE MODE: WIKI-ONLY (OVERRIDE — HIGHEST PRIORITY)

The `search_knowledge_base` tool is DISABLED in this configuration. Wherever the
instructions below tell you to call `search_knowledge_base`, you MUST instead
use the wiki navigation tools as your SOLE grounding source:

1. Call `wiki_map` with a full-sentence question to get a map of relevant nodes.
2. Call `wiki_open` on the 2–4 most relevant node ids (include the `doc_id` for
   the cross-document overview) to read their distilled `page` + `content`.
3. If a node's `children`/`related` handles point at something more specific,
   open those too (one more round at most), then answer.

Base your answer on the opened `page` + `content` and cite each node's `link`.
Do NOT mention that `search_knowledge_base` is unavailable — just navigate the
wiki. Every domain question still REQUIRES a `wiki_map` + `wiki_open` sequence."""


# Prepended to the base instruction when KNOWLEDGE_MODE=routed. A single
# `retrieve` tool replaces both `search_knowledge_base` and the wiki navigation
# tools; it fuses tree routing + KB recall + overview synthesis server-side.
_ROUTED_DIRECTIVE = """# KNOWLEDGE MODE: ROUTED KNOWLEDGE RETRIEVAL (OVERRIDE — HIGHEST PRIORITY)

Your ONLY knowledge-retrieval tool is `retrieve`. The `search_knowledge_base`,
`wiki_map`, and `wiki_open` tools are NOT available. Wherever the instructions
below tell you to call `search_knowledge_base` (or any wiki tool), call
`retrieve` instead — ONCE per domain question, in the turn-1 parallel batch,
passing the `tenant_id` from the active skill's [skill_tenant_id] line.

`retrieve` returns three kinds of reference, in this order:
1. **Domain knowledge: <area>** — distilled core rules/decorators/constraints an
   expert knows about the area. Use it to frame your understanding.
2. **… (knowledge)** — a document knowledge card: dense, declarative facts, exact
   decorator/API/property names and their effects, steps, defaults, gotchas.
3. Knowledge-base chunks — the original source text.

ANSWER KNOWLEDGE-FIRST: build your answer FROM the knowledge references (1 and 2)
as your understanding of the topic, then use the knowledge-base chunks (3) to
confirm exact wording/names and fill any specific gaps. Prefer the facts stated
in the knowledge cards; if a KB chunk contradicts a card, trust the KB source and
note it. Lead with the direct answer, cover every specific fact the question
needs, and cite the source `link` of the references you used (knowledge cards and
KB chunks carry links; the domain digest does not). Do NOT mention tool
availability. Every domain question REQUIRES a `retrieve` call."""


async def main() -> None:
    """Start the hosted Chat Agent as an HTTP server."""
    await app_config.init()

    # NOTE: Do NOT attach a manual OTel LoggingHandler here.
    # ResponsesHostServer already calls configure_azure_monitor() which
    # auto-instruments the root logger.  Adding a second handler causes
    # every log to appear twice in Application Insights.

    agent_client = get_agent_client()
    # Limit tool-call loop iterations to prevent infinite loops.
    agent_client.function_invocation_configuration["max_iterations"] = (
        MAX_TOOL_CALL_ITERATIONS
    )
    agent_dir = Path(__file__).parent
    instructions = _load_instructions(agent_dir / "instruction.md")
    with open(agent_dir / "agent.yaml", encoding="utf-8") as f:
        agent_config = yaml.safe_load(f)
    agent_name = agent_config["name"]

    # Append agent version so Foundry can filter traces per version.
    agent_version = os.environ.get("APP_VERSION")
    agent_id = f"{agent_name}:{agent_version}" if agent_version else agent_name
    project_client = get_project_client()

    # Init Tools (synchronous / instant)
    knowledge_tools = KnowledgeTools()
    wiki_nav_tools = WikiNavTools()
    retrieve_tools = RetrieveTools()
    web_tools = WebTools()
    pipeline_tools = PipelineTools()
    web_search_tool = agent_client.get_web_search_tool(
        search_context_size="medium",
    )

    # Knowledge mode controls which retrieval paths the agent gets:
    #   * hybrid    — KB search + wiki-tree navigation (both tools, agent merges).
    #   * routed    — single coarse-to-fine `retrieve` tool: the tree routes,
    #                 KB recalls wide within scope, overview pages synthesise
    #                 (server-side fused). Default.
    #   * wiki_only — wiki-tree navigation only (KB tool removed).
    #   * kb_only   — KB search only (no wiki tools).
    knowledge_mode = cfg("KNOWLEDGE_MODE", "routed").lower()
    enable_kb = knowledge_mode in ("hybrid", "kb_only")
    enable_wiki = knowledge_mode in ("hybrid", "wiki_only")
    enable_routed = knowledge_mode == "routed"

    tools = []
    if enable_routed:
        tools.append(retrieve_tools.retrieve)
    if enable_kb:
        tools.append(knowledge_tools.search_knowledge_base)
    if enable_wiki:
        tools.append(wiki_nav_tools.wiki_map)
        tools.append(wiki_nav_tools.wiki_open)
    tools.extend(
        [
            web_tools.web_fetch,
            pipeline_tools.azsdk_analyze_pipeline,
            web_search_tool,
        ]
    )
    logger.info(
        "Knowledge mode=%s (routed=%s, kb=%s, wiki=%s)",
        knowledge_mode,
        enable_routed,
        enable_kb,
        enable_wiki,
    )

    # Some modes remove search_knowledge_base, but the base instruction mandates
    # it. Prepend an override so the agent grounds on the available tool(s)
    # without narrating the KB tool's absence.
    if knowledge_mode == "routed":
        instructions = _ROUTED_DIRECTIVE + "\n\n" + instructions

    # In wiki-only mode the KB tool is absent, but the base instruction mandates
    # search_knowledge_base. Prepend an override so the agent grounds solely on
    # the wiki navigation tools without narrating the KB tool's absence.
    if knowledge_mode == "wiki_only":
        instructions = _WIKI_ONLY_DIRECTIVE + "\n\n" + instructions

    # Parallelise slow async startup tasks to reduce cold-start latency.
    async def _init_memory() -> None:
        try:
            await ensure_user_memory_store(project_client)
        except Exception:
            logger.exception("Memory store initialization failed, skipped")

    async def _init_mcp(factory):
        try:
            return await factory()
        except Exception:
            logger.exception("%s failed to initialize, skipped", factory.__name__)
            return None

    # Memory can be disabled (e.g. for evaluation) via ENABLE_MEMORY=false so the
    # agent does not read historical Q&A from the tenant/user memory stores or
    # write this turn back into them. Defaults to enabled (production unaffected).
    memory_enabled = cfg("ENABLE_MEMORY", "true").lower() == "true"

    # Ensuring the memory store is idempotent and usually a no-op, so run it in
    # the background instead of gating readiness on it.
    if memory_enabled:
        memory_init_task = asyncio.create_task(_init_memory())
        _background_tasks.add(memory_init_task)
        memory_init_task.add_done_callback(_background_tasks.discard)
    else:
        logger.warning("ENABLE_MEMORY=false — user/tenant memory read+write disabled")

    # Only the MCP tools must exist before building the agent; create in
    # parallel (connection is lazy).
    ado_task, github_task = await asyncio.gather(
        _init_mcp(create_ado_mcp_tool),
        _init_mcp(create_github_mcp_tool),
    )

    for mcp_tool in (ado_task, github_task):
        if mcp_tool is not None:
            tools.append(mcp_tool)

    # Memory context provider (memory store initializes in background; may not be ready yet)
    memory_provider = MemoryContextProvider(project_client) if memory_enabled else None

    # Compaction provider — compact history before and after each turn
    compaction_provider = CompactionProvider(
        before_strategy=ToolResultCompactionStrategy(keep_last_tool_call_groups=2),
        after_strategy=ToolResultCompactionStrategy(keep_last_tool_call_groups=1),
    )

    # Init Skills
    skills = create_tenant_skills()
    skills_provider = SkillsProvider(skills)

    context_providers = [skills_provider]
    if memory_provider is not None:
        context_providers.append(memory_provider)
    context_providers.append(compaction_provider)

    reasoning_effort = cfg("AI_FOUNDRY_AGENT_REASONING_EFFORT")
    agent = Agent(
        agent_client,
        name=agent_name,
        id=agent_id,
        instructions=instructions,
        tools=tools,
        context_providers=context_providers,
        default_options={
            "reasoning": {"effort": reasoning_effort},
            "max_tool_calls": MAX_TOOL_CALLS_PER_TURN,
            "include": ["web_search_call.action.sources"],
        },
    )

    server = ResponsesHostServer(agent)
    await server.run_async()


if __name__ == "__main__":
    # Logging — configure first so every subsequent step is observable.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    # Silence noisy loggers that flood container logs.
    for noisy_logger, level in [
        ("azure.core.pipeline.policies.http_logging_policy", logging.WARNING),  # HTTP request/response headers
        ("azure.cosmos._cosmos_http_logging_policy", logging.WARNING),  # Cosmos DB request/response logging
        ("azure.monitor.opentelemetry.exporter", logging.WARNING),  # telemetry transmission
        ("uvicorn.access", logging.WARNING),  # health-probe GET /readiness /liveness
        ("uvicorn", logging.WARNING),  # uvicorn root logger (also emits access logs)
        ("microsoft.opentelemetry.a365.core.exporters.agent365_exporter", logging.CRITICAL),  # A365 telemetry export 403s
        ("microsoft.opentelemetry._distro", logging.ERROR),  # benign "No module named 'agents'" (openai-agents SDK unused)
    ]:
        logging.getLogger(noisy_logger).setLevel(level)

    logger.info("Agent container starting...")

    asyncio.run(main())
