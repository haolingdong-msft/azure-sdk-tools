"""Build orchestration — corpus → wiki-tree + embeddings.

Ties the stages together: parse the ToC skeleton, synthesise summaries/pages
bottom-up, then discover cross-links and node embeddings. Returns the finished
:class:`~.models.WikiTree` and the ``{node_id: vector}`` embedding map for the
retrieval index. Pure (no I/O): callers persist the result via :mod:`snapshot`.
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid

from .crosslinks import add_cross_links
from .embeddings import EmbeddingIndex
from .llm import ExtractiveSynthesizer, build_embedder, build_synthesizer
from .models import WikiTree
from .synthesis import synthesize_tree
from .toc import build_toc_tree

logger = logging.getLogger(__name__)


def new_build_id() -> str:
    """Return a sortable, filesystem-safe build id (``<UTC ts>-<short>``)."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def build_wiki_tree(
    corpus: list[tuple[str, str]],
    *,
    synth_mode: str = "auto",
    embed_mode: str = "auto",
    top_k_links: int = 3,
    min_link_sim: float = 0.55,
) -> tuple[WikiTree, EmbeddingIndex]:
    """Build the wiki tree and node embeddings from a corpus.

    ``synth_mode``:
      * ``extractive`` — extractive summaries + extractive pages (no LLM).
      * ``knowledge`` — extractive summaries (stable embedding entry) +
        **LLM knowledge cards** on document nodes (dense facts extracted from the
        full doc text) + **domain-knowledge digests** on folder nodes. The
        artifact the agent reasons FROM. Default LLM mode.
      * ``overview`` / ``llm`` — legacy navigation roll-up pages (weaker on facts).
      * ``auto`` — knowledge mode when Azure OpenAI is configured, else extractive.
    """
    embedder = build_embedder(embed_mode)

    # summarizer writes the (cheap, stable) embedding-entry summaries; page_writer
    # does the LLM-heavy knowledge extraction / roll-up.
    synth_style = "knowledge"
    if synth_mode == "extractive":
        summarizer = ExtractiveSynthesizer()
        page_writer = ExtractiveSynthesizer()
    elif synth_mode in ("knowledge",):
        summarizer = ExtractiveSynthesizer()
        page_writer = build_synthesizer("llm")
    elif synth_mode in ("overview", "llm"):
        summarizer = ExtractiveSynthesizer()
        page_writer = build_synthesizer("llm")
        synth_style = "overview"
    else:  # auto
        summarizer = ExtractiveSynthesizer()
        page_writer = build_synthesizer("auto")

    logger.info("Building ToC skeleton from %d documents…", len(corpus))
    tree = build_toc_tree(corpus)

    logger.info("Synthesising (synth_mode=%s, style=%s)…", synth_mode, synth_style)
    synthesize_tree(tree, summarizer, page_writer, mode=synth_style)

    logger.info("Discovering cross-links and node embeddings…")
    index = add_cross_links(tree, embedder, top_k=top_k_links, min_sim=min_link_sim)

    tree.build_id = new_build_id()
    logger.info("Wiki tree build complete: build_id=%s stats=%s", tree.build_id, tree.stats)
    return tree, index
