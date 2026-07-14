"""Pluggable synthesis + embedding backends for the wiki-tree build.

The build needs two model capabilities: **text synthesis** (node summaries and
rolled-up wiki pages) and **embeddings** (cross-link discovery + retrieval
entry). Both are abstracted behind small protocols with two implementations:

* :class:`AzureOpenAISynthesizer` / :class:`AzureOpenAIEmbedder` — production,
  via Azure OpenAI (AAD or key auth).
* :class:`ExtractiveSynthesizer` / :class:`HashingEmbedder` — deterministic,
  dependency-free fallbacks so the whole pipeline (and the offline retrieval
  eval) runs locally without any Azure credentials.

Which backend is used is decided by :func:`build_synthesizer` /
:func:`build_embedder` from configuration + availability, so callers never
branch on it.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Protocol, Sequence

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #
class Synthesizer(Protocol):
    """Turns node context into a short summary or a rolled-up wiki page."""

    def summarize(self, title: str, body: str) -> str: ...

    def roll_up(self, title: str, child_briefs: Sequence[str], preamble: str) -> str: ...


class Embedder(Protocol):
    """Maps texts to unit-length vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# --------------------------------------------------------------------------- #
# Deterministic fallbacks (no Azure / no network)
# --------------------------------------------------------------------------- #
def _first_sentences(text: str, n: int = 2, limit: int = 320) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    sentences = _SENTENCE_RE.split(text)
    out = " ".join(sentences[:n]).strip()
    return out[:limit].rstrip()


class ExtractiveSynthesizer:
    """LLM-free synthesizer: lead-sentence summaries + bulleted roll-ups.

    Good enough to exercise the full pipeline and produce a navigable,
    retrievable tree offline; production uses :class:`AzureOpenAISynthesizer`.
    """

    def summarize(self, title: str, body: str) -> str:
        lead = _first_sentences(body)
        return lead or title

    def roll_up(self, title: str, child_briefs: Sequence[str], preamble: str) -> str:
        parts: list[str] = []
        if preamble.strip():
            parts.append(_first_sentences(preamble, n=3, limit=500))
        for brief in child_briefs:
            brief = brief.strip()
            if brief:
                parts.append(f"- {brief}")
        return "\n".join(parts).strip()


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (offline cosine works).

    Not semantically strong, but stable and dependency-free — sufficient for
    local cross-link discovery and the lexical-entry retrieval eval.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in _WORD_RE.findall(text.lower()):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


# --------------------------------------------------------------------------- #
# Azure OpenAI backends
# --------------------------------------------------------------------------- #
_SUMMARY_SYS = (
    "You are indexing Azure SDK / TypeSpec documentation. Write a single concise "
    "sentence (max 40 words) describing what this section explains, so an agent "
    "can decide whether to open it. No preamble, no markdown, just the sentence."
)
_ROLLUP_SYS = (
    "You are authoring an internal wiki overview page for Azure SDK / TypeSpec "
    "documentation. Given a topic and short briefs of its sub-sections, write a "
    "concise cross-document overview (120-200 words) that explains how the pieces "
    "relate and where to look for what. Be factual and specific; do not invent "
    "APIs or facts beyond the briefs. Output markdown prose, no headings."
)


class AzureOpenAISynthesizer:
    """Azure OpenAI chat-completions synthesizer (AAD or API-key auth)."""

    def __init__(self, client, deployment: str):
        self._client = client
        self._deployment = deployment

    def _complete(self, system: str, user: str, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def summarize(self, title: str, body: str) -> str:
        body = body.strip()
        if not body:
            return title
        user = f"Section title: {title}\n\nContent:\n{body[:6000]}"
        try:
            return self._complete(_SUMMARY_SYS, user, max_tokens=80) or _first_sentences(body)
        except Exception:
            logger.warning("summarize failed, using extractive fallback", exc_info=True)
            return _first_sentences(body) or title

    def roll_up(self, title: str, child_briefs: Sequence[str], preamble: str) -> str:
        briefs = "\n".join(f"- {b.strip()}" for b in child_briefs if b.strip())
        user = f"Topic: {title}\n\nPreamble:\n{preamble[:2000]}\n\nSub-section briefs:\n{briefs[:6000]}"
        try:
            return self._complete(_ROLLUP_SYS, user, max_tokens=400)
        except Exception:
            logger.warning("roll_up failed, using extractive fallback", exc_info=True)
            return ExtractiveSynthesizer().roll_up(title, child_briefs, preamble)


class AzureOpenAIEmbedder:
    """Azure OpenAI embeddings backend."""

    def __init__(self, client, deployment: str, batch_size: int = 64):
        self._client = client
        self._deployment = deployment
        self._batch = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        items = list(texts)
        for i in range(0, len(items), self._batch):
            chunk = [t[:8000] or " " for t in items[i : i + self._batch]]
            resp = self._client.embeddings.create(model=self._deployment, input=chunk)
            out.extend(d.embedding for d in resp.data)
        return out


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def _azure_openai_client():
    """Build a synchronous AzureOpenAI client from env, or return None.

    Uses ``AZURE_OPENAI_API_KEY`` when present, else AAD via
    ``azure-identity``. Endpoint from ``AZURE_OPENAI_ENDPOINT``. Returns
    ``None`` when the SDK or endpoint is unavailable so callers fall back.
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        return None
    try:
        from openai import AzureOpenAI
    except ImportError:
        logger.warning("openai package not installed; using deterministic backend")
        return None

    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if api_key:
        return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
    except Exception:
        logger.warning("Could not build AAD AzureOpenAI client; using fallback", exc_info=True)
        return None


def build_synthesizer(mode: str = "auto") -> Synthesizer:
    """Return a synthesizer. ``mode`` ∈ {auto, llm, extractive}."""
    if mode == "extractive":
        return ExtractiveSynthesizer()
    client = _azure_openai_client()
    if client is None:
        if mode == "llm":
            raise RuntimeError("LLM synthesizer requested but Azure OpenAI is unavailable")
        logger.info("Synthesizer: using deterministic ExtractiveSynthesizer")
        return ExtractiveSynthesizer()
    deployment = os.environ.get("WIKI_SYNTHESIS_DEPLOYMENT", "gpt-4.1-mini")
    logger.info("Synthesizer: using AzureOpenAI deployment=%s", deployment)
    return AzureOpenAISynthesizer(client, deployment)


def build_embedder(mode: str = "auto") -> Embedder:
    """Return an embedder. ``mode`` ∈ {auto, llm, hashing}."""
    if mode == "hashing":
        return HashingEmbedder()
    client = _azure_openai_client()
    if client is None:
        if mode == "llm":
            raise RuntimeError("LLM embedder requested but Azure OpenAI is unavailable")
        logger.info("Embedder: using deterministic HashingEmbedder")
        return HashingEmbedder()
    deployment = os.environ.get("WIKI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    logger.info("Embedder: using AzureOpenAI embeddings deployment=%s", deployment)
    return AzureOpenAIEmbedder(client, deployment)
