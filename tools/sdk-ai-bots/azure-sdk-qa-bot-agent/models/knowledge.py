"""Data models for knowledge retrieval and search."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Knowledge source definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeSource:
    """A searchable knowledge source in Azure AI Search.

    Attributes:
        name:        Unique identifier used as the index / source filter value.
        description: Human-readable description so the LLM knows *when* to
                     query this source.
        base_url:    Base URL prefix for resolving documentation links.
        trim_format: Whether to strip .md/.mdx suffix and ``docs/`` prefix
                     from the title before appending to *base_url*.
        suffix:      String appended to the path after trimming (e.g. ".tsp").
        link_fn:     Optional callable ``(title) -> str`` for sources that need
                     custom link logic (overrides *base_url* when set).
    """

    name: str
    description: str
    base_url: str = ""
    trim_format: bool = False
    suffix: str = ""
    link_fn: "Callable[[str], str] | None" = field(default=None, repr=False)

    def get_link(self, title: str) -> str:
        """Resolve the documentation URL for a chunk title."""
        if title.startswith("version-release-notes-index"):
            return "Please reference link from document content"
        if self.link_fn is not None:
            return self.link_fn(title)
        if not self.base_url:
            return ""
        path = title.replace("#", "/")
        if self.trim_format:
            path = _trim_file_format(path)
        return self.base_url + path + self.suffix

    def to_display_dict(self) -> dict[str, str]:
        """Return a minimal dict suitable for showing the LLM."""
        return {"name": self.name, "description": self.description}


def _trim_file_format(path: str) -> str:
    """Strip .md / .mdx suffix and leading ``docs/`` prefix."""
    for ext in (".md", ".mdx"):
        if path.endswith(ext):
            path = path[: -len(ext)]
    if path.startswith("docs/"):
        path = path[5:]
    return path


# ---------------------------------------------------------------------------
# Search result models
# ---------------------------------------------------------------------------


class KnowledgeChunk(BaseModel):
    """A single chunk returned from Azure AI Search.

    Field aliases allow direct construction from the search index
    ``source_data`` dict via ``model_validate(source_data)``.
    """

    model_config = {"populate_by_name": True}

    chunk_id: str = ""
    title: str = ""
    content: str = Field(default="", validation_alias="chunk")
    source: str = Field(default="", validation_alias="context_id")
    link: str = ""
    header1: str = Field(default="", validation_alias="header_1")
    header2: str = Field(default="", validation_alias="header_2")
    header3: str = Field(default="", validation_alias="header_3")
    rerank_score: float = Field(default=0.0, validation_alias="@search.reranker_score")

    @field_validator("rerank_score", mode="before")
    @classmethod
    def _coerce_rerank_score(cls, v: object) -> float:
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        return float(str(v))


class KnowledgeResult(BaseModel):
    """Processed knowledge item ready for prompt injection."""

    title: str
    source: str
    link: str
    content: str


class Reference(BaseModel):
    """A reference to a document used to generate the answer."""

    title: str
    source: str
    link: str
    content: str = ""
    score: float = 0.0

    @field_validator("title", mode="after")
    @classmethod
    def _strip_trailing_pipes(cls, v: str) -> str:
        """Strip trailing pipe characters the LLM copies from search index titles."""
        return v.strip().rstrip("| ").strip()


class SearchKnowledgeBaseResult(BaseModel):
    """Output of the search_knowledge_base tool call."""

    results: list[Reference] = []


class WikiSearchResult(BaseModel):
    """Output of the ``search_wiki`` tool / ``POST /wiki/query`` endpoint.

    Mirrors ``SearchKnowledgeBaseResult`` / ``GraphSearchResult``: a flat list
    of :class:`Reference` objects plus the echoed query, so wiki-tree hits fuse
    uniformly with KB hits. ``content`` carries either a source section excerpt
    (``kind='section'``) or a rolled-up overview page (``kind='synthesis'``);
    ``source`` is the originating ``KnowledgeSource.name``.
    """

    references: list[Reference] = []
    query: str = ""


class WikiQueryRequest(BaseModel):
    """Request body for the ``POST /wiki/query`` endpoint.

    The chat agent posts here to delegate wiki-tree retrieval to the
    long-running backend server, which keeps a warm ``WikiTreeService``
    singleton (the snapshot + node embeddings load once per pod). When
    ``tenant_id`` resolves to a known ``TenantConfig``, retrieval is
    restricted to that tenant's ``KnowledgeSource`` folders — same scoping
    as ``search_knowledge_base``. Unknown / empty ``tenant_id`` falls back
    to unscoped retrieval.
    """

    query: str = Field(..., description="Natural-language query to retrieve wiki-tree references for.")
    tenant_id: str | None = Field(
        default=None,
        description=(
            "Optional tenant identifier; when set to a known TenantID, "
            "retrieval is restricted to that tenant's knowledge source folders."
        ),
    )


# ---------------------------------------------------------------------------
# Wiki-tree navigation (map / open) — the reasoning-based retrieval path
# ---------------------------------------------------------------------------
# Instead of collapsing the wiki tree into flat (chunk, link) references, the
# navigation tools let the chat agent reason over the structure like PageIndex:
# ``wiki_map`` returns a lightweight MAP of relevant nodes (title path +
# summary, no body) so the agent can decide *where* to look; ``wiki_open`` then
# returns the full distilled page + evidence + children/related handles for the
# nodes the agent chose, so it can read whole pages and drill further.


class WikiMapEntry(BaseModel):
    """One node on the map: enough to decide whether to open it, no body text."""

    id: str = Field(..., description="Opaque node handle to pass to wiki_open.")
    title_path: str = Field("", description="H1 | H2 | H3 heading path of the node.")
    summary: str = Field("", description="One-line description of what the node covers.")
    source: str = ""
    kind: str = Field("", description="root | folder | doc | section.")
    has_children: bool = Field(False, description="Whether the node can be drilled into.")
    doc_id: str = Field("", description="Handle of the enclosing document node (openable for its overview).")
    doc_title: str = ""
    score: float = 0.0


class WikiMapResult(BaseModel):
    """Output of ``wiki_map`` / ``POST /wiki/map`` — a ranked map of nodes."""

    entries: list[WikiMapEntry] = []
    query: str = ""


class WikiMapRequest(BaseModel):
    """Request body for ``POST /wiki/map``."""

    query: str = Field(..., description="Natural-language query to locate on the wiki tree.")
    tenant_id: str | None = None


class WikiChildView(BaseModel):
    """A child or related-node handle the agent can open next."""

    id: str
    title: str = ""
    summary: str = ""
    source: str = ""


class WikiNodeView(BaseModel):
    """A fully opened node: the distilled page + evidence + navigation handles."""

    id: str
    title_path: str = ""
    source: str = ""
    link: str = Field("", description="Resolved source document URL for citation.")
    page: str = Field("", description="Rolled-up cross-document overview page (may be empty for leaves).")
    content: str = Field("", description="Raw source section text — the citable evidence.")
    children: list[WikiChildView] = Field(default_factory=list, description="Sub-sections to drill into.")
    related: list[WikiChildView] = Field(default_factory=list, description="Cross-linked nodes in other documents.")


class WikiOpenResult(BaseModel):
    """Output of ``wiki_open`` / ``POST /wiki/open``."""

    nodes: list[WikiNodeView] = []


class WikiOpenRequest(BaseModel):
    """Request body for ``POST /wiki/open``."""

    node_ids: list[str] = Field(default_factory=list, description="Node handles from wiki_map (or children/related).")
    tenant_id: str | None = None


class DocumentContext(BaseModel):
    """A knowledge document in the eval-pipeline format (document_* keys)."""

    document_title: str = ""
    document_link: str = ""
    document_source: str = ""
    document_content: str = ""
    score: float = 0.0

    @staticmethod
    def from_reference(ref: Reference) -> "DocumentContext":
        return DocumentContext(
            document_title=ref.title,
            document_link=ref.link,
            document_source=ref.source,
            document_content=ref.content,
            score=ref.score,
        )
