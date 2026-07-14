"""Data model for the wiki-tree knowledge index.

The index is a single **wiki tree**: a hierarchy of :class:`WikiNode` (the
PageIndex-style table-of-contents spine) whose nodes additionally carry
LLM-distilled content (the WeKnora-style wiki page) and associative
``related`` cross-links (the lightweight knowledge-graph edges). Every node
keeps a **source anchor** (``source`` + ``source_path`` + ``header_path``) so
each retrieval hit resolves back to a concrete document and link — the same
provenance contract the KB and graph paths use.

This module is intentionally dependency-free (pure dataclasses + JSON) so the
build project, the tests, and the backend retrieval service can all share one
serialisation format without importing each other.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


# Node kinds, coarsest → finest.
KIND_ROOT = "root"
KIND_FOLDER = "folder"
KIND_DOC = "doc"
KIND_SECTION = "section"


@dataclass
class WikiNode:
    """A single node in the wiki tree.

    A node is simultaneously:

    * a **structural** position (``kind`` / ``parent`` / ``children`` /
      ``header_path``) — the PageIndex navigation spine;
    * a **content** carrier (``summary`` for navigation, ``page`` for
      synthesised reading, ``section_text`` for the raw excerpt) — the
      WeKnora distilled wiki;
    * an **anchor** (``source`` / ``source_path`` / ``rel_title``) — provenance
      so the backend can resolve the same link the KB tool would; and
    * an **associative** hub (``related``) — the lightweight cross-link graph.
    """

    id: str
    kind: str
    title: str
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    depth: int = 0

    # -- provenance / anchoring ------------------------------------------
    source: str = ""  # KnowledgeSource folder name, e.g. "typespec_docs"
    source_path: str = ""  # full corpus path, e.g. "typespec_docs/foo.md"
    rel_title: str = ""  # source_path minus folder, "#"-encoded (KB link key)
    header_path: list[str] = field(default_factory=list)  # ["H1","H2","H3"]

    # -- content ---------------------------------------------------------
    section_text: str = ""  # raw own-text of the section (bounded), for citation
    summary: str = ""  # short navigation summary (LLM or extractive)
    page: str = ""  # synthesised wiki page (roll-up for internal nodes)

    # -- associative graph ----------------------------------------------
    related: list[str] = field(default_factory=list)  # cross-link node ids

    # -- bookkeeping -----------------------------------------------------
    content_hash: str = ""

    def title_path(self) -> str:
        """Return the ``H1 | H2 | H3`` heading path (KB reference title shape)."""
        return " | ".join(self.header_path) if self.header_path else self.title


@dataclass
class WikiTree:
    """A serialisable wiki-tree snapshot: nodes + roots + build metadata."""

    nodes: dict[str, WikiNode] = field(default_factory=dict)
    roots: list[str] = field(default_factory=list)
    build_id: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, node: WikiNode) -> None:
        self.nodes[node.id] = node

    def get(self, node_id: str) -> WikiNode | None:
        return self.nodes.get(node_id)

    def children_of(self, node_id: str) -> list[WikiNode]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [self.nodes[c] for c in node.children if c in self.nodes]

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_id": self.build_id,
            "stats": self.stats,
            "roots": self.roots,
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WikiTree":
        tree = cls(
            build_id=data.get("build_id", ""),
            stats=data.get("stats", {}),
            roots=list(data.get("roots", [])),
        )
        for nid, raw in data.get("nodes", {}).items():
            tree.nodes[nid] = WikiNode(**raw)
        return tree

    @classmethod
    def from_json(cls, text: str) -> "WikiTree":
        return cls.from_dict(json.loads(text))
