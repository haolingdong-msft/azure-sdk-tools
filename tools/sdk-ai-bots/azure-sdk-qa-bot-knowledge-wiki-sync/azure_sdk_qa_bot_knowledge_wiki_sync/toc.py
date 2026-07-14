"""Deterministic markdown → wiki-tree structure builder (the PageIndex spine).

Parses the markdown corpus into a single navigable tree:

``root → folder (KnowledgeSource) → doc → section → subsection …``

The structure is derived **purely** from the markdown ``#`` header hierarchy —
no LLM, no chunking — so it is fast, reproducible, and exactly mirrors the
document structure the KB tool already relies on (``header_1/2/3``). Content
distillation (:mod:`synthesis`) and cross-linking (:mod:`crosslinks`) are layered
on top of the tree this module produces.
"""

from __future__ import annotations

import hashlib
import re

from .models import (
    KIND_DOC,
    KIND_FOLDER,
    KIND_ROOT,
    KIND_SECTION,
    WikiNode,
    WikiTree,
)

# ATX header: 1–6 ``#`` then a space then text (trailing ``#`` stripped).
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# Fenced code-block delimiter (``` or ~~~); ``#`` inside a fence is not a header.
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")

# Cap raw section text stored per node so the snapshot stays compact while
# still carrying enough to cite (the KB tool caps returned content at 3000).
_MAX_SECTION_CHARS = 4000


def _short_id(*parts: str) -> str:
    """Stable short id from the joined parts (source_path + header path)."""
    h = hashlib.sha1("\u0000".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _split_folder(source_path: str) -> tuple[str, str]:
    """Return ``(folder, rel_title)`` for a corpus path.

    ``folder`` is the first path segment (the KnowledgeSource name);
    ``rel_title`` is the folder-relative path, ``#``-encoded, matching the KB
    tool's link key (``title.replace('#','/')``). Mirrors the graph path's
    ``_source_path_to_rel_title``.
    """
    path = source_path.strip().lstrip("/")
    parts = path.split("/")
    folder = parts[0] if len(parts) > 1 else ""
    rel = path
    if folder:
        rel = path[len(folder) + 1 :]
    return folder, rel.replace("/", "#")


class _Header:
    __slots__ = ("level", "text", "line_start")

    def __init__(self, level: int, text: str, line_start: int):
        self.level = level
        self.text = text
        self.line_start = line_start


def _scan_headers(text: str) -> list[_Header]:
    """Return the ATX headers of *text* in document order (fences skipped)."""
    headers: list[_Header] = []
    in_fence = False
    for i, line in enumerate(text.splitlines()):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADER_RE.match(line)
        if not m:
            continue
        heading = m.group(2).strip()
        if heading:
            headers.append(_Header(len(m.group(1)), heading, i))
    return headers


def _section_body(lines: list[str], start_line: int, end_line: int) -> str:
    """Own text of a section: lines (start_line, end_line) excluding the header."""
    body = "\n".join(lines[start_line + 1 : end_line]).strip()
    if len(body) > _MAX_SECTION_CHARS:
        body = body[:_MAX_SECTION_CHARS].rstrip() + "\n… [truncated]"
    return body


def parse_document(tree: WikiTree, source_path: str, text: str) -> str:
    """Parse one document into the tree; return the doc node id.

    Builds a ``doc`` node plus a nested tree of ``section`` nodes from the
    markdown header hierarchy. Preamble before the first header is attached to
    the doc node's ``section_text``.
    """
    folder, rel_title = _split_folder(source_path)
    lines = text.splitlines()
    headers = _scan_headers(text)

    doc_title = headers[0].text if headers else (rel_title or source_path)
    doc_id = _short_id(source_path, "::doc")
    preamble_end = headers[0].line_start if headers else len(lines)
    preamble = "\n".join(lines[:preamble_end]).strip()[:_MAX_SECTION_CHARS]

    doc_node = WikiNode(
        id=doc_id,
        kind=KIND_DOC,
        title=doc_title,
        source=folder,
        source_path=source_path,
        rel_title=rel_title,
        header_path=[],
        section_text=preamble,
    )
    tree.add(doc_node)

    # Walk headers, maintaining a stack of (node_id, level) for parenting.
    stack: list[tuple[str, int]] = []  # excludes the doc root
    header_stack: list[str] = []  # heading text by depth, for header_path
    for idx, hdr in enumerate(headers):
        end_line = headers[idx + 1].line_start if idx + 1 < len(headers) else len(lines)
        body = _section_body(lines, hdr.line_start, end_line)

        # Pop to the parent: nearest node with a strictly smaller level.
        while stack and stack[-1][1] >= hdr.level:
            stack.pop()
            header_stack.pop()
        parent_id = stack[-1][0] if stack else doc_id
        header_path = header_stack[:] + [hdr.text]

        node_id = _short_id(source_path, ">".join(header_path))
        node = WikiNode(
            id=node_id,
            kind=KIND_SECTION,
            title=hdr.text,
            parent=parent_id,
            depth=len(header_path),
            source=folder,
            source_path=source_path,
            rel_title=rel_title,
            header_path=header_path,
            section_text=body,
            content_hash=_hash_text(body),
        )
        tree.add(node)
        tree.nodes[parent_id].children.append(node_id)

        stack.append((node_id, hdr.level))
        header_stack.append(hdr.text)

    doc_node.content_hash = _hash_text(preamble + "".join(doc_node.children))
    return doc_id


def build_toc_tree(corpus: list[tuple[str, str]]) -> WikiTree:
    """Build the full corpus wiki-tree skeleton from ``(source_path, text)`` pairs.

    Produces ``root → folder → doc → section`` structure only; ``summary`` /
    ``page`` / ``related`` are filled in by later stages.
    """
    tree = WikiTree()
    root_id = _short_id("::root")
    tree.add(WikiNode(id=root_id, kind=KIND_ROOT, title="Azure SDK Knowledge Wiki"))
    tree.roots = [root_id]

    folder_ids: dict[str, str] = {}
    for source_path, text in sorted(corpus):
        folder, _ = _split_folder(source_path)
        folder_key = folder or "(root)"
        folder_id = folder_ids.get(folder_key)
        if folder_id is None:
            folder_id = _short_id("::folder", folder_key)
            tree.add(
                WikiNode(
                    id=folder_id,
                    kind=KIND_FOLDER,
                    title=folder_key,
                    parent=root_id,
                    depth=0,
                    source=folder,
                )
            )
            tree.nodes[root_id].children.append(folder_id)
            folder_ids[folder_key] = folder_id

        doc_id = parse_document(tree, source_path, text)
        doc_node = tree.nodes[doc_id]
        doc_node.parent = folder_id
        tree.nodes[folder_id].children.append(doc_id)

    tree.stats = {
        "documents": sum(1 for n in tree.nodes.values() if n.kind == KIND_DOC),
        "sections": sum(1 for n in tree.nodes.values() if n.kind == KIND_SECTION),
        "folders": len(folder_ids),
        "nodes": len(tree.nodes),
    }
    return tree
