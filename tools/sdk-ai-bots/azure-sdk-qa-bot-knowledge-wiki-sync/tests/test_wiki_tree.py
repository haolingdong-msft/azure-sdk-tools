"""Unit tests for the deterministic wiki-tree build + retrieval.

Run with: ``pytest`` (or ``python -m pytest``) from the project root.
These tests are Azure-free — they use the deterministic Extractive/Hashing
backends, so they exercise the full pipeline offline.
"""

from __future__ import annotations

import numpy as np

from azure_sdk_qa_bot_knowledge_wiki_sync.build import build_wiki_tree
from azure_sdk_qa_bot_knowledge_wiki_sync.embeddings import EmbeddingIndex
from azure_sdk_qa_bot_knowledge_wiki_sync.llm import HashingEmbedder
from azure_sdk_qa_bot_knowledge_wiki_sync.models import (
    KIND_DOC,
    KIND_FOLDER,
    KIND_ROOT,
    KIND_SECTION,
    WikiTree,
)
from azure_sdk_qa_bot_knowledge_wiki_sync.retrieval import WikiRetriever
from azure_sdk_qa_bot_knowledge_wiki_sync.toc import build_toc_tree

CORPUS = [
    (
        "typespec_docs/versioning.md",
        "# Versioning\nIntro to versioning.\n\n"
        "## @added decorator\nUse @added to mark when a member was introduced.\n"
        "```\n# not a header inside a fence\n```\n\n"
        "## @removed decorator\nUse @removed to mark removal.\n",
    ),
    (
        "typespec_docs/paging.md",
        "# Paging\nHow paging works.\n\n## Cursor paging\nCursor-based paging details.\n",
    ),
    (
        "api_guidelines/review.md",
        "# API Review\nThe review process and @added compatibility rules.\n",
    ),
]


def test_toc_structure():
    tree = build_toc_tree(CORPUS)
    roots = [tree.nodes[r] for r in tree.roots]
    assert len(roots) == 1 and roots[0].kind == KIND_ROOT
    folders = [n for n in tree.nodes.values() if n.kind == KIND_FOLDER]
    assert {f.title for f in folders} == {"typespec_docs", "api_guidelines"}
    docs = [n for n in tree.nodes.values() if n.kind == KIND_DOC]
    assert len(docs) == 3

    versioning = next(n for n in docs if n.source_path.endswith("versioning.md"))
    section_titles = {tree.nodes[c].title for c in versioning.children}
    # Top-level H1 becomes a section under the doc; its children are the H2s.
    h1 = next(tree.nodes[c] for c in versioning.children)
    child_titles = {tree.nodes[c].title for c in h1.children}
    assert "@added decorator" in child_titles
    assert "@removed decorator" in child_titles


def test_fence_lines_not_headers():
    tree = build_toc_tree(CORPUS)
    # The '# not a header inside a fence' line must not create a node.
    assert not any(n.title == "not a header inside a fence" for n in tree.nodes.values())


def test_rel_title_and_source():
    tree = build_toc_tree(CORPUS)
    doc = next(
        n for n in tree.nodes.values()
        if n.kind == KIND_DOC and n.source_path == "typespec_docs/versioning.md"
    )
    assert doc.source == "typespec_docs"
    assert doc.rel_title == "versioning.md"


def test_header_path_encoding_with_subdirs():
    corpus = [("azure-sdk-docs-eng/docs#design#api-review.md", "# API Review\nText.\n")]
    tree = build_toc_tree(corpus)
    doc = next(n for n in tree.nodes.values() if n.kind == KIND_DOC)
    assert doc.source == "azure-sdk-docs-eng"
    assert doc.rel_title == "docs#design#api-review.md"


def test_build_and_retrieve_scoped():
    tree, index = build_wiki_tree(
        CORPUS, synth_mode="extractive", embed_mode="hashing", top_k_links=2
    )
    assert isinstance(index, EmbeddingIndex)
    assert index.matrix.shape[0] > 0
    assert "synthesized" in tree.stats and "cross_links" in tree.stats

    retriever = WikiRetriever(tree, index)
    he = HashingEmbedder()
    refs = retriever.search(
        "What does the @added decorator do?",
        embed_query=lambda q: he.embed([q])[0],
        top_k=4,
        synthesis_k=1,
    )
    assert refs, "expected at least one reference"
    paths = {r.source_path for r in refs}
    assert "typespec_docs/versioning.md" in paths
    # A synthesis (overview) reference should be present.
    assert any(r.kind == "synthesis" for r in refs)

    # Scoping to a source folder excludes others.
    scoped = retriever.search(
        "@added compatibility",
        embed_query=lambda q: he.embed([q])[0],
        allowed_sources={"api_guidelines"},
        top_k=4,
    )
    assert all(r.source == "api_guidelines" for r in scoped)


def test_serialisation_round_trip():
    tree, index = build_wiki_tree(CORPUS, synth_mode="extractive", embed_mode="hashing")
    restored = WikiTree.from_json(tree.to_json())
    assert restored.build_id == tree.build_id
    assert set(restored.nodes) == set(tree.nodes)
    # Embedding index round-trips through bytes.
    idx2 = EmbeddingIndex.from_bytes(index.to_npy_bytes(), index.ids)
    assert np.allclose(idx2.matrix, index.matrix)


def test_map_returns_handles_no_body():
    tree, index = build_wiki_tree(CORPUS, synth_mode="extractive", embed_mode="hashing")
    retriever = WikiRetriever(tree, index)
    he = HashingEmbedder()
    entries = retriever.map(
        "How does the @added decorator work?",
        embed_query=lambda q: he.embed([q])[0],
        entry_k=6,
    )
    assert entries, "map should return entries"
    e = entries[0]
    # Map entries carry handles + summary, never raw section body.
    assert set(["id", "title_path", "summary", "kind", "has_children", "doc_id"]).issubset(e)
    assert "content" not in e and "page" not in e
    assert e["id"] in tree.nodes


def test_open_returns_page_evidence_and_handles():
    tree, index = build_wiki_tree(CORPUS, synth_mode="extractive", embed_mode="hashing")
    retriever = WikiRetriever(tree, index)
    # Open the versioning doc node → expect an overview page + child handles.
    doc = next(
        n for n in tree.nodes.values()
        if n.kind == KIND_DOC and n.source_path == "typespec_docs/versioning.md"
    )
    opened = retriever.open([doc.id])
    assert len(opened) == 1
    node = opened[0]
    assert node["id"] == doc.id
    assert node["page"], "doc node should carry a rolled-up overview page"
    assert isinstance(node["children"], list)
    assert all("id" in c and "title" in c for c in node["children"])
    assert "rel_title" in node  # for backend link resolution


def test_map_open_scoped():
    tree, index = build_wiki_tree(CORPUS, synth_mode="extractive", embed_mode="hashing")
    retriever = WikiRetriever(tree, index)
    he = HashingEmbedder()
    entries = retriever.map(
        "@added compatibility",
        embed_query=lambda q: he.embed([q])[0],
        allowed_sources={"api_guidelines"},
    )
    assert all(e["source"] == "api_guidelines" for e in entries)
    # Opening an out-of-scope node id is filtered out.
    ts_doc = next(
        n for n in tree.nodes.values()
        if n.kind == KIND_DOC and n.source == "typespec_docs"
    )
    assert retriever.open([ts_doc.id], allowed_sources={"api_guidelines"}) == []
