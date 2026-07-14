#!/usr/bin/env python3
"""CLI entry point for the Azure SDK Knowledge Wiki Sync.

Two modes:

* **local**  — build from a local markdown directory and write a snapshot to a
  local output directory (used for development, tests, and the offline
  retrieval eval)::

      python -m azure_sdk_qa_bot_knowledge_wiki_sync.main \
          --input ../azure-sdk-qa-bot-knowledge-sync/knowledge \
          --output .wiki-out --synth-mode extractive --embed-mode hashing

* **blob**   — read the shared knowledge container and publish an immutable
  snapshot + manifest to the wiki output container (production)::

      python -m azure_sdk_qa_bot_knowledge_wiki_sync.main --blob

Document collection is *not* done here — that is the
``azure-sdk-qa-bot-knowledge-sync`` project's job; this reads the markdown it
already maintains, exactly like the GraphRAG sync does.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("wiki_sync")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sync-knowledge-wiki")
    p.add_argument("--blob", action="store_true", help="Build from blob and publish (production).")
    p.add_argument("--input", help="Local markdown dir (local mode).")
    p.add_argument("--output", default=".wiki-out", help="Local snapshot output dir (local mode).")
    p.add_argument("--synth-mode", default="auto", choices=["auto", "llm", "extractive"])
    p.add_argument("--embed-mode", default="auto", choices=["auto", "llm", "hashing"])
    p.add_argument("--top-k-links", type=int, default=3)
    p.add_argument("--min-link-sim", type=float, default=0.55)
    return p.parse_args(argv)


def _run_local(args: argparse.Namespace) -> None:
    from .build import build_wiki_tree
    from .reader import read_local_dir
    from .snapshot import write_snapshot_dir

    if not args.input:
        raise SystemExit("--input <dir> is required in local mode (or pass --blob)")

    corpus = read_local_dir(args.input)
    if not corpus:
        raise SystemExit(f"No markdown found under {args.input}")

    tree, embeddings = build_wiki_tree(
        corpus,
        synth_mode=args.synth_mode,
        embed_mode=args.embed_mode,
        top_k_links=args.top_k_links,
        min_link_sim=args.min_link_sim,
    )
    snap_dir = write_snapshot_dir(args.output, tree, embeddings)
    logger.info("Local wiki snapshot ready: %s", snap_dir)


async def _run_blob(args: argparse.Namespace) -> None:
    from azure.identity.aio import DefaultAzureCredential
    from azure.storage.blob.aio import ContainerClient

    from .build import build_wiki_tree
    from .reader import read_blob_container
    from .snapshot import publish_snapshot_blob

    account_url = os.environ["STORAGE_BLOB_ENDPOINT"]
    knowledge_container = os.environ.get("STORAGE_KNOWLEDGE_CONTAINER", "knowledge")
    wiki_container = os.environ.get("STORAGE_WIKI_OUTPUT_CONTAINER", "wiki")

    credential = DefaultAzureCredential()
    try:
        in_client = ContainerClient(account_url, knowledge_container, credential=credential)
        async with in_client:
            corpus = await read_blob_container(in_client)
        if not corpus:
            raise SystemExit(f"No markdown in container {knowledge_container}")

        tree, embeddings = build_wiki_tree(
            corpus,
            synth_mode=args.synth_mode,
            embed_mode=args.embed_mode,
            top_k_links=args.top_k_links,
            min_link_sim=args.min_link_sim,
        )

        out_client = ContainerClient(account_url, wiki_container, credential=credential)
        async with out_client:
            manifest = await publish_snapshot_blob(out_client, tree, embeddings)
        logger.info("Published wiki snapshot: %s", manifest.get("build_id"))
    finally:
        await credential.close()


def main() -> None:
    args = _parse_args(sys.argv[1:])
    try:
        if args.blob:
            asyncio.run(_run_blob(args))
        else:
            _run_local(args)
    except Exception as exc:  # noqa: BLE001 — top-level CLI guard
        logger.error("Knowledge wiki sync failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
