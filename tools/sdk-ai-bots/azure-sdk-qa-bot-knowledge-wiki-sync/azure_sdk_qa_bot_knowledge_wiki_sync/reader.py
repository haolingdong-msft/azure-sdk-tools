"""Corpus readers — load ``(source_path, text)`` pairs for the wiki-tree build.

Two sources, same output shape:

* :func:`read_local_dir` — walk a local directory of markdown (used for local
  builds, tests, and the offline retrieval eval on the sample corpus).
* :func:`read_blob_container` — stream markdown from the shared knowledge blob
  container (production), mirroring how the GraphRAG sync reads the same corpus.

``source_path`` is always the container-relative path with ``/`` separators
(e.g. ``"typespec_docs/sub/foo.md"``) so folder attribution and link resolution
match the KB and graph paths.
"""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

_MD_SUFFIXES = {".md", ".mdx"}


def read_local_dir(root: str | Path) -> list[tuple[str, str]]:
    """Read every ``*.md``/``*.mdx`` under *root* as ``(source_path, text)``.

    ``source_path`` is the POSIX path relative to *root*, so a file
    ``<root>/typespec_docs/foo.md`` becomes ``"typespec_docs/foo.md"``.
    """
    root_path = Path(root).resolve()
    out: list[tuple[str, str]] = []
    for path in sorted(root_path.rglob("*")):
        if path.suffix.lower() not in _MD_SUFFIXES or not path.is_file():
            continue
        rel = PurePosixPath(path.relative_to(root_path).as_posix())
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        out.append((str(rel), text))
    logger.info("read_local_dir: %d markdown files under %s", len(out), root_path)
    return out


async def read_blob_container(
    container_client, prefix: str = ""
) -> list[tuple[str, str]]:
    """Read every markdown blob under *prefix* as ``(source_path, text)``.

    *container_client* is an ``azure.storage.blob.aio.ContainerClient`` already
    bound to the knowledge container. Blob names are used verbatim as
    ``source_path``.
    """
    out: list[tuple[str, str]] = []
    async for blob in container_client.list_blobs(name_starts_with=prefix or None):
        name = blob.name
        if PurePosixPath(name).suffix.lower() not in _MD_SUFFIXES:
            continue
        downloader = await container_client.download_blob(name)
        data = await downloader.readall()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        out.append((name, text))
    logger.info("read_blob_container: %d markdown blobs under %r", len(out), prefix)
    return out
