"""Snapshot serialisation + publication.

A snapshot is a small set of artefacts, mirroring the GraphRAG snapshot pattern
(immutable, timestamped, activated by a manifest flip):

* ``tree.json``           — the wiki tree (nodes + roots + stats + build id);
* ``embeddings.npy``      — ``float32`` [N, dim] node-embedding matrix;
* ``embedding_ids.json``  — node ids aligned to the matrix rows;
* ``latest.json``         — manifest pointing at the current snapshot build id.

:func:`write_snapshot_dir` persists locally (offline build + eval);
:func:`publish_snapshot_blob` uploads under ``<container>/snapshots/<build_id>/``
and flips ``latest.json`` — the backend picks it up on its next manifest poll.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .embeddings import EmbeddingIndex
from .models import WikiTree

logger = logging.getLogger(__name__)

TREE_BLOB = "tree.json"
EMB_NPY = "embeddings.npy"
EMB_IDS = "embedding_ids.json"
MANIFEST_BLOB = "latest.json"
SNAPSHOTS_PREFIX = "snapshots"


def write_snapshot_dir(
    out_dir: str | Path, tree: WikiTree, index: EmbeddingIndex
) -> Path:
    """Write the snapshot under ``out_dir/<build_id>/`` + root ``latest.json``."""
    root = Path(out_dir)
    snap_dir = root / tree.build_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    (snap_dir / TREE_BLOB).write_text(tree.to_json(indent=2), encoding="utf-8")
    index.save(snap_dir / EMB_NPY, snap_dir / EMB_IDS)
    (root / MANIFEST_BLOB).write_text(
        json.dumps({"build_id": tree.build_id, "stats": tree.stats}, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote snapshot to %s (%d nodes)", snap_dir, len(tree.nodes))
    return snap_dir


def load_snapshot_dir(
    root: str | Path, build_id: str | None = None
) -> tuple[WikiTree, EmbeddingIndex]:
    """Load a local snapshot (``build_id`` or the manifest's latest)."""
    root_path = Path(root)
    if build_id is None:
        manifest = json.loads((root_path / MANIFEST_BLOB).read_text(encoding="utf-8"))
        build_id = manifest["build_id"]
    snap_dir = root_path / build_id
    tree = WikiTree.from_json((snap_dir / TREE_BLOB).read_text(encoding="utf-8"))
    index = EmbeddingIndex.load(snap_dir / EMB_NPY, snap_dir / EMB_IDS)
    return tree, index


async def publish_snapshot_blob(
    container_client, tree: WikiTree, index: EmbeddingIndex
) -> dict:
    """Upload the snapshot to blob and flip ``latest.json``. Returns the manifest.

    *container_client* is an ``azure.storage.blob.aio.ContainerClient`` bound to
    the wiki output container.
    """
    prefix = f"{SNAPSHOTS_PREFIX}/{tree.build_id}"

    await container_client.upload_blob(
        name=f"{prefix}/{TREE_BLOB}", data=tree.to_json().encode("utf-8"), overwrite=True
    )
    await container_client.upload_blob(
        name=f"{prefix}/{EMB_NPY}", data=index.to_npy_bytes(), overwrite=True
    )
    await container_client.upload_blob(
        name=f"{prefix}/{EMB_IDS}",
        data=json.dumps(index.ids).encode("utf-8"),
        overwrite=True,
    )
    manifest = {"build_id": tree.build_id, "prefix": prefix, "stats": tree.stats}
    await container_client.upload_blob(
        name=MANIFEST_BLOB, data=json.dumps(manifest, indent=2).encode("utf-8"), overwrite=True
    )
    logger.info("Published wiki snapshot %s to blob (%s)", tree.build_id, prefix)
    return manifest
