"""Compact, numpy-backed embedding index for the wiki tree.

Node embeddings are stored as a single ``float32`` matrix (rows aligned to an
``ids`` list) rather than a dict-of-lists JSON — ~4x smaller on disk and fast to
score with a single matmul. Rows are L2-normalised so cosine similarity is just
``matrix @ query``.

Used by both the build (cross-link discovery) and the warm backend service
(query-time entry ranking).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


@dataclass
class EmbeddingIndex:
    """``ids`` + an aligned, row-normalised ``float32`` embedding matrix."""

    ids: list[str]
    matrix: np.ndarray  # shape [N, dim], float32, unit rows
    _pos: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._pos:
            self._pos = {nid: i for i, nid in enumerate(self.ids)}

    @classmethod
    def from_dict(cls, embeddings: dict[str, list[float]]) -> "EmbeddingIndex":
        ids = list(embeddings.keys())
        matrix = np.asarray([embeddings[i] for i in ids], dtype=np.float32)
        if matrix.ndim != 2:
            matrix = matrix.reshape(len(ids), -1)
        return cls(ids=ids, matrix=_normalise_rows(matrix))

    @classmethod
    def from_rows(cls, ids: list[str], rows: list[list[float]]) -> "EmbeddingIndex":
        matrix = np.asarray(rows, dtype=np.float32)
        return cls(ids=ids, matrix=_normalise_rows(matrix))

    def vector(self, node_id: str) -> np.ndarray | None:
        pos = self._pos.get(node_id)
        return None if pos is None else self.matrix[pos]

    def cosine_all(self, query: np.ndarray) -> np.ndarray:
        """Cosine of *query* against every row (query need not be normalised)."""
        q = np.asarray(query, dtype=np.float32)
        n = float(np.linalg.norm(q)) or 1.0
        return self.matrix @ (q / n)

    # -- persistence -----------------------------------------------------
    def save(self, npy_path: str | Path, ids_path: str | Path) -> None:
        np.save(str(npy_path), self.matrix)
        Path(ids_path).write_text(json.dumps(self.ids), encoding="utf-8")

    def to_npy_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.save(buf, self.matrix)
        return buf.getvalue()

    @classmethod
    def load(cls, npy_path: str | Path, ids_path: str | Path) -> "EmbeddingIndex":
        matrix = np.load(str(npy_path)).astype(np.float32)
        ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
        return cls(ids=ids, matrix=matrix)

    @classmethod
    def from_bytes(cls, npy_bytes: bytes, ids: list[str]) -> "EmbeddingIndex":
        matrix = np.load(io.BytesIO(npy_bytes)).astype(np.float32)
        return cls(ids=ids, matrix=matrix)
