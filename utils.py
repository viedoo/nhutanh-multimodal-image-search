"""
Shared utilities used by app.py, qdrant_ingest.py and any future tooling.
Centralizes ID derivation and path normalization so that a single source of
truth drives both ingestion and runtime lookups.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from typing import Iterable

import numpy as np


# -----------------------------------------------------------------
# UUID derivation (must be identical in ingest & runtime)
# -----------------------------------------------------------------

def generate_uuid(image_id: str) -> str:
    """Deterministically derive a UUID from an image_id.

    Same algorithm in qdrant_ingest.py → identical UUIDs on re-ingest.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, image_id))


# -----------------------------------------------------------------
# Image path normalization
# -----------------------------------------------------------------

_KAGGLE_PREFIXES = (
    "/kaggle/input/dataset/",
    "/kaggle/input/",
    "/kaggle/working/dataset/",
)

def clean_image_path(raw_path: str) -> str:
    """Strip Kaggle mount prefix so paths resolve under the local dataset root.

    Stored paths used to look like '/kaggle/input/dataset/ladybugs/1.jpg' on
    Kaggle. Locally the dataset lives under 'test_kaggle/dataset/'.
    """
    p = raw_path.replace("\\", "/")
    for prefix in _KAGGLE_PREFIXES:
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    return p.lstrip("/")


def normalize_query(text: str) -> str:
    """Normalize a text query before hashing for cache lookup.

    Lowercases and collapses internal whitespace so 'Sleeping  Cat' and
    'sleeping cat' share the same cache entry.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def hash_query(text: str) -> str:
    """Cache key hash derived from the normalized query."""
    return hashlib.sha256(normalize_query(text).encode("utf-8")).hexdigest()[:32]


def hash_id(image_id: str) -> str:
    """Stable hash for image_id / similar-material cache keys (no normalization)."""
    return hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:32]


def to_image_bytes(arr: np.ndarray) -> str:
    """Serialize a float32 numpy vector to a hex string for Redis storage."""
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes().hex()


def from_image_bytes(raw: str, dim: int) -> np.ndarray:
    """Inverse of :func:`to_image_bytes`. Returns shape (1, dim)."""
    arr = np.frombuffer(bytes.fromhex(raw), dtype=np.float32)
    return arr.reshape(1, dim)


def batched(iterable: Iterable, n: int):
    """Yield successive n-sized chunks from an iterable (generator-safe)."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
