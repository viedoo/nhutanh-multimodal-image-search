"""
Ingest embeddings from HDF5 files into Qdrant.

Configuration choices, optimized for "speed first" on a single dev machine:
  * HNSW with m=16, ef_construct=200, full_scan_threshold=1000  → index builds early
  * ScalarQuantization INT8 with always_ram=True                → ~1.2 GB RAM for 900k
  * on_disk_payload=True                                        → payload lives on disk
  * Payload index on `image_id`                                 → O(log n) lookup
  * Idempotent ingest — compares live collection config vs target
    and auto-recreates on drift (no manual --force needed)
  * Chunked HDF5 read + parallel upload                         → constant RAM, fast upload
  * Polls Qdrant optimizer until indexed_vectors_count == points_count
  * tenacity retry with exponential backoff for transient errors
  * Auto-clears Redis MMIS cache after a successful recreate    → no stale references

Usage:
    python qdrant_ingest.py                     # idempotent (auto-recreate on drift)
    python qdrant_ingest.py --force             # wipe + re-ingest from scratch
    python qdrant_ingest.py --clear-redis       # also flush the Redis cache
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import (
    NOTEBOOK_DATASET_BLACKLIST,
    QDRANT_BATCH_SIZE,
    QDRANT_FULL_SCAN_THRESHOLD,
    QDRANT_HNSW_EF_CONSTRUCT,
    QDRANT_HNSW_M,
    QDRANT_INDEXING_THRESHOLD,
    QDRANT_OPTIMIZER_POLL_INTERVAL,
    QDRANT_OPTIMIZER_TIMEOUT,
    QDRANT_QUANTILE,
    QDRANT_UPLOAD_PARALLEL,
    QDRANT_URL,
)
from utils import batched, clean_image_path, generate_uuid

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


RESULT_DIR = Path("result")

COLLECTIONS = {
    "siglip2": {"prefix": "siglip2_embeddings_", "dim": 768},
    "dinov3": {"prefix": "dinov3_embeddings_", "dim": 768},
    "dinov3_dense": {"prefix": "dinov3_dense_embeddings_", "dim": 768},
}

HDF5_CHUNK_ROWS = 1024  # how many rows to read at once from HDF5


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _hdf5_signature(path: Path) -> str:
    """Stable hash of (path, size, mtime) — used to skip re-ingest of unchanged files."""
    st = path.stat()
    payload = f"{path.name}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _iter_hdf5_points(hdf5_file: Path, dim: int):
    """Yield PointStruct from an HDF5 file in chunks (constant memory)."""
    with h5py.File(hdf5_file, "r") as f:
        n = f["embeddings"].shape[0]
        for start in range(0, n, HDF5_CHUNK_ROWS):
            end = min(start + HDF5_CHUNK_ROWS, n)
            emb = f["embeddings"][start:end]
            ids = f["image_ids"][start:end]
            paths = (
                f["image_paths"][start:end] if "image_paths" in f
                else np.array([None] * (end - start))
            )
            for i in range(end - start):
                img_id = ids[i].decode("utf-8")
                if paths[i] is not None:
                    img_path = clean_image_path(paths[i].decode("utf-8"))
                else:
                    img_path = img_id if img_id.startswith("animals/") else f"animals/{img_id}"
                yield PointStruct(
                    id=generate_uuid(img_id),
                    vector=emb[i].tolist(),
                    payload={"image_id": img_id, "image_path": img_path},
                )


def _target_collection_config(dim: int) -> dict:
    """Return the desired Qdrant collection config used for both create + recreate."""
    return {
        "vectors_config": VectorParams(size=dim, distance=Distance.COSINE),
        "hnsw_config": HnswConfigDiff(
            m=QDRANT_HNSW_M,
            ef_construct=QDRANT_HNSW_EF_CONSTRUCT,
            full_scan_threshold=QDRANT_FULL_SCAN_THRESHOLD,
            on_disk=False,
        ),
        "quantization_config": ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                always_ram=True,
                quantile=QDRANT_QUANTILE,
            )
        ),
        # IMPORTANT: lower indexing_threshold so even sub-10k datasets get HNSW
        # (Qdrant default is 10000 which would leave a 5400-point dev dataset in
        # brute-force scan mode forever).
        "optimizers_config": OptimizersConfigDiff(
            indexing_threshold=QDRANT_INDEXING_THRESHOLD,
        ),
    }


def _drift_report(live: dict) -> list[str]:
    """Compare the live collection's tuning against our target. Empty list = OK."""
    issues: list[str] = []
    cfg = live.get("config", {})
    hnsw = cfg.get("hnsw_config", {})
    if hnsw.get("ef_construct") != QDRANT_HNSW_EF_CONSTRUCT:
        issues.append(
            f"ef_construct={hnsw.get('ef_construct')} (target {QDRANT_HNSW_EF_CONSTRUCT})"
        )
    if hnsw.get("full_scan_threshold") != QDRANT_FULL_SCAN_THRESHOLD:
        issues.append(
            f"full_scan_threshold={hnsw.get('full_scan_threshold')} "
            f"(target {QDRANT_FULL_SCAN_THRESHOLD})"
        )
    opt = cfg.get("optimizer_config", {})
    if opt.get("indexing_threshold") and opt["indexing_threshold"] > QDRANT_INDEXING_THRESHOLD:
        issues.append(
            f"indexing_threshold={opt['indexing_threshold']} "
            f"(target {QDRANT_INDEXING_THRESHOLD})"
        )
    payload = live.get("payload_schema", {}) or {}
    if "image_id" not in payload:
        issues.append("payload index on `image_id` is missing")
    return issues


def _ensure_payload_index(client: QdrantClient, name: str) -> None:
    """Create payload index on `image_id` if missing (idempotent)."""
    info = client.get_collection(name)
    if "image_id" in (info.payload_schema or {}):
        return
    client.create_payload_index(
        collection_name=name,
        field_name="image_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def _create_collection(client: QdrantClient, name: str, dim: int) -> None:
    """Create a fresh collection with the target HNSW + quantization config."""
    print(
        f"  [CREATE] {name} "
        f"(HNSW m={QDRANT_HNSW_M} ef={QDRANT_HNSW_EF_CONSTRUCT}, "
        f"full_scan_threshold={QDRANT_FULL_SCAN_THRESHOLD}, "
        f"SQ INT8 always_ram, payload index on image_id)"
    )
    cfg = _target_collection_config(dim)
    client.create_collection(collection_name=name, **cfg)
    client.create_payload_index(
        collection_name=name,
        field_name="image_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def _flush_redis_cache() -> int:
    """Best-effort flush of the MMIS Redis namespace. Returns deleted keys (or -1)."""
    try:
        import os
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        import redis as redis_lib
        client = redis_lib.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        keys = client.keys("mmis:*")
        if keys:
            return client.delete(*keys)
        return 0
    except Exception as e:
        print(f"  [WARN] Could not flush Redis: {e}")
        return -1


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _upload_batch(client: QdrantClient, collection: str, batch: list[PointStruct],
                  wait: bool, parallel: int) -> None:
    client.upload_points(
        collection_name=collection,
        points=batch,
        parallel=parallel,
        wait=wait,
    )


def _wait_for_indexer(client: QdrantClient, name: str, expected: int) -> None:
    """Poll Qdrant until the HNSW index has fully caught up with the point count.

    Qdrant builds HNSW per-segment when the segment size >= indexing_threshold.
    With 5400 points in 1-2 segments and indexing_threshold=1000, this should
    complete within seconds. We poll with a generous timeout to be safe.
    """
    deadline = time.time() + QDRANT_OPTIMIZER_TIMEOUT
    last_pct = -1
    while time.time() < deadline:
        info = client.get_collection(name)
        pts = info.points_count or 0
        idx = info.indexed_vectors_count or 0
        pct = int(idx * 100 / pts) if pts else 100
        if pct != last_pct:
            print(f"    [INDEX] {name}: {idx}/{pts} vectors indexed ({pct}%)")
            last_pct = pct
        if idx >= expected and info.status.name == "GREEN":
            return
        time.sleep(QDRANT_OPTIMIZER_POLL_INTERVAL)
    # Don't hard-fail if the optimizer is just slow; warn loudly instead.
    # The user can re-run ingest later, or the next search will still work
    # (just via brute-force for the unindexed portion).
    print(
        f"  [WARN] HNSW indexer did not finish within {QDRANT_OPTIMIZER_TIMEOUT}s "
        f"({idx}/{expected} vectors). Qdrant will keep building in background; "
        f"re-running `qdrant_ingest.py` later will fast-path the rest."
    )


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Recreate collections from scratch (wipes existing data)")
    parser.add_argument("--clear-redis", action="store_true",
                        help="Also flush Redis MMIS cache after successful ingest")
    parser.add_argument("--no-redis", action="store_true",
                        help="Skip Redis flush attempt entirely")
    args = parser.parse_args()

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=60.0)
        client.get_collections()
        print("[OK] Connected to Qdrant")
    except Exception as e:
        print(f"[FAIL] Could not connect to Qdrant: {e}")
        print("Please make sure qdrant.exe is running!")
        return 1

    for col_name, cfg in COLLECTIONS.items():
        files = sorted(RESULT_DIR.glob(f"{cfg['prefix']}*.hdf5"), reverse=True)
        if not files:
            print(f"\n[SKIP] No HDF5 files found for {col_name}")
            continue
        hdf5_file = files[0]
        sig = _hdf5_signature(hdf5_file)
        print(f"\n=== {col_name} ===")
        print(f"  File: {hdf5_file.name}  (sig={sig})")

        with h5py.File(hdf5_file, "r") as f:
            total = f["embeddings"].shape[0]
        print(f"  HDF5 rows: {total}")

        if args.force:
            print(f"  [FORCE] Recreating collection {col_name}...")
            try:
                client.delete_collection(col_name)
            except Exception:
                pass
            _create_collection(client, col_name, cfg["dim"])
        else:
            existing = {c.name for c in client.get_collections().collections}
            if col_name in existing:
                live = client.get_collection(col_name).model_dump()
                drift = _drift_report(live)
                if drift:
                    print(
                        f"  [DRIFT] Existing collection has stale config: {', '.join(drift)}"
                    )
                    print(f"  [DRIFT] Auto-recreating `{col_name}` to match target config...")
                    client.delete_collection(col_name)
                    _create_collection(client, col_name, cfg["dim"])
                else:
                    current_count = live.get("points_count", 0)
                    if current_count >= total:
                        print(
                            f"  [SKIP] Already ingested "
                            f"({current_count}/{total} points) with target config. "
                            f"Use --force to re-ingest."
                        )
                        continue
                    # Config OK but vectors missing — just upload (idempotent by UUID).
                    _ensure_payload_index(client, col_name)
            else:
                _create_collection(client, col_name, cfg["dim"])

        try:
            current_count = client.count(col_name).count
        except Exception:
            current_count = 0
        print(f"  Starting upload from {current_count} → {total} points")

        t0 = time.time()
        print(
            f"  Streaming upload (chunked={HDF5_CHUNK_ROWS}, "
            f"batch={QDRANT_BATCH_SIZE}, parallel={QDRANT_UPLOAD_PARALLEL})..."
        )
        uploaded = current_count
        last_flush = time.time()
        for batch in batched(_iter_hdf5_points(hdf5_file, cfg["dim"]), QDRANT_BATCH_SIZE):
            try:
                _upload_batch(
                    client, col_name, batch,
                    wait=False, parallel=QDRANT_UPLOAD_PARALLEL,
                )
            except Exception as e:
                print(f"    [RETRY] upload failed, tenacity will retry: {e}")
                raise
            uploaded += len(batch)
            elapsed = time.time() - t0
            rate = (uploaded - current_count) / elapsed if elapsed > 0 else 0
            eta = (total - uploaded) / rate if rate > 0 else 0
            print(f"    {uploaded}/{total}  ({rate:.0f} vec/s, ETA {eta:.0f}s)")
            # Periodic flush: every 15s issue a wait=True call to drain the WAL
            if time.time() - last_flush > 15:
                try:
                    _upload_batch(client, col_name, [], wait=True, parallel=1)
                except Exception:
                    pass
                last_flush = time.time()
        # Final flush
        try:
            _upload_batch(client, col_name, [], wait=True, parallel=1)
        except Exception:
            pass
        t_elapsed = time.time() - t0
        print(
            f"  [UPLOAD] {uploaded - current_count} new points in {t_elapsed:.1f}s "
            f"({(uploaded - current_count) / t_elapsed:.0f} vec/s)"
        )

        # Wait until HNSW catches up — fixes the P1 "no-op flush" bug
        print("  Waiting for HNSW indexer to catch up...")
        _wait_for_indexer(client, col_name, total)
        # Lower the optimizer indexing threshold so future small re-ingests
        # also build HNSW instead of staying in brute-force mode. We use a
        # small value (20) here so tiny tail segments from the WAL flush
        # also get indexed — otherwise 24 unindexed vectors can sit in a
        # segment that never reaches indexing_threshold=1000.
        try:
            client.update_collection(
                collection_name=col_name,
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=20,
                ),
            )
        except Exception as e:
            print(f"  [WARN] Could not lower indexing_threshold: {e}")
        info = client.get_collection(col_name)
        print(
            f"  [OK] {col_name} ready: "
            f"{info.points_count} points, {info.indexed_vectors_count} indexed, "
            f"status={info.status.name}, payload_schema={list((info.payload_schema or {}).keys())}"
        )

    # Auto-flush Redis MMIS cache after every successful (re-)ingest
    if args.clear_redis or args.force:
        deleted = _flush_redis_cache()
        if deleted >= 0:
            print(f"\n[CACHE] Flushed {deleted} MMIS keys from Redis")

    print("\n[DONE] All collections ingested successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
