from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import threading
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from pathlib import Path
from dotenv import load_dotenv
from cachetools import LRUCache
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from cache import RedisCache
from config import QDRANT_URL, QDRANT_SEARCH_EF, VIDEO_COLLECTION, VIDEO_EMBED_INSTRUCTION
from utils import (
    clean_image_path,
    generate_uuid,
    hash_id,
    hash_query,
    normalize_query,
)

# Fix windows console unicode errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------
# Redis Cache
# -----------------------------------------------------------------
cache = RedisCache()

MODEL_ID = "google/siglip2-base-patch16-naflex"
DATASET_ROOT = Path(__file__).parent / "dataset"

# -----------------------------------------------------------------
# Qdrant Database Setup
# -----------------------------------------------------------------
print(f"Connecting to Qdrant at {QDRANT_URL}...")
try:
    # prefer_grpc=True: gRPC persistent connection avoids the ~2s TCP
    # handshake penalty that REST suffers from on every request (qdrant_client
    # does NOT pool HTTP connections in the way `requests.Session` does).
    # gRPC also brings ~5-9ms per query vs ~2000ms over HTTP on this box.
    qdrant = QdrantClient(url=QDRANT_URL, prefer_grpc=True, timeout=5.0)
    collections = [c.name for c in qdrant.get_collections().collections]
    print(f"[OK] Connected to Qdrant (gRPC). Available collections: {collections}")

    if "siglip2" not in collections:
        print("[WARN] Collection 'siglip2' not found in Qdrant! Please run qdrant_ingest.py first.")
        sys.exit(1)

except Exception as e:
    print(f"\n[FATAL ERROR] Could not connect to Qdrant Database: {e}")
    print("Please make sure qdrant.exe is running on Windows (or via Docker) before starting this app.")
    print("To conserve RAM and ensure stability, the app will now exit.")
    sys.exit(1)


# -----------------------------------------------------------------
# Local SigLIP2 Model Setup (For Text Queries)
# -----------------------------------------------------------------
print("Loading SigLIP 2 model for text embedding...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
# FP16 is only a real win on CUDA — on CPU the matmuls are usually slower
# (AVX-512 FP16 is not available on most desktop CPUs). Stay on FP32 by default.
has_cuda = torch.cuda.is_available()
_dtype = torch.float16 if has_cuda else torch.float32
model = AutoModel.from_pretrained(
    MODEL_ID, torch_dtype=_dtype, low_cpu_mem_usage=True
)
model.eval()
device = "cuda" if has_cuda else "cpu"
# Cap threads: SigLIP2 forward is small, too many threads hurts cache locality.
torch.set_num_threads(max(1, min(4, (torch.get_num_threads() or 2))))
print(f"[OK] SigLIP 2 model loaded ({_dtype}, threads={torch.get_num_threads()}, device={device})!")

# In-process stampede protection: only one thread runs the model for the same query
# Bounded LRU so the dict cannot leak memory across millions of unique queries.
_EMBED_LOCK_MAX = 2048
_embed_locks: "LRUCache[str, threading.Lock]" = LRUCache(maxsize=_EMBED_LOCK_MAX)
_embed_locks_guard = threading.Lock()


def _get_embed_lock(key: str) -> threading.Lock:
    with _embed_locks_guard:
        lock = _embed_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _embed_locks[key] = lock
        return lock


def embed_text(text: str) -> list[float]:
    """Compute (or fetch from cache) the SigLIP2 text embedding."""
    norm = normalize_query(text)
    cache_key = hash_query(norm)

    # 1. Try Redis cache first
    cached_emb = cache.get_embedding(cache_key)
    if cached_emb is not None:
        return cached_emb.tolist()[0]

    # 2. Stampede protection: only one thread embeds the same query at a time
    lock = _get_embed_lock(cache_key)
    with lock:
        # Re-check after acquiring the lock (another thread may have just computed it)
        cached_emb = cache.get_embedding(cache_key)
        if cached_emb is not None:
            return cached_emb.tolist()[0]

        # 3. Cache MISS - compute via SigLIP2
        # IMPORTANT: must use padding='max_length' with a fixed length, NOT
        # dynamic padding=True. SigLIP2's text encoder is sensitive to sequence
        # length — a 2-token input for "cat" (padded dynamically) lands in a
        # different region of embedding space than a 64-token fixed input.
        # Diagnostic: with dynamic padding, cat-vs-cat sim dropped from 0.11
        # to -0.03. Fixed 64-token padding matches the Kaggle notebook output.
        inputs = processor(
            text=[norm], return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():  # ~5% faster than torch.no_grad()
            outputs = model.get_text_features(**inputs)
            text_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
            text_features = F.normalize(text_features, p=2, dim=-1)

        embedding = text_features.cpu().numpy().astype(np.float32)

    # 4. Store in Redis for next time (outside the lock to keep critical section short)
    cache.set_embedding(cache_key, embedding)
    return embedding.tolist()[0]

print("Ready for searches!")


def _search_params() -> qmodels.SearchParams:
    """HNSW ef + scalar-quant rescore. Rescore restores ~99% of full-precision recall
    while keeping the 4x RAM + bandwidth savings of INT8 quantization."""
    return qmodels.SearchParams(
        hnsw_ef=QDRANT_SEARCH_EF,
        quantization=qmodels.QuantizationSearchParams(
            ignore=False,
            rescore=True,
            oversampling=2.0,
        ),
    )


def _search_by_image_id(collection: str, image_id: str, top_k: int):
    """Find points similar to an existing image using Qdrant's `recommend` API.

    1-RTT: Qdrant reads the seed vector internally, then runs cosine KNN
    against `dinov3` / `dinov3_dense`. Previously we did a separate
    `retrieve(with_vectors=True)` + `query_points`, which was 2 RTTs.
    """
    point_uuid = generate_uuid(image_id)
    try:
        resp = qdrant.query_points(
            collection_name=collection,
            query=qmodels.RecommendQuery(
                recommend=qmodels.RecommendInput(positive=[point_uuid])
            ),
            limit=top_k + 1,  # +1 to drop the seed from results
            with_payload=True,
            with_vectors=False,
            search_params=_search_params(),
        )
    except Exception as e:
        msg = str(e)
        if "Not found" in msg or "Point id" in msg:
            return None, 'not_found'
        raise
    hits = [p for p in resp.points if str(p.id) != point_uuid][:top_k]
    return hits, 'ok'


# -----------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/healthz')
def healthz():
    """Liveness + dependency probe. Used by Docker / load balancers.

    Policy: Qdrant is REQUIRED. Redis is optional — if missing, we fall back
    to the in-memory cache and report `degraded` (200), not 503. This matches
    the README contract.
    """
    qdrant_ok = True
    try:
        qdrant.get_collections()
    except Exception:
        qdrant_ok = False
    redis_ok = cache.available
    healthy = qdrant_ok
    return jsonify({
        'status': 'ok' if healthy else 'unavailable',
        'qdrant': qdrant_ok,
        'redis': redis_ok,
        'embedding_dim': 768,
    }), 200 if healthy else 503

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 10)

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    t_start = time.perf_counter()

    # 1. Check search-result cache (key uses normalized text + top_k)
    cache_key = f"{normalize_query(query)}|k={top_k}"
    cached = cache.get_results('text', cache_key, top_k)
    if cached is not None:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return jsonify({'results': cached, 'cache': 'HIT', 'latency_ms': round(elapsed_ms, 2)})

    # 2. Cache MISS - embed query then search Qdrant
    query_vector = embed_text(query)

    try:
        search_response = qdrant.query_points(
            collection_name="siglip2",
            query=query_vector,
            limit=top_k,
            with_payload=True,
            search_params=_search_params(),
        )
        search_result = search_response.points
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    results = []
    for rank, hit in enumerate(search_result):
        payload = hit.payload or {}
        img_id = payload.get('image_id', str(hit.id))
        img_relative = clean_image_path(payload.get('image_path', img_id))

        results.append({
            'rank': rank + 1,
            'image_id': img_id,
            'image_url': f"/images/{img_relative}",
            'score': hit.score
        })

    # 3. Store results in cache (use normalized key for hit-rate)
    cache.set_results('text', cache_key, results, top_k)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    return jsonify({'results': results, 'cache': 'MISS', 'latency_ms': round(elapsed_ms, 2)})


@app.route('/api/search-similar', methods=['POST'])
def search_similar():
    data = request.json
    image_id = data.get('image_id', '')
    top_k = data.get('top_k', 10)

    if not image_id:
        return jsonify({'error': 'image_id is required'}), 400

    t_start = time.perf_counter()

    # 1. Check result cache
    cached = cache.get_results('similar', hash_id(image_id), top_k)
    if cached is not None:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return jsonify({'results': cached, 'cache': 'HIT', 'latency_ms': round(elapsed_ms, 2)})

    # 2. Cache MISS - 1-RTT recommend against the dinov3 collection
    hits, status = _search_by_image_id("dinov3", image_id, top_k)
    if status == 'not_found':
        return jsonify({'error': 'Image not found in dinov3 collection'}), 404
    if status != 'ok':
        return jsonify({'error': 'Similarity search failed: ' + status}), 503

    results = []
    for rank, hit in enumerate(hits):
        payload = hit.payload or {}
        res_id = payload.get('image_id', str(hit.id))
        img_relative = clean_image_path(payload.get('image_path', res_id))

        results.append({
            'rank': rank + 1,
            'image_id': res_id,
            'image_url': f"/images/{img_relative}",
            'score': hit.score
        })

    # 3. Store in cache
    cache.set_results('similar', hash_id(image_id), results, top_k)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    return jsonify({
        'results': results,
        'reference_image_url': f"/images/{clean_image_path(image_id)}",
        'cache': 'MISS',
        'latency_ms': round(elapsed_ms, 2),
    })


@app.route('/api/search-similar-material', methods=['POST'])
def search_similar_material():
    """Material / texture similarity using DINOv3 dense patch-mean features."""
    data = request.json
    image_id = data.get('image_id', '')
    top_k = data.get('top_k', 10)

    if not image_id:
        return jsonify({'error': 'image_id is required'}), 400

    t_start = time.perf_counter()

    # 1. Check result cache
    cached = cache.get_results('material', hash_id(image_id), top_k)
    if cached is not None:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return jsonify({'results': cached, 'cache': 'HIT', 'latency_ms': round(elapsed_ms, 2)})

    # 2. Cache MISS - 1-RTT recommend against the dinov3_dense collection
    hits, status = _search_by_image_id("dinov3_dense", image_id, top_k)
    if status == 'not_found':
        return jsonify({'error': 'Image not found in dinov3_dense collection'}), 404
    if status != 'ok':
        return jsonify({'error': 'Material search failed: ' + status}), 503

    results = []
    for rank, hit in enumerate(hits):
        payload = hit.payload or {}
        res_id = payload.get('image_id', str(hit.id))
        img_relative = clean_image_path(payload.get('image_path', res_id))

        results.append({
            'rank': rank + 1,
            'image_id': res_id,
            'image_url': f"/images/{img_relative}",
            'score': hit.score
        })

    # 3. Store in cache
    cache.set_results('material', hash_id(image_id), results, top_k)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    return jsonify({
        'results': results,
        'reference_image_url': f"/images/{clean_image_path(image_id)}",
        'cache': 'MISS',
        'latency_ms': round(elapsed_ms, 2),
    })


# -----------------------------------------------------------------
# Cache Management Endpoints
# -----------------------------------------------------------------

@app.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """Return Redis cache statistics (hits, misses, memory, key counts)."""
    return jsonify(cache.stats())

@app.route('/api/cache/clear', methods=['POST'])
def cache_clear():
    """Clear all cached search results and embeddings."""
    count = cache.clear_all()
    return jsonify({'status': 'ok', 'deleted_keys': count})


# -----------------------------------------------------------------
# Video Search (Qwen3-VL-Embedding) — lazy-loaded
# -----------------------------------------------------------------

_qwen3vl_model = None
_qwen3vl_lock = threading.Lock()
QWEN3VL_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"


def _get_qwen3vl_model():
    """Lazy-load the Qwen3-VL-Embedding model on first video search.

    The model is ~4 GB in BF16 — adding it to startup would slow boot and
    double RAM usage. Only loaded when /api/search-video is actually called.
    """
    global _qwen3vl_model
    if _qwen3vl_model is not None:
        return _qwen3vl_model
    with _qwen3vl_lock:
        if _qwen3vl_model is not None:
            return _qwen3vl_model
        from scripts.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: PLC0415
        _dtype = torch.bfloat16 if has_cuda else torch.float32
        print(f"[LAZY] Loading {QWEN3VL_MODEL_ID} for video search...")
        t0 = time.time()
        _qwen3vl_model = Qwen3VLEmbedder(
            model_name_or_path=QWEN3VL_MODEL_ID,
            max_length=8192,
            max_pixels=224 * 224,    # match notebook's tuned setting
            fps=0.5,
            max_frames=16,
            torch_dtype=_dtype,
        )
        print(f"[LAZY] Qwen3-VL-Embedding ready in {time.time() - t0:.1f}s")
        return _qwen3vl_model


def embed_text_for_video(text: str) -> list[float]:
    """Embed a text query into Qwen3-VL-Embedding space (1024d, L2-normalized)."""
    norm = normalize_query(text)
    cache_key = hash_query(norm)

    # 1. Try Redis cache first
    cached_emb = cache.get_embedding(f"video|{cache_key}")
    if cached_emb is not None:
        return cached_emb.tolist()[0]

    lock = _get_embed_lock(f"video|{cache_key}")
    with lock:
        cached_emb = cache.get_embedding(f"video|{cache_key}")
        if cached_emb is not None:
            return cached_emb.tolist()[0]

        # 2. Compute via Qwen3VLEmbedder
        model = _get_qwen3vl_model()
        inputs = [{"text": norm, "instruction": VIDEO_EMBED_INSTRUCTION}]
        with torch.inference_mode():
            emb = model.process(inputs)  # (1, 2048) torch tensor, L2-normalized
        # Truncate to 1024d (MRL) + re-normalize
        emb = emb[:, :1024]
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        emb_np = emb.cpu().float().numpy()

    cache.set_embedding(f"video|{cache_key}", emb_np)
    return emb_np.tolist()[0]


@app.route('/api/search-video', methods=['POST'])
def search_video():
    """Text-to-video search using Qwen3-VL-Embedding."""
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 10)

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    t_start = time.perf_counter()

    # 1. Result cache check
    cache_key = f"{normalize_query(query)}|k={top_k}"
    cached = cache.get_results('video', cache_key, top_k)
    if cached is not None:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return jsonify({'results': cached, 'cache': 'HIT', 'latency_ms': round(elapsed_ms, 2)})

    # 2. Embed + search qwen3vl collection
    try:
        query_vector = embed_text_for_video(query)
    except Exception as e:
        return jsonify({'error': f'Failed to embed query: {e}'}), 500

    try:
        search_response = qdrant.query_points(
            collection_name=VIDEO_COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            search_params=_search_params(),
        )
        search_result = search_response.points
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    results = []
    for rank, hit in enumerate(search_result):
        payload = hit.payload or {}
        vid_id = payload.get('image_id', str(hit.id))
        # Strip Kaggle mount + dataset-slug prefixes so the URL works
        # both on Kaggle and locally (serve_video has its own fallback too).
        vid_relative = clean_image_path(payload.get('image_path', vid_id))
        for prefix in ("datasets/anhpham2710/my-videos-test/", "datasets/", "anhpham2710/my-videos-test/"):
            if vid_relative.startswith(prefix):
                vid_relative = vid_relative[len(prefix):]
                break
        results.append({
            'rank': rank + 1,
            'video_id': vid_id,
            'video_url': f"/videos/{vid_relative}",
            'score': hit.score,
        })

    cache.set_results('video', cache_key, results, top_k)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    return jsonify({'results': results, 'cache': 'MISS', 'latency_ms': round(elapsed_ms, 2)})


@app.route('/videos/<path:filepath>')
def serve_video(filepath):
    """Serve a video file from DATASET_ROOT.

    The notebook stored video_ids relative to /kaggle/input/datasets/<slug>/,
    so the request path may carry a Kaggle-specific prefix like
    "datasets/anhpham2710/my-videos-test/test/...". Strip known prefixes
    and try a few candidate roots; return the first match.
    """
    # Strip Kaggle mount / dataset-slug prefix from the front of the path
    p = filepath.replace("\\", "/")
    for prefix in (
        "datasets/anhpham2710/my-videos-test/",
        "datasets/",
        "anhpham2710/my-videos-test/",
    ):
        if p.startswith(prefix):
            p = p[len(prefix):]
            break

    candidates = [
        DATASET_ROOT / p,
        DATASET_ROOT / "videos" / p,
        DATASET_ROOT / "videos" / "UCF101_subset" / p,
    ]
    for full_path in candidates:
        if full_path.exists() and full_path.is_file():
            response = send_from_directory(full_path.parent, full_path.name, conditional=True)
            response.headers['Cache-Control'] = 'public, max-age=86400, immutable'
            response.headers['Accept-Ranges'] = 'bytes'   # required for HTML5 video seek
            return response
    return jsonify({'error': 'Video not found', 'tried': [str(x) for x in candidates]}), 404


@app.route('/images/<path:filepath>')
def serve_image(filepath):
    # filepath is relative to DATASET_ROOT (e.g., 'animals/cat/001.jpg').
    # Some Qdrant records were generated when the dataset was mounted flat
    # (e.g. 'cat/001.jpg' with no 'animals/' segment), so we fall back to
    # prepending 'animals/' if the direct path is missing.
    full_path = DATASET_ROOT / filepath
    if not full_path.exists():
        alt = DATASET_ROOT / "animals" / filepath
        if alt.exists():
            full_path = alt
    if not full_path.exists():
        return jsonify({'error': 'Image not found', 'tried': [str(full_path)]}), 404
    response = send_from_directory(full_path.parent, full_path.name)
    # Images are content-addressed via image_id (UUID) — safe to cache for 1 day
    response.headers['Cache-Control'] = 'public, max-age=86400, immutable'
    return response

if __name__ == '__main__':
    import os
    port = int(os.environ.get("APP_PORT", "5000"))
    # Production WSGI server: multi-threaded, no debug reloader, Windows-friendly.
    # Waitress has no external deps and supports threads (Flask dev server doesn't).
    from waitress import serve
    print(f"\n[BOOT] Waitress serving on 0.0.0.0:{port} (threads=4)")
    serve(app, host='0.0.0.0', port=port, threads=4, ident='mmis')
