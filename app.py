from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from pathlib import Path
import h5py
import pickle

app = Flask(__name__)
CORS(app)

MODEL_ID = "google/siglip2-base-patch16-naflex"
DATA_DIR = Path("result")
DATASET_ROOT = Path(__file__).parent / "dataset"

print("Loading embeddings and indices...")

# Find latest embedding files (HDF5 format)
siglip_files = sorted(DATA_DIR.glob("siglip2_embeddings_*.hdf5"), reverse=True)
siglip_indices = sorted(DATA_DIR.glob("siglip2_faiss_index_*.pkl"), reverse=True)
dinov3_files = sorted(DATA_DIR.glob("dinov3_embeddings_*.hdf5"), reverse=True)
dinov3_indices = sorted(DATA_DIR.glob("dinov3_faiss_index_*.pkl"), reverse=True)
dinov3_dense_files = sorted(DATA_DIR.glob("dinov3_dense_embeddings_*.hdf5"), reverse=True)
dinov3_dense_indices = sorted(DATA_DIR.glob("dinov3_dense_faiss_index_*.pkl"), reverse=True)

if not siglip_files or not siglip_indices:
    raise FileNotFoundError(f"SigLIP2 embeddings not found in {DATA_DIR}/")

print(f"Loading SigLIP 2 from {siglip_files[0].name}...")
with h5py.File(siglip_files[0], 'r') as f:
    image_ids = [s.decode('utf-8') for s in f['image_ids'][:]]
    image_paths = [s.decode('utf-8') for s in f['image_paths'][:]]
with open(siglip_indices[0], 'rb') as f:
    siglip_index = pickle.load(f)
print(f"Loaded {len(image_ids)} images for text search")

# Build id→relative-path lookup from SigLIP metadata (shared across all endpoints)
# image_ids are stored as 'cat/001.jpg' but actual files are under 'animals/cat/001.jpg'
# Prepend 'animals/' if the path doesn't already start with it
def _make_rel_path(img_id: str) -> str:
    if img_id.startswith("animals/"):
        return img_id
    return f"animals/{img_id}"

id_to_path = {img_id: _make_rel_path(img_id) for img_id in image_ids}

# Keep file paths for lazy loading (memory optimization for large datasets)
siglip_embedding_file = siglip_files[0]
dinov3_embedding_file = None
dinov3_index = None
dinov3_ids = []
dinov3_id_to_idx = {}

if dinov3_files and dinov3_indices:
    print(f"Loading DINOv3 from {dinov3_files[0].name}...")
    dinov3_embedding_file = dinov3_files[0]
    with h5py.File(dinov3_embedding_file, 'r') as f:
        dinov3_ids = [s.decode('utf-8') for s in f['image_ids'][:]]
    with open(dinov3_indices[0], 'rb') as f:
        dinov3_index = pickle.load(f)
    dinov3_id_to_idx = {img_id: idx for idx, img_id in enumerate(dinov3_ids)}
    print(f"Loaded {len(dinov3_ids)} images for image search")
else:
    print("DINOv3 embeddings not found, image similarity search disabled")

# --- DINOv3 Dense (material / texture similarity) ---
dinov3_dense_embedding_file = None
dinov3_dense_index = None
dinov3_dense_ids = []
dinov3_dense_id_to_idx = {}

if dinov3_dense_files and dinov3_dense_indices:
    print(f"Loading DINOv3 Dense from {dinov3_dense_files[0].name}...")
    dinov3_dense_embedding_file = dinov3_dense_files[0]
    with h5py.File(dinov3_dense_embedding_file, 'r') as f:
        dinov3_dense_ids = [s.decode('utf-8') for s in f['image_ids'][:]]
    with open(dinov3_dense_indices[0], 'rb') as f:
        dinov3_dense_index = pickle.load(f)
    dinov3_dense_id_to_idx = {img_id: idx for idx, img_id in enumerate(dinov3_dense_ids)}
    print(f"Loaded {len(dinov3_dense_ids)} images for material/texture search")
else:
    print("DINOv3 Dense embeddings not found, material similarity search disabled")

print("Loading SigLIP 2 model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
device = "cpu"
print("SigLIP 2 model loaded and ready!")

@lru_cache(maxsize=1000)
def embed_text_cached(text: str) -> tuple:
    """Cache text embeddings to avoid re-computing (LRU cache optimization)"""
    inputs = processor(text=[text], return_tensors="pt", padding="max_length", max_length=64, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        text_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
        text_features = F.normalize(text_features, p=2, dim=-1)
    return tuple(text_features.cpu().numpy().flatten().tolist())

def embed_text(text: str) -> np.ndarray:
    """Convert cached tuple back to numpy array"""
    cached = embed_text_cached(text)
    return np.array(cached, dtype=np.float32).reshape(1, -1)

print("Ready for searches!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 10)
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    query_embedding = embed_text(query)
    scores, indices = siglip_index.search(query_embedding.astype(np.float32), top_k)
    
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        img_id = image_ids[idx]
        img_relative = id_to_path.get(img_id, _make_rel_path(img_id))
        results.append({
            'rank': rank + 1,
            'image_id': img_id,
            'image_url': f"/images/{img_relative}",
            'score': float(score)
        })
    
    return jsonify({'results': results})

@app.route('/api/search-similar', methods=['POST'])
def search_similar():
    if dinov3_index is None:
        return jsonify({'error': 'Image similarity search not available'}), 503
    
    data = request.json
    image_id = data.get('image_id', '')
    top_k = data.get('top_k', 10)
    
    if not image_id:
        return jsonify({'error': 'image_id is required'}), 400
    
    if image_id not in dinov3_id_to_idx:
        return jsonify({'error': 'Image not found'}), 404
    
    idx = dinov3_id_to_idx[image_id]
    
    # Lazy load only the needed embedding (memory optimization for large datasets)
    with h5py.File(dinov3_embedding_file, 'r') as f:
        query_embedding = f['embeddings'][idx:idx+1]
    
    scores, indices = dinov3_index.search(query_embedding.astype(np.float32), top_k + 1)
    
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        result_id = dinov3_ids[idx]
        if result_id == image_id:
            continue

        img_relative = id_to_path.get(result_id, result_id)
        results.append({
            'rank': len(results) + 1,
            'image_id': result_id,
            'image_url': f"/images/{img_relative}",
            'score': float(score)
        })
        if len(results) >= top_k:
            break
    
    return jsonify({'results': results})

@app.route('/api/search-similar-material', methods=['POST'])
def search_similar_material():
    """Material / texture similarity using DINOv3 dense patch-mean features."""
    if dinov3_dense_index is None:
        return jsonify({'error': 'Material similarity search not available (DINOv3 Dense embeddings not loaded)'}), 503

    data = request.json
    image_id = data.get('image_id', '')
    top_k = data.get('top_k', 10)

    if not image_id:
        return jsonify({'error': 'image_id is required'}), 400

    if image_id not in dinov3_dense_id_to_idx:
        return jsonify({'error': 'Image not found in dense index'}), 404

    idx = dinov3_dense_id_to_idx[image_id]

    # Lazy-load only the single query embedding (memory efficient)
    with h5py.File(dinov3_dense_embedding_file, 'r') as f:
        query_embedding = f['embeddings'][idx:idx+1]

    scores, indices = dinov3_dense_index.search(query_embedding.astype(np.float32), top_k + 1)

    results = []
    for rank, (res_idx, score) in enumerate(zip(indices[0], scores[0])):
        result_id = dinov3_dense_ids[res_idx]
        if result_id == image_id:
            continue

        img_relative = id_to_path.get(result_id, result_id)
        results.append({
            'rank': len(results) + 1,
            'image_id': result_id,
            'image_url': f"/images/{img_relative}",
            'score': float(score)
        })
        if len(results) >= top_k:
            break

    return jsonify({'results': results})


@app.route('/images/<path:filepath>')
def serve_image(filepath):
    # filepath is relative to DATASET_ROOT (e.g., 'animals/cat/001.jpg')
    full_path = DATASET_ROOT / filepath
    if not full_path.exists():
        return jsonify({'error': 'Image not found'}), 404
    return send_from_directory(full_path.parent, full_path.name)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
