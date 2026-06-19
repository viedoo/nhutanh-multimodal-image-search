"""
Centralized configuration for Kaggle pipeline
All paths and settings in one place for easy maintenance
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
RESULT_DIR = PROJECT_ROOT / "result"
DATASET_DIR = PROJECT_ROOT / "dataset"

# Kaggle settings
KAGGLE_DATASET_MOUNT_NAME = "my-dataset"  # This will be the folder name in /kaggle/input/
KAGGLE_HF_SECRETS_SLUG = "my-hf-secrets"  # Private dataset that holds HF_TOKEN.txt

# Qdrant settings
QDRANT_URL = "http://localhost:6333"
# HNSW + quantization tuning (target = max speed, build HNSW early so even
# the 5400-image dev dataset uses ANN instead of brute-force)
QDRANT_HNSW_M = 16
QDRANT_HNSW_EF_CONSTRUCT = 200
QDRANT_FULL_SCAN_THRESHOLD = 1000
QDRANT_INDEXING_THRESHOLD = 1000
QDRANT_SEARCH_EF = 128                # search-time ef for KNN
QDRANT_QUANTILE = 0.99                # scalar quant outlier clamp
QDRANT_BATCH_SIZE = 2048              # points per upload_points() call
QDRANT_UPLOAD_PARALLEL = 4            # concurrent in-flight batches
QDRANT_OPTIMIZER_POLL_INTERVAL = 2.0  # seconds
QDRANT_OPTIMIZER_TIMEOUT = 300        # seconds
# Datasets that should NEVER be auto-detected as the image dataset
NOTEBOOK_DATASET_BLACKLIST = (
    KAGGLE_HF_SECRETS_SLUG,
    "kaggle-data-overlay",
)

# Model configurations
MODELS = {
    "siglip2": {
        "notebook_file": "siglip2-embed.ipynb",
        "kernel_slug_suffix": "siglip2-embed",
        "model_id": "google/siglip2-base-patch16-naflex",
        "requires_hf_token": False,
        "embedding_dim": 768,
    },
    "dinov3": {
        "notebook_file": "dino-v3-embed.ipynb",
        "kernel_slug_suffix": "dinov3-embed",
        "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "requires_hf_token": True,
        "embedding_dim": 768,
    },
    "dinov3_dense": {
        "notebook_file": "dino-v3-dense-embed.ipynb",
        "kernel_slug_suffix": "dinov3-dense-embed",
        "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "requires_hf_token": True,
        "embedding_dim": 768,
    },
    "qwen3vl": {
        "notebook_file": "qwen3-vl-embed.ipynb",
        "kernel_slug_suffix": "qwen3-vl-embed",
        "model_id": "Qwen/Qwen3-VL-Embedding-2B",
        "requires_hf_token": False,
        "embedding_dim": 1024,
        "uses_qwen3vl_embedder": True,
    },
}

# Pipeline defaults
DEFAULT_MONITOR_INTERVAL = 10  # seconds (was 30s; reduced to minimize poll-overhead lag)
DEFAULT_MAX_WAIT = 1200  # seconds (20 minutes)
DEFAULT_BATCH_CHECK_DELAY = 5  # seconds before starting monitoring

# Output file patterns
OUTPUT_PATTERNS = {
    "embeddings": "{model_type}_embeddings_{timestamp}.hdf5",
}

# Dataset validation
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

# Video search settings (text-to-video via Qwen3-VL-Embedding)
VIDEO_COLLECTION = "qwen3vl"
VIDEO_EMBED_INSTRUCTION = "Represent the user's input for video retrieval."

def get_model_config(model_type: str) -> dict:
    """Get configuration for a specific model type"""
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODELS.keys())}")
    return MODELS[model_type]

def get_kaggle_dataset_path(username: str, dataset_name: str = None) -> str:
    """
    Get the Kaggle dataset path that will be used in notebooks
    
    Args:
        username: Kaggle username
        dataset_name: Dataset name (defaults to KAGGLE_DATASET_MOUNT_NAME)
    
    Returns:
        Path string like '/kaggle/input/my-dataset'
    """
    if dataset_name is None:
        dataset_name = KAGGLE_DATASET_MOUNT_NAME
    return f"/kaggle/input/{dataset_name}"

def get_dataset_slug(username: str, dataset_name: str = None) -> str:
    """
    Get the full Kaggle dataset slug
    
    Args:
        username: Kaggle username
        dataset_name: Dataset name (defaults to KAGGLE_DATASET_MOUNT_NAME)
    
    Returns:
        Slug string like 'username/my-dataset'
    """
    if dataset_name is None:
        dataset_name = KAGGLE_DATASET_MOUNT_NAME
    return f"{username}/{dataset_name}"
