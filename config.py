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
}

# Pipeline defaults
DEFAULT_MONITOR_INTERVAL = 30  # seconds
DEFAULT_MAX_WAIT = 1200  # seconds (20 minutes)
DEFAULT_BATCH_CHECK_DELAY = 5  # seconds before starting monitoring

# Output file patterns
OUTPUT_PATTERNS = {
    "embeddings": "{model_type}_embeddings_{timestamp}.hdf5",
    "faiss_index": "{model_type}_faiss_index_{timestamp}.pkl",
}

# Dataset validation
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

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
