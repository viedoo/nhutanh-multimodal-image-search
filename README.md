# Kaggle Image Embedding Pipeline

Automated pipeline for generating image embeddings using vision models (SigLIP2, DINOv3) on Kaggle's free GPU infrastructure. Features 3-tier secret management (Kaggle Secrets, Private Dataset, Local .env) for seamless CI/CD integration.

## Features

- **Multi-Model Support**: SigLIP2 and DINOv3 with optimized inference
- **GPU Auto-Detection**: Adaptive batch sizing for T4/T4x2 (2x faster with dual GPUs)
- **Production-Ready Secrets**: 3-tier fallback (Kaggle Secrets → Private Dataset → .env)
- **Full Automation**: Push → Monitor → Download in one command
- **Zero Configuration**: Auto-detects dataset paths and hardware
- **Timestamped Outputs**: HDF5 embeddings + FAISS indices with collision-free naming

## Quick Start

```bash
# Install dependencies
uv sync

# Run full pipeline (push + monitor + download)
uv run python kaggle_pipeline.py --type siglip2 --auto-download
```

**Result**: `result/siglip2_embeddings_20260611_HHMMSS.hdf5` (3.1 MB) + FAISS index in ~2 minutes.

## Prerequisites

| Requirement | Setup |
|------------|-------|
| **Kaggle API** | Download `kaggle.json` from [kaggle.com/settings](https://kaggle.com/settings) → Place in `~/.kaggle/` (chmod 600) |
| **Python 3.11+** | Install [uv](https://github.com/astral-sh/uv): `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **HF Token** (DINOv3 only) | Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd test_kaggle

# Install dependencies with uv
uv sync

# Configure Kaggle API (Linux/Mac)
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
move kaggle.json %USERPROFILE%\.kaggle\
```

## Secret Management (3-Tier Fallback)

### Option 1: Kaggle Secrets (Manual Runs)
1. Open notebook on Kaggle → **Add-ons** → **Secrets**
2. Add `HF_TOKEN` with your HuggingFace token
3. Toggle **"Attach to notebook"**

### Option 2: Private Dataset (API Automation) ⭐ Recommended for CI/CD

```bash
# 1. Create secrets directory
mkdir secrets
echo "your_hf_token_here" > secrets/HF_TOKEN.txt

# 2. Create dataset metadata
cat > secrets/dataset-metadata.json << EOF
{
  "title": "my-hf-secrets",
  "id": "YOUR_USERNAME/my-hf-secrets",
  "licenses": [{"name": "unknown"}]
}
EOF

# 3. Upload as private dataset
cd secrets
kaggle datasets create -p .
```

The pipeline automatically attaches this private dataset for DINOv3 notebooks.

### Option 3: Local Development (.env)

```bash
# Copy template
cp .env.example .env

# Edit with your token
echo "HF_TOKEN=hf_your_token_here" > .env
```

**How it works**: Code checks in order: Kaggle Secrets → Private Dataset → .env file

## Usage

### Full Automated Pipeline

```bash
# SigLIP2 (no secrets required)
uv run python kaggle_pipeline.py --type siglip2 --auto-download

# DINOv3 (requires HF_TOKEN setup)
uv run python kaggle_pipeline.py --type dinov3 --auto-download

# Custom monitoring intervals
uv run python kaggle_pipeline.py --type siglip2 --auto-download \
  --interval 60 --max-wait 1800
```

### Partial Workflows

```bash
# Push only (manual check later)
uv run python kaggle_pipeline.py --type siglip2 --push-only

# Download existing outputs
uv run python kaggle_download.py --type siglip2

# Check status
uv run python kaggle_download.py --type siglip2 --check-status
```

### Custom Dataset

```bash
uv run python kaggle_pipeline.py --type siglip2 \
  --dataset your-username/your-dataset \
  --auto-download
```

## Project Structure

```
├── notebook/
│   ├── siglip2-embed.ipynb      # Optimized SigLIP2 notebook (T4x2 support)
│   └── dino-v3-embed.ipynb      # DINOv3 with 3-tier secret fallback
├── result/                       # Pipeline outputs (gitignored)
├── secrets/                      # Private dataset source (gitignored)
│   ├── HF_TOKEN.txt
│   └── dataset-metadata.json
├── kaggle_pipeline.py            # Main pipeline (push + monitor + download)
├── kaggle_download.py            # Standalone download utility
├── .env.example                  # Local dev secret template
├── pyproject.toml                # uv dependencies
└── README.md
```

## Output Format

Files saved to `result/` with timestamp:

| Model | Embeddings | FAISS Index | Execution Time |
|-------|-----------|-------------|----------------|
| SigLIP2 | 3.1 MB (HDF5) | 2.9 MB (pickle) | ~90 sec |
| DINOv3 | 3.1 MB (HDF5) | 2.9 MB (pickle) | ~90 sec |

**HDF5 Structure**:
```python
import h5py
with h5py.File('result/siglip2_embeddings_*.hdf5', 'r') as f:
    embeddings = f['embeddings'][:]      # (N, 768) float32
    image_ids = f['image_ids'][:]        # (N,) utf-8 strings
    image_paths = f['image_paths'][:]    # (N,) utf-8 strings
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **ConnectionError: Connection error trying to communicate with service** | Kaggle Secrets don't work with API push. Use Private Dataset (see Secret Management Option 2) |
| **401 Unauthorized (DINOv3)** | Setup HF_TOKEN using one of the 3 methods above |
| **"Dataset not found"** | Update `dataset_sources` in pipeline or use `--dataset` flag |
| **Kernel times out** | Normal for large datasets. Check manually or use `kaggle_download.py` later |
| **Import kaggle_secrets failed** | Expected locally. Code falls back to .env automatically |

## Performance Notes

- **T4 GPU (single)**: Batch size 192, ~90 seconds for 1000 images
- **T4x2 GPU (dual)**: Batch size 256, ~60 seconds (2x GPUs detected automatically)
- **FP16 Precision**: Enabled on GPU for 2x speedup with no accuracy loss
- **DataLoader**: Async prefetching with persistent workers for optimal throughput

## Advanced: GitHub Actions CI/CD

Use the private dataset approach for automated workflows:

```yaml
# .github/workflows/embeddings.yml
name: Generate Embeddings
on: [push]
jobs:
  embed:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: astral-sh/setup-uv@v1
      - run: uv sync
      - name: Setup Kaggle
        run: |
          mkdir -p ~/.kaggle
          echo '${{ secrets.KAGGLE_JSON }}' > ~/.kaggle/kaggle.json
      - name: Run Pipeline
        run: uv run python kaggle_pipeline.py --type siglip2 --auto-download
```

**No HF_TOKEN in GitHub Secrets needed** - private dataset handles it automatically.

## Security Notes

- **Private datasets**: Only you can access, even if notebook is public
- **Token exposure**: Never hardcode tokens in notebooks or commit .env
- **Forked notebooks**: Users must supply their own credentials
- **Git safety**: `.env` and `secrets/` are gitignored

## Model Details

### SigLIP2 (`google/siglip2-base-patch16-naflex`)
- **Embedding dimension**: 768
- **License**: Apache 2.0
- **Strengths**: Fast, no token required, excellent for general images

### DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`)
- **Embedding dimension**: 768
- **License**: Apache 2.0 (gated - requires HF acceptance)
- **Strengths**: State-of-art vision representations, better for fine-grained similarity

## Development

```bash
# Local embedding generation (CPU)
uv run python prepare_embeddings.py

# Image search web app
uv run python app.py
# Open http://localhost:5000

# Benchmark models
uv run python benchmark_siglip2.py
uv run python benchmark_dinov3.py
```

## License

MIT

## Contributing

PRs welcome. Please ensure:
1. Code passes `uv run pytest` (if tests exist)
2. No secrets committed
3. Update README for new features

## Support

- Kaggle API docs: https://github.com/Kaggle/kaggle-api
- Factory AI docs: https://docs.factory.ai
- Issues: [GitHub Issues](your-repo-url/issues)
