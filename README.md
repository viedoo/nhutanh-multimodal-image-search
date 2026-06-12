# Multimodal Image Search Engine

Hệ thống tìm kiếm ảnh đa phương thức sử dụng **SigLIP2** (tìm theo text) và **DINOv3** (tìm theo ảnh tương tự / chất liệu). Embedding được tạo trên Kaggle GPU miễn phí và phục vụ qua giao diện Flask.

## Tính năng

- **Text Search** — Gõ mô tả tiếng Anh, tìm ảnh khớp nhất (SigLIP2 + FAISS)
- **Similar Image Search** — Click "Find Similar", tìm ảnh trông giống nhau (DINOv3)
- **Material / Texture Search** — Click "Find Material", tìm ảnh cùng chất liệu / màu sắc (DINOv3 Dense)
- **Kaggle GPU Pipeline** — Tự động push notebook → chờ chạy → tải embedding về máy
- **3-Tier Secret Management** — Kaggle Secrets → Private Dataset → Local `.env`

---

## Cấu trúc project

```
├── notebook/
│   ├── siglip2-embed.ipynb          # Notebook tạo embedding SigLIP2 (text search)
│   ├── dino-v3-embed.ipynb          # Notebook tạo embedding DINOv3 (similar search)
│   └── dino-v3-dense-embed.ipynb   # Notebook tạo embedding DINOv3 Dense (material search)
│
├── dataset/
│   └── animals/                     # Ảnh dataset, cấu trúc: animals/<class>/<file>.jpg
│
├── result/                          # Output của pipeline (gitignored)
│   ├── siglip2_embeddings_*.hdf5
│   ├── siglip2_faiss_index_*.pkl
│   ├── dinov3_embeddings_*.hdf5
│   ├── dinov3_faiss_index_*.pkl
│   ├── dinov3_dense_embeddings_*.hdf5
│   └── dinov3_dense_faiss_index_*.pkl
│
├── secrets/                         # HF Token lưu riêng (gitignored)
│   ├── HF_TOKEN.txt
│   └── dataset-metadata.json
│
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/index.html
│
├── app.py                  # Flask web app — chạy để dùng UI tìm kiếm
├── config.py               # Cấu hình model, đường dẫn (dùng chung bởi pipeline)
├── kaggle_pipeline.py      # Pipeline chính: push → monitor → download
├── kaggle_download.py      # Tải output từ kernel đã chạy xong
├── kaggle_dataset.py       # Upload dataset ảnh lên Kaggle
├── run.py                  # Orchestrator: gộp tất cả bước vào 1 lệnh
├── .env.example            # Template secret cho dev local
└── pyproject.toml
```

---

## Yêu cầu cài đặt

| Thứ cần có | Cách lấy |
|---|---|
| **Kaggle API key** | [kaggle.com/settings](https://kaggle.com/settings) → API → Download `kaggle.json` → đặt vào `%USERPROFILE%\.kaggle\` |
| **Python 3.11+** + **uv** | `pip install uv` hoặc xem [astral.sh/uv](https://astral.sh/uv) |
| **HF Token** (chỉ cần cho DINOv3) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

```bash
# Cài dependencies
uv sync
```

---

## Các pipeline và cách chạy

Dự án có **4 pipeline** riêng biệt, mỗi cái làm một việc:

---

### Pipeline 1 — Upload dataset lên Kaggle

**Script:** `kaggle_dataset.py`  
**Khi nào dùng:** Lần đầu tiên, hoặc khi dataset thay đổi (thêm/xóa ảnh).

```bash
# Upload thư mục dataset/ lên Kaggle (tạo mới hoặc update version)
uv run python kaggle_dataset.py --folder dataset

# Chỉ định tên slug cụ thể
uv run python kaggle_dataset.py --folder dataset --slug ten-dataset-cua-ban

# Upload public
uv run python kaggle_dataset.py --folder dataset --public
```

**Kết quả:** Dataset có tại `https://www.kaggle.com/datasets/<username>/<folder-name>`

---

### Pipeline 2 — Tạo embedding trên Kaggle GPU

**Script:** `kaggle_pipeline.py`  
**Khi nào dùng:** Khi cần tạo/tái tạo embedding (dataset mới, hoặc đổi model).

```bash
# SigLIP2 — dùng cho Text Search (không cần HF Token)
uv run python kaggle_pipeline.py --type siglip2 --dataset-slug <username>/dataset --auto-download

# DINOv3 — dùng cho Similar Image Search (cần HF Token)
uv run python kaggle_pipeline.py --type dinov3 --dataset-slug <username>/dataset --auto-download

# DINOv3 Dense — dùng cho Material/Texture Search (cần HF Token)
uv run python kaggle_pipeline.py --type dinov3_dense --dataset-slug <username>/dataset --auto-download
```

**Pipeline tự động làm 3 bước:**
1. **STEP 0** *(tuỳ chọn)* — Upload dataset nếu dùng `--upload-dataset`
2. **STEP 1** — Push notebook lên Kaggle, chạy với GPU T4
3. **STEP 2** — Theo dõi tiến trình (check mỗi 30s, timeout sau 20 phút)
4. **STEP 3** — Tải file `.hdf5` và `.pkl` về thư mục `result/`

**Tuỳ chỉnh:**
```bash
# Chỉ push, không chờ (kiểm tra thủ công sau)
uv run python kaggle_pipeline.py --type siglip2 --dataset-slug <username>/dataset --push-only

# Tăng thời gian chờ và giảm interval check
uv run python kaggle_pipeline.py --type siglip2 --dataset-slug <username>/dataset \
  --auto-download --interval 60 --max-wait 3600
```

---

### Pipeline 3 — Tải embedding về (khi kernel đã chạy xong)

**Script:** `kaggle_download.py`  
**Khi nào dùng:** Khi pipeline bị timeout hoặc đã push trước đó và giờ muốn tải về.

```bash
# Tải embedding về result/
uv run python kaggle_download.py --type siglip2
uv run python kaggle_download.py --type dinov3
uv run python kaggle_download.py --type dinov3_dense

# Kiểm tra trạng thái kernel trước khi tải
uv run python kaggle_download.py --type siglip2 --check-status
```

---

### Pipeline 4 — Orchestrator (gộp tất cả)

**Script:** `run.py`  
**Khi nào dùng:** Muốn chạy toàn bộ quy trình từ đầu đến cuối bằng 1 lệnh.

```bash
# Upload dataset + chạy pipeline + tải về + khởi động server
uv run python run.py --type siglip2 --upload-dataset dataset

# Chỉ upload dataset
uv run python run.py --upload-dataset dataset --upload-only

# Chạy pipeline với dataset đã có sẵn trên Kaggle
uv run python run.py --type siglip2 --dataset-slug <username>/dataset

# Chỉ khởi động server (embedding đã có)
uv run python run.py --server-only
```

---

### Chạy web app tìm kiếm

```bash
uv run python app.py
# Mở trình duyệt: http://localhost:5000
```

**Cách dùng UI:**
| Thao tác | Kết quả |
|---|---|
| Gõ mô tả (VD: `sleeping cat`) → Search | Tìm ảnh theo text |
| Click **Find Similar** trên một ảnh | Tìm ảnh trông giống |
| Click **Find Material** trên một ảnh | Tìm ảnh cùng chất liệu/màu sắc |

---

## Quy trình đầy đủ từ đầu

```bash
# Bước 1: Cài dependencies
uv sync

# Bước 2: Upload dataset lên Kaggle (chỉ làm 1 lần)
uv run python kaggle_dataset.py --folder dataset

# Bước 3: Tạo embedding SigLIP2
uv run python kaggle_pipeline.py --type siglip2 \
  --dataset-slug <username>/dataset --auto-download

# Bước 4: Tạo embedding DINOv3 (cần setup HF Token trước)
uv run python kaggle_pipeline.py --type dinov3 \
  --dataset-slug <username>/dataset --auto-download

# Bước 5: Tạo embedding DINOv3 Dense
uv run python kaggle_pipeline.py --type dinov3_dense \
  --dataset-slug <username>/dataset --auto-download

# Bước 6: Khởi động web app
uv run python app.py
```

---

## Setup HF Token cho DINOv3

DINOv3 cần HuggingFace Token vì model là gated. Có 3 cách:

### Cách 1: Private Dataset trên Kaggle (khuyến nghị cho API)
```bash
mkdir secrets
echo "hf_your_token_here" > secrets/HF_TOKEN.txt

# Tạo metadata
echo '{"title":"my-hf-secrets","id":"<username>/my-hf-secrets","licenses":[{"name":"unknown"}]}' > secrets/dataset-metadata.json

# Upload lên Kaggle (private)
cd secrets && kaggle datasets create -p .
```

### Cách 2: Local `.env` (cho dev)
```bash
cp .env.example .env
# Chỉnh .env: HF_TOKEN=hf_your_token_here
```

### Cách 3: Kaggle Secrets (chỉ dùng khi chạy tay trên Kaggle)
Vào notebook trên Kaggle → Add-ons → Secrets → Thêm `HF_TOKEN`

---

## Output format

Files được lưu vào `result/` với timestamp tránh trùng tên:

```
result/
├── siglip2_embeddings_20260612_164950.hdf5     (~16 MB)
├── siglip2_faiss_index_20260612_164950.pkl     (~16 MB)
├── dinov3_embeddings_20260612_170536.hdf5      (~16 MB)
├── dinov3_faiss_index_20260612_170536.pkl      (~16 MB)
├── dinov3_dense_embeddings_20260612_170849.hdf5 (~16 MB)
└── dinov3_dense_faiss_index_20260612_170849.pkl (~16 MB)
```

**Đọc HDF5:**
```python
import h5py
with h5py.File('result/siglip2_embeddings_*.hdf5', 'r') as f:
    embeddings = f['embeddings'][:]   # shape: (N, 768) float32
    image_ids  = f['image_ids'][:]    # VD: b'cat/8af49688fa.jpg'
    image_paths = f['image_paths'][:] # VD: b'/kaggle/input/dataset/cat/...'
```

---

## Troubleshooting

| Lỗi | Cách xử lý |
|---|---|
| `401 Unauthorized` (DINOv3) | Setup HF_TOKEN theo 1 trong 3 cách ở trên |
| `Dataset not found` | Kiểm tra slug đúng chưa, dùng `--dataset-slug username/ten-dataset` |
| Pipeline timeout | Kernel vẫn đang chạy. Dùng `kaggle_download.py --check-status` sau |
| Ảnh 404 trên web | Kiểm tra dataset nằm đúng tại `dataset/animals/<class>/*.jpg` |
| `UnicodeEncodeError` khi download | Bình thường trên Windows, file embedding vẫn được tải đúng |

---

## License

MIT
