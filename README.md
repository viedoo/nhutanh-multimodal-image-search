# Multimodal Image Search Engine

Hệ thống tìm kiếm ảnh đa phương thức sử dụng **SigLIP2** (tìm theo text) và **DINOv3** (tìm theo ảnh tương tự / chất liệu). Embedding được tạo trên Kaggle GPU miễn phí và phục vụ qua giao diện Flask.

## Tính năng

- **Text Search** — Gõ mô tả tiếng Anh, tìm ảnh khớp nhất (SigLIP2 + FAISS)
- **Similar Image Search** — Click "Find Similar", tìm ảnh trông giống nhau (DINOv3)
- **Material / Texture Search** — Click "Find Material", tìm ảnh cùng chất liệu / màu sắc (DINOv3 Dense)
- **Kaggle GPU Pipeline** — Tự động push notebook → chờ chạy → tải embedding về máy
- **3-Tier Secret Management** — Kaggle Secrets → Private Dataset → Local `.env`

---

## Screenshots

**Trang chủ** — giao diện search với input text + kết quả grid:
![Homepage](screenshots/homepage.png)

**Text Search** — gõ "sleeping cat" → SigLIP2 tìm ảnh khớp nhất:
![Text search results](screenshots/text_search_results.png)

**Similar Image Search** — click "Find Similar" trên 1 ảnh → DINOv3 tìm ảnh trông giống:
![Similar search with reference](screenshots/similar_search_with_reference.png)

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
│   ├── dinov3_embeddings_*.hdf5
│   └── dinov3_dense_embeddings_*.hdf5
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
├── cache.py                # Redis cache + in-memory fallback
├── config.py               # Cấu hình model, đường dẫn (dùng chung bởi pipeline)
├── qdrant_ingest.py        # Nạp embeddings vào Qdrant (idempotent, có HNSW + ScalarQuant)
├── kaggle_pipeline.py      # Pipeline chính: push → monitor → download
├── kaggle_download.py      # Tải output từ kernel đã chạy xong
├── kaggle_dataset.py       # Upload dataset ảnh lên Kaggle
├── run.py                  # Orchestrator: gộp tất cả bước vào 1 lệnh
├── utils.py                # Shared helpers (UUID, path cleanup, cache hashing)
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

## Quick Start (chạy từ đầu đến cuối)

Đây là flow ngắn nhất. Chi tiết từng bước xem ở các section dưới.

**Lần đầu tiên:**
```bash
# 1. Cài deps + setup env
uv sync
cp .env.example .env          # rồi điền HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY

# 2. Khởi động Qdrant (mở terminal riêng, để chạy nền)
cd F:\qdrant && .\qdrant.exe

# 3. Khởi động Redis (WSL, optional — app vẫn chạy nếu Redis down)
wsl -d Ubuntu-24.04 -- sudo service redis-server start

# 4. Upload dataset lên Kaggle (1 lần duy nhất)
uv run python kaggle_dataset.py --folder dataset

# 5. Tạo embedding cho 3 model (mỗi model ~5-10 phút trên Kaggle T4)
uv run python kaggle_pipeline.py --type siglip2    --dataset-slug <username>/<dataset> --auto-download
uv run python kaggle_pipeline.py --type dinov3     --dataset-slug <username>/<dataset> --auto-download
uv run python kaggle_pipeline.py --type dinov3_dense --dataset-slug <username>/<dataset> --auto-download

# 6. Nạp embeddings vào Qdrant (idempotent — chạy lại OK)
uv run python qdrant_ingest.py

# 7. Khởi động web server (Waitress, mặc định port 5000)
uv run python app.py
# → Mở http://localhost:5000
```

**Các lần sau (dataset/model không đổi):**
```bash
# 1. Start Qdrant + Redis (nếu chưa chạy)
# 2. Chỉ cần start server
uv run python app.py
```

**Khi muốn thay đổi gì đó:**
| Tình huống | Cần chạy lại |
|---|---|
| Thêm/xoá ảnh trong `dataset/` | Bước 4 (re-upload) + Bước 5 (re-embed) + Bước 6 (re-ingest) |
| Đổi model SigLIP2/DINOv3 | Bước 5 của model đó + Bước 6 với `--force` |
| Đổi cấu hình HNSW/Quantization | Bước 6 với `--force` (wipe + re-ingest) |
| Cache cũ/mới sau re-ingest | Bước 6 với `--clear-redis` |

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

### Khởi động Qdrant Vector Database (Windows)

Hệ thống sử dụng **Qdrant** làm Vector Database chính để tìm kiếm hàng triệu ảnh siêu nhanh (chỉ tốn ~1.2GB RAM nhờ Scalar Quantization). Cấu hình mặc định trong `qdrant_ingest.py`:

- **HNSW**: `m=16`, `ef_construct=200`, `full_scan_threshold=1000` (build chỉ mục ngay từ 1000 vector trở lên, không phải 10k như mặc định)
- **Quantization**: Scalar INT8, `always_ram=True` (~4× giảm RAM)
- **Payload index** trên `image_id` để lookup O(log n)
- **gRPC** (`prefer_grpc=True`) — kết nối persistent, **15x nhanh hơn** HTTP (search latency: 2200ms → 140-170ms)

**Bước 1: Tải và giải nén (Làm 1 lần)**
1. Tải file `qdrant-x86_64-pc-windows-msvc.zip` từ [Qdrant GitHub Releases](https://github.com/qdrant/qdrant/releases/latest).
2. Giải nén vào ổ F (hoặc ổ nào còn trống nhiều), ví dụ: `F:\qdrant`.

**Bước 2: Chạy Qdrant (Mỗi lần bật máy)**
```powershell
# Mở PowerShell, cd vào thư mục đã giải nén
cd F:\qdrant
.\qdrant.exe
```
> **Lưu ý:** Qdrant sẽ tự tạo thư mục `F:\qdrant\storage` để lưu dữ liệu. Bạn có thể xem giao diện quản lý DB tại http://localhost:6333/dashboard

**Bước 3: Nạp embeddings vào Qdrant (idempotent — chạy lại nhiều lần OK)**
```bash
# Lần đầu hoặc khi có embeddings mới
uv run python qdrant_ingest.py

# Bắt buộc wipe + ingest lại từ đầu (vd khi đổi model)
uv run python qdrant_ingest.py --force

# Đồng thời xoá Redis cache (an toàn sau khi re-ingest)
uv run python qdrant_ingest.py --force --clear-redis
```
> Mặc định `qdrant_ingest.py` sẽ **skip** nếu collection đã đủ số vector, dùng `--force` khi muốn xoá hết và ingest lại.

---

### Khởi động Redis Cache (WSL)

Redis chạy trong WSL để tăng tốc search. Cần khởi động **mỗi lần bật máy**:

```bash
# Lần đầu — cài Redis (chỉ làm 1 lần)
wsl -d Ubuntu-24.04
sudo apt-get update && sudo apt-get install -y redis-server
sudo sed -i 's/^bind 127.0.0.1 -::1/bind 0.0.0.0/' /etc/redis/redis.conf

# Mỗi lần khởi động máy — start Redis
wsl -d Ubuntu-24.04 -- sudo service redis-server start

# Kiểm tra đang chạy (phải ra PONG)
wsl -d Ubuntu-24.04 -- redis-cli ping
```

> **Lưu ý:** Nếu không có Redis, hệ thống tự động fallback sang in-memory cache — app vẫn chạy bình thường nhưng không chia sẻ cache giữa các process.

---

### Chạy web app tìm kiếm

App chạy bằng **Waitress** (production WSGI server, 4 threads) thay vì Flask dev server — ổn định hơn và handle được concurrent requests tốt hơn.

```bash
# Bước 1: Đảm bảo Qdrant đang chạy (cửa sổ qdrant.exe đang mở)
# Bước 2: Đảm bảo Redis đang chạy (nếu dùng WSL)
wsl -d Ubuntu-24.04 -- redis-cli ping

# Bước 3: Khởi động server (Waitress ở http://localhost:5000)
uv run python app.py
# Mở trình duyệt: http://localhost:5000

# Hoặc chạy trực tiếp bằng waitress (tương đương, hữu ích cho systemd/supervisor):
uv run waitress-serve --listen=*:5000 --threads=4 'app:create_app()'
```

**Cách dùng UI:**
| Thao tác | Kết quả |
|---|---|
| Gõ mô tả (VD: `sleeping cat`) → Search | Tìm ảnh theo text (SigLIP2) |
| Click **Find Similar** trên một ảnh | Tìm ảnh trông giống (DINOv3) |
| Click **Find Material** trên một ảnh | Tìm ảnh cùng chất liệu/màu sắc (DINOv3 Dense) |

**Cache API:**
| Endpoint | Mô tả |
|---|---|
| `GET /healthz` | Health probe — trả 200 OK nếu Qdrant lên. Redis optional (200 degraded nếu Redis down) |
| `GET /api/cache/stats` | Xem thống kê cache (hits, misses, memory) |
| `POST /api/cache/clear` | Xóa toàn bộ cache |

**Đổi port:**
```bash
# Mặc định 5000. Đặt biến môi trường APP_PORT trước khi chạy:
APP_PORT=8080 uv run python app.py
# hoặc dùng orchestrator
uv run python run.py --server-only --port 8080
```

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

# Bước 6: Khởi động Redis cache (WSL)
wsl -d Ubuntu-24.04 -- sudo service redis-server start

# Bước 7: Khởi động web app
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
├── siglip2_embeddings_20260617_170000.hdf5     (~16 MB cho 5400 ảnh)
├── dinov3_embeddings_20260617_171200.hdf5      (~16 MB)
└── dinov3_dense_embeddings_20260617_172400.hdf5 (~16 MB)
```

> Hệ thống không còn sinh file `.pkl` (FAISS đã được thay bằng Qdrant ở phiên bản 1.1). Mọi search chạy trực tiếp trên Qdrant collections (`siglip2`, `dinov3`, `dinov3_dense`).

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
