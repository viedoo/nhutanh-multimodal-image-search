"""
Download UCF-101 subset (~171MB, 405 video clips) for video embedding test.

Dataset: aisuko/ucf101-subset (Apache 2.0 license)
Source: https://www.kaggle.com/datasets/aisuko/ucf101-subset
Original: https://huggingface.co/datasets/sayakpaul/ucf101-subset

Usage:
    uv run python download_videos.py                       # default: 100 videos
    uv run python download_videos.py --max-videos 200      # custom limit
    uv run python download_videos.py --dataset-slug <slug> # different Kaggle dataset
"""
import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


DEFAULT_DATASET_SLUG = "aisuko/ucf101-subset"
DATASET_DIR = Path(__file__).parent / "dataset" / "videos"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def get_secret(key: str, fallback_path: str = None, default=None) -> str:
    """Mirror of get_secret() used in Kaggle notebooks."""
    import os
    if fallback_path and Path(fallback_path).exists():
        return Path(fallback_path).read_text().strip()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    value = os.environ.get(key, default)
    return value


def download_dataset(slug: str, target_dir: Path) -> Path:
    """Download Kaggle dataset zip and extract to target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {slug} to {target_dir}...")

    api = KaggleApi()
    api.authenticate()

    # Download as zip (default for UCF101 subset)
    api.dataset_download_files(slug, path=str(target_dir), quiet=False)

    # Find downloaded zip
    zips = list(target_dir.glob("*.zip"))
    if not zips:
        # Try to find any archive
        archives = list(target_dir.glob("*.tar*"))
        if archives:
            return _extract_archive(archives[0], target_dir)
        raise FileNotFoundError(f"No archive downloaded to {target_dir}")

    return _extract_archive(zips[0], target_dir, delete_after=True)


def _extract_archive(archive: Path, target_dir: Path, delete_after: bool = False) -> Path:
    """Extract zip/tar/tar.gz to target_dir."""
    print(f"Extracting {archive.name} ({archive.stat().st_size / 1024 / 1024:.1f} MB)...")

    # Strip top-level wrapper dir if present (Kaggle datasets often add one)
    def _is_within_target(member_path: str) -> bool:
        # Avoid zip-slip
        resolved = (target_dir / member_path).resolve()
        return str(resolved).startswith(str(target_dir.resolve()))

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            for member in zf.infolist():
                if _is_within_target(member.filename):
                    zf.extract(member, target_dir)
    else:
        with tarfile.open(archive, "r:*") as tf:
            for member in tf.getmembers():
                if member.isfile() and _is_within_target(member.name):
                    tf.extract(member, target_dir)

    if delete_after:
        archive.unlink()
        print(f"  Removed archive {archive.name}")

    return target_dir


def limit_videos(target_dir: Path, max_videos: int) -> int:
    """Keep only the first N video files (move the rest to _extra/)."""
    videos = sorted(
        p for p in target_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(videos) <= max_videos:
        return len(videos)

    extra_dir = target_dir / "_extra"
    extra_dir.mkdir(exist_ok=True)
    for v in videos[max_videos:]:
        target = extra_dir / v.name
        # Handle name collisions by prefixing with parent dir
        if target.exists():
            target = extra_dir / f"{v.parent.name}_{v.name}"
        shutil.move(str(v), str(target))

    print(f"  Moved {len(videos) - max_videos} extra videos to {extra_dir}/")
    return max_videos


def main():
    parser = argparse.ArgumentParser(description="Download UCF-101 subset for video embedding test")
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG,
                        help=f"Kaggle dataset slug (default: {DEFAULT_DATASET_SLUG})")
    parser.add_argument("--max-videos", type=int, default=100,
                        help="Cap total videos kept (default: 100)")
    parser.add_argument("--target-dir", type=Path, default=DATASET_DIR,
                        help=f"Target directory (default: {DATASET_DIR})")
    parser.add_argument("--keep-all", action="store_true",
                        help="Keep all 405 videos (no cap)")
    args = parser.parse_args()

    # Get HF token (kaggle doesn't need it but mirror the notebook pattern)
    _ = get_secret("HF_TOKEN", fallback_path="secrets/HF_TOKEN.txt")

    target = args.target_dir.resolve()
    existing_videos = [
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if target.exists() and existing_videos:
        n = len(existing_videos)
        print(f"[OK] {target} already exists with {n} videos. Skipping download.")
    else:
        try:
            download_dataset(args.dataset_slug, target)
        except Exception as e:
            print(f"[FAIL] Download failed: {e}")
            print(f"  Tip: check that you have Kaggle credentials in ~/.kaggle/kaggle.json")
            return 1

    # Limit to N videos
    if not args.keep_all and args.max_videos > 0:
        kept = limit_videos(target, args.max_videos)
    else:
        kept = sum(
            1 for p in target.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )

    # Summary
    videos = [
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    total_size = sum(p.stat().st_size for p in videos) / 1024 / 1024
    classes = sorted({p.parent.name for p in videos})
    print(f"\n[OK] Dataset ready at {target}")
    print(f"  Videos: {kept}")
    print(f"  Total size: {total_size:.1f} MB")
    print(f"  Classes: {len(classes)} ({', '.join(classes[:5])}{'...' if len(classes) > 5 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
