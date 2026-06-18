"""
Upload a local dataset folder to Kaggle as a dataset.

Usage:
    python kaggle_dataset.py --folder dataset
    python kaggle_dataset.py --folder dataset --slug custom-name
    python kaggle_dataset.py --folder dataset --slug custom-name --public

This creates or updates a Kaggle dataset so notebooks can mount it via
dataset_sources in kernel-metadata.json.
"""
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import argparse
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from config import SUPPORTED_IMAGE_EXTENSIONS


def validate_dataset_folder(folder: Path) -> dict:
    """
    Validate dataset folder and return statistics.
    
    Returns:
        dict with keys: total_images, total_size_mb, subdirs, sample_files
    """
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")
    
    # Scan for images
    image_files = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_files.extend(folder.rglob(f"*{ext}"))
    
    if not image_files:
        raise ValueError(
            f"No images found in {folder}\n"
            f"Supported extensions: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    
    # Calculate statistics
    total_size = sum(f.stat().st_size for f in image_files)
    subdirs = set(f.parent.relative_to(folder) for f in image_files if f.parent != folder)
    
    # Get sample files for verification
    sample_files = [str(f.relative_to(folder)) for f in image_files[:5]]
    
    return {
        "total_images": len(image_files),
        "total_size_mb": total_size / (1024 ** 2),
        "subdirs": sorted([str(d) for d in subdirs]),
        "sample_files": sample_files,
    }


def upload_dataset(api, folder, slug=None, public=False, version_notes="Updated via API"):
    """Upload a local folder to Kaggle as a dataset.

    Returns the dataset slug on success.
    """
    folder = Path(folder).resolve()
    
    # Validate dataset
    print(f"[UPLOAD] Validating dataset folder: {folder}")
    stats = validate_dataset_folder(folder)
    
    print(f"  Found {stats['total_images']} images ({stats['total_size_mb']:.1f} MB)")
    if stats['subdirs']:
        print(f"  Subdirectories: {len(stats['subdirs'])} ({', '.join(stats['subdirs'][:3])}{'...' if len(stats['subdirs']) > 3 else ''})")
    print(f"  Sample files:")
    for sample in stats['sample_files']:
        print(f"    - {sample}")

    # Generate or validate slug
    username = api.get_config_value("username")
    if not slug:
        slug = f"{username}/{folder.name}"
    elif "/" not in slug:
        slug = f"{username}/{slug}"

    print(f"\n  Target slug: {slug}")
    print(f"  Visibility: {'Public' if public else 'Private'}")
    print(f"  Kaggle mount path: /kaggle/input/{slug.split('/')[-1]}")

    # Create dataset-metadata.json if missing
    meta_path = folder / "dataset-metadata.json"
    created_meta = False
    if not meta_path.exists():
        slug_parts = slug.split("/")
        metadata = {
            "title": slug_parts[-1].replace("-", " ").replace("_", " ").title(),
            "id": slug,
            "licenses": [{"name": "CC0-1.0"}],
            "isPrivate": not public,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\n  Created dataset-metadata.json")
        created_meta = True
    else:
        # Ensure the id matches our slug and update privacy setting
        with open(meta_path, "r") as f:
            existing = json.load(f)
        if existing.get("id") != slug:
            existing["id"] = slug
        existing["isPrivate"] = not public
        with open(meta_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\n  Updated dataset-metadata.json id to: {slug}")

    try:
        # Try updating existing dataset first
        print(f"\n  Uploading to Kaggle (tarball mode for nested folders)...")
        api.dataset_create_version(str(folder), version_notes, dir_mode='tar')
        print(f"  [OK] Dataset version created successfully!")
        print(f"    URL: https://www.kaggle.com/datasets/{slug}")
        return slug

    except Exception as e:
        error_msg = str(e).lower()

        if "not found" in error_msg or "404" in error_msg or "403" in error_msg or "forbidden" in error_msg:
            # Dataset doesn't exist yet, create new (403/404 both indicate dataset doesn't exist)
            print(f"  Dataset not found, creating new...")
            try:
                api.dataset_create_new(str(folder), dir_mode='tar')
                print(f"  [OK] Dataset created successfully!")
                print(f"    URL: https://www.kaggle.com/datasets/{slug}")
                return slug
            except Exception as e2:
                raise RuntimeError(f"Failed to create dataset: {e2}") from e2
        else:
            raise RuntimeError(f"Failed to update dataset: {e}") from e

    finally:
        # Clean up auto-generated metadata if we created it
        if created_meta and meta_path.exists():
            meta_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Upload a local dataset folder to Kaggle"
    )
    parser.add_argument(
        "--folder", required=True,
        help="Local folder path containing dataset files"
    )
    parser.add_argument(
        "--slug",
        help="Dataset slug (format: username/dataset-name). Auto-generated from folder name if not provided."
    )
    parser.add_argument(
        "--public", action="store_true",
        help="Make dataset public (default: private)"
    )
    parser.add_argument(
        "--version-notes", default="Updated via API",
        help="Version notes for the upload"
    )

    args = parser.parse_args()

    # Authenticate
    print("Initializing Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    username = api.get_config_value("username")
    print(f"Authenticated as: {username}\n")

    # Upload
    try:
        slug = upload_dataset(
            api, args.folder, args.slug, args.public, args.version_notes
        )
        print(f"\n[OK] Dataset uploaded successfully!")
        print(f"  Use this slug in kaggle_pipeline.py: --dataset {slug}")
        return 0
    except Exception as e:
        print(f"\n[FAIL] Upload failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
