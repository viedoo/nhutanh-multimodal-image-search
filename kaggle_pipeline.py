"""
Simplified Kaggle Pipeline - Auto-detects dataset structure
No more hardcoded paths or complex rewriting logic

Usage:
    python kaggle_pipeline.py --type siglip2 --upload-dataset dataset --auto-download
    python kaggle_pipeline.py --type dinov3 --upload-dataset dataset --auto-download
"""
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
from config import (
    DEFAULT_MAX_WAIT,
    DEFAULT_MONITOR_INTERVAL,
    KAGGLE_HF_SECRETS_SLUG,
    RESULT_DIR,
    get_model_config,
)


def push_notebook(api, notebook_type, dataset_slug, username):
    """Push notebook to Kaggle and return tracking info"""
    print(f"\n[PUSH] Pushing {notebook_type} notebook to Kaggle...")
    
    model_config = get_model_config(notebook_type)
    kernel_dir = Path(f"./temp_kernel_{notebook_type}")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Copy notebook
        notebook_src = Path(f"./notebook/{model_config['notebook_file']}")
        if not notebook_src.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_src}")
        
        code_filename = model_config['notebook_file']
        shutil.copy(notebook_src, kernel_dir / code_filename)
        
        kernel_slug = f"{username}/{model_config['kernel_slug_suffix']}"
        
        # Prepare dataset sources
        dataset_sources = [dataset_slug] if dataset_slug else []
        
        # Add HF secrets dataset for models that need it
        if model_config['requires_hf_token']:
            secrets_dataset = f"{username}/{KAGGLE_HF_SECRETS_SLUG}"
            dataset_sources.append(secrets_dataset)
            print(f"  Added secrets dataset: {secrets_dataset}")
        
        # Create metadata
        metadata = {
            "id": kernel_slug,
            "title": f"{notebook_type.upper()} Embed",
            "code_file": code_filename,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": dataset_sources,
            "competition_sources": [],
            "kernel_sources": []
        }
        
        with open(kernel_dir / "kernel-metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Pushing to Kaggle with T4 GPU...")
        print(f"  Dataset sources: {dataset_sources}")
        response = api.kernels_push(str(kernel_dir), acc="NvidiaTeslaT4")
        
        version_number = response.version_number
        kernel_ref = response.ref
        
        print(f"  [OK] Kernel pushed successfully!")
        print(f"    Ref: {kernel_ref}")
        print(f"    Version: {version_number}")
        print(f"    URL: https://www.kaggle.com/code/{kernel_slug}")
        
        # Save tracking info
        tracking_file = Path(f"./temp_kernel_{notebook_type}_version.json")
        tracking_data = {
            "kernel_slug": kernel_slug,
            "version_number": version_number,
            "ref": kernel_ref,
            "pushed_at": datetime.now().isoformat(),
            "notebook_type": notebook_type,
            "dataset_slug": dataset_slug
        }
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        return tracking_data
        
    finally:
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir, ignore_errors=True)


def fetch_kernel_log(api, kernel_slug, notebook_type, last_n_lines=60):
    """Download and display the tail of a kernel log so the user can see actual errors.

    Returns True if a log was found and printed, False otherwise.

    The kaggle==2.2.0 API has no `kernels_log` method, so we reuse
    `kernels_output` — it bundles the .log file alongside the output artifacts.
    """
    temp_dir = Path(f"./temp_log_{notebook_type}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[LOG] Fetching kernel log: {kernel_slug}")
    try:
        try:
            # Force a fresh download even if Kaggle has cached the version.
            api.kernels_output(kernel_slug, path=str(temp_dir), force=True, quiet=True)
        except UnicodeEncodeError:
            # Kaggle CLI tried to print non-ASCII to a cp1252 console; that's fine —
            # the log file itself has already been written to disk.
            print("  (Console encoding issue while downloading — log file may still be available.)")
        except Exception as e:
            print(f"  Could not fetch log: {e}")
            return False

        candidates = list(temp_dir.glob("*.log"))
        if not candidates:
            print(f"  No .log file in {temp_dir} (files: {[p.name for p in temp_dir.iterdir()]})")
            return False

        # Read with utf-8 + 'replace' so any odd byte never crashes the script.
        log_path = candidates[0]
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        print(f"  Log file: {log_path.name} ({len(lines)} lines, showing last {min(last_n_lines, len(lines))})")
        print("  " + "-" * 56)
        tail = lines[-last_n_lines:] if len(lines) > last_n_lines else lines
        for line in tail:
            print("  " + line.rstrip())
        print("  " + "-" * 56)
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def monitor_kernel(api, tracking_data, interval=30, max_wait=1200):
    """Monitor kernel execution with proper status tracking"""
    kernel_slug = tracking_data['kernel_slug']
    version_number = tracking_data['version_number']
    notebook_type = tracking_data['notebook_type']
    
    print(f"\n[MONITOR] Monitoring kernel: {kernel_slug}")
    print(f"  Version: {version_number}")
    print(f"  Check interval: {interval}s")
    print(f"  Max wait: {max_wait}s ({max_wait/60:.0f} minutes)")
    print(f"  URL: https://www.kaggle.com/code/{kernel_slug}\n")
    
    start_time = time.time()
    check_count = 0
    last_status = None
    consecutive_errors = 0
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > max_wait:
            print(f"\n[TIMEOUT] after {max_wait}s ({max_wait/60:.0f} minutes)")
            print(f"  Kernel may still be running. Check manually:")
            print(f"  https://www.kaggle.com/code/{kernel_slug}")
            print(f"\n  To download later:")
            print(f"  python kaggle_download.py --type {notebook_type}")
            return False
        
        try:
            status_response = api.kernels_status(kernel_slug)
            consecutive_errors = 0
            
            # Extract status
            if hasattr(status_response, 'status'):
                status_obj = status_response.status
                current_status = status_obj.name.lower() if hasattr(status_obj, 'name') else str(status_obj).lower()
            else:
                current_status = str(status_response.get('status', 'unknown')).lower()
            
            # Check completion
            if current_status == 'complete':
                print(f"\n[DONE] Kernel completed!")
                print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} minutes)")
                return True
            
            # Check failures
            if current_status in ['error', 'cancelacknowledged', 'cancelled', 'failed']:
                failure_msg = getattr(status_response, 'failure_message', None)
                print(f"\n[ERROR] Kernel failed: {current_status}")
                if failure_msg:
                    print(f"  Message: {failure_msg}")
                print(f"  Check logs: https://www.kaggle.com/code/{kernel_slug}")
                # Try to grab the actual log so the user can see the real stacktrace.
                fetch_kernel_log(api, kernel_slug, notebook_type)
                return False
            
            # Print status updates
            if current_status != last_status or check_count % 3 == 0:
                print(f"[{int(elapsed)}s] Status: {current_status.replace('_', ' ').title()}")
                last_status = current_status
            
            check_count += 1
            
        except Exception as e:
            consecutive_errors += 1
            error_msg = str(e)
            
            if 'not' in error_msg.lower() and 'found' in error_msg.lower():
                if elapsed < 60:
                    print(f"[{int(elapsed)}s] Kernel initializing...")
                else:
                    print(f"[{int(elapsed)}s] Warning: Kernel not found")
            else:
                print(f"[{int(elapsed)}s] API error: {error_msg}")
            
            if consecutive_errors >= 5:
                print(f"\n[FAIL] Too many errors ({consecutive_errors}). Check manually.")
                return False
        
        time.sleep(interval)


def download_outputs(api, tracking_data):
    """Download kernel outputs"""
    kernel_slug = tracking_data['kernel_slug']
    notebook_type = tracking_data['notebook_type']
    
    temp_dir = Path(f"./temp_output_{notebook_type}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[DOWNLOAD] Downloading from {kernel_slug}...")
    
    try:
        api.kernels_output_cli(kernel_slug, path=str(temp_dir))
    except UnicodeEncodeError:
        print("  (Log encoding issue - continuing with binary outputs)")
    except Exception as e:
        print(f"  Warning: {e}")
    
    try:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        moved = 0
        
        for f in temp_dir.glob("*"):
            if f.is_file() and f.suffix != ".log":
                if "embedding" in f.name.lower() and f.suffix in [".npz", ".hdf5"]:
                    new_name = f"{notebook_type}_embeddings_{timestamp}{f.suffix}"
                elif "faiss" in f.name.lower() and f.suffix in [".bin", ".pkl"]:
                    new_name = f"{notebook_type}_faiss_index_{timestamp}{f.suffix}"
                elif f.suffix in [".npz", ".hdf5", ".bin", ".pkl"]:
                    new_name = f"{notebook_type}_{f.stem}_{timestamp}{f.suffix}"
                else:
                    continue
                
                dest = RESULT_DIR / new_name
                shutil.move(str(f), str(dest))
                size_mb = dest.stat().st_size / 1024**2
                print(f"  [OK] {new_name} ({size_mb:.2f} MB)")
                moved += 1
        
        if moved == 0:
            print(f"  No output files found")
            # If the kernel "completed" but produced no output, the notebook
            # almost certainly errored out (e.g. missing HDF5 save cell, OOM).
            # Fetch the log to show the actual root cause.
            try:
                fetch_kernel_log(api, kernel_slug, notebook_type)
            except Exception as e:
                print(f"  (Could not fetch log: {e})")
            return False
        
        print(f"  [OK] Downloaded {moved} files")
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Kaggle Pipeline - Auto-detects dataset structure")
    parser.add_argument("--type", required=True, choices=["siglip2", "dinov3", "dinov3_dense", "qwen3vl"],
                       help="Model type to run")
    parser.add_argument("--upload-dataset", default=None,
                       help="Local dataset folder to upload (e.g., 'dataset')")
    parser.add_argument("--dataset-slug", default=None,
                       help="Override dataset slug (default: username/folder-name)")
    parser.add_argument("--dataset-public", action="store_true",
                       help="Make uploaded dataset public")
    parser.add_argument("--auto-download", action="store_true",
                       help="Auto-download outputs after completion")
    parser.add_argument("--interval", type=int, default=DEFAULT_MONITOR_INTERVAL,
                       help=f"Status check interval (default: {DEFAULT_MONITOR_INTERVAL}s)")
    parser.add_argument("--max-wait", type=int, default=DEFAULT_MAX_WAIT,
                       help=f"Max wait time (default: {DEFAULT_MAX_WAIT}s)")
    parser.add_argument("--push-only", action="store_true",
                       help="Only push notebook")
    
    args = parser.parse_args()
    
    # Initialize API
    print("Initializing Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    username = api.get_config_value('username')
    print(f"Authenticated as: {username}\n")
    
    # Step 0: Upload dataset if provided
    dataset_slug = args.dataset_slug
    if args.upload_dataset:
        print("=" * 60)
        print("STEP 0: Upload Dataset")
        print("=" * 60)
        from kaggle_dataset import upload_dataset
        try:
            dataset_slug = upload_dataset(
                api, args.upload_dataset,
                slug=args.dataset_slug,
                public=args.dataset_public
            )
            print(f"\n[OK] Dataset uploaded: {dataset_slug}\n")
        except Exception as e:
            print(f"\n[FAIL] Dataset upload failed: {e}")
            return 1
    
    if not dataset_slug:
        print("[FAIL] No dataset specified. Use --upload-dataset or --dataset-slug")
        return 1
    
    # Step 1: Push notebook
    print("=" * 60)
    print("STEP 1: Push Notebook")
    print("=" * 60)
    try:
        tracking_data = push_notebook(api, args.type, dataset_slug, username)
    except Exception as e:
        print(f"\n[FAIL] Push failed: {e}")
        return 1
    
    if args.push_only:
        print(f"\n[OK] Push complete")
        return 0
    
    # Step 2: Monitor
    print("\n" + "=" * 60)
    print("STEP 2: Monitor Execution")
    print("=" * 60)
    time.sleep(5)
    
    success = monitor_kernel(api, tracking_data, args.interval, args.max_wait)
    
    if not success:
        print(f"\n[FAIL] Pipeline incomplete")
        return 1
    
    # Step 3: Download
    if args.auto_download:
        print("\n" + "=" * 60)
        print("STEP 3: Download Outputs")
        print("=" * 60)
        download_success = download_outputs(api, tracking_data)
        if download_success:
            print(f"\n[OK] Pipeline completed!")
            return 0
        else:
            print(f"\n[WARN] Download failed")
            return 1
    else:
        print(f"\n[OK] Kernel complete. Download with:")
        print(f"  python kaggle_download.py --type {args.type}")
        return 0


if __name__ == "__main__":
    exit(main())
