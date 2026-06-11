"""
Unified Kaggle Pipeline with proper version tracking
Solves the "not_found" and "stale status" issues

Usage: python kaggle_pipeline_unified.py --type siglip2 [--auto-download]
"""
import time
import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi

def push_notebook(api, notebook_type, dataset_slug, username):
    """Push notebook and return version info"""
    print(f"[PUSH] Pushing {notebook_type} notebook to Kaggle...")
    
    # Validate username
    if not username or username.strip() == "":
        raise ValueError(f"Kaggle username is empty or None. Check your credentials setup.")
    
    print(f"  Username: {username}")
    
    # Prepare kernel directory
    kernel_dir = Path(f"./temp_kernel_{notebook_type}")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Copy notebook – map logical type → actual filename
        _notebook_name_map = {
            "dinov3":       "dino-v3-embed.ipynb",
            "dinov3_dense": "dino-v3-dense-embed.ipynb",
            "siglip2":      "siglip2-embed.ipynb",
        }
        notebook_src = Path(f"./notebook/{_notebook_name_map.get(notebook_type, notebook_type + '-embed.ipynb')}") 
        if not notebook_src.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_src}")
        
        # Kaggle slug chỉ cho phép chữ thường, số và dấu '-' (không cho '_')
        slug_name = notebook_type.replace("_", "-")  # e.g. dinov3_dense → dinov3-dense
        code_filename = f"{slug_name}-embed.ipynb"
        shutil.copy(notebook_src, kernel_dir / code_filename)

        kernel_slug = f"{username}/{slug_name}-embed"

        # Create metadata (accelerator set in .ipynb file, not here)
        dataset_sources = [dataset_slug]
        # Add private secrets dataset for dinov3 / dinov3_dense (requires HF_TOKEN)
        if notebook_type in ("dinov3", "dinov3_dense"):
            dataset_sources.append(f"{username}/my-hf-secrets")

        metadata = {
            "id": kernel_slug,
            "title": f"{slug_name.upper()} Embed",
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
        
        # Push using Python API (not CLI) to capture response
        print(f"  Pushing to Kaggle API with T4 accelerator...")
        response = api.kernels_push(str(kernel_dir), acc="NvidiaTeslaT4")
        
        # Extract version info from response
        version_number = response.version_number
        kernel_ref = response.ref
        
        print(f"  ✓ Kernel pushed successfully!")
        print(f"    Ref: {kernel_ref}")
        print(f"    Version: {version_number}")
        print(f"    URL: https://www.kaggle.com/code/{kernel_slug}")
        
        # Save version info for tracking
        tracking_file = Path(f"./temp_kernel_{notebook_type}_version.json")
        tracking_data = {
            "kernel_slug": kernel_slug,
            "version_number": version_number,
            "ref": kernel_ref,
            "pushed_at": datetime.now().isoformat(),
            "notebook_type": notebook_type
        }
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        return tracking_data
        
    finally:
        # Cleanup temp directory
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir, ignore_errors=True)

def monitor_kernel(api, tracking_data, interval=30, max_wait=1200):
    """Monitor kernel execution with proper version tracking"""
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
            print(f"\n⚠ Timeout after {max_wait}s ({max_wait/60:.0f} minutes)")
            print(f"  Kernel may still be running. Check manually:")
            print(f"  https://www.kaggle.com/code/{kernel_slug}")
            print(f"\n  To download later:")
            print(f"  python kaggle_download.py --type {notebook_type}")
            return False
        
        try:
            # Check status using kernel slug (gets latest version)
            # Note: We can't specify version in status check, but after push
            # the latest version should be our newly pushed one
            status_response = api.kernels_status(kernel_slug)
            
            # Reset error counter on success
            consecutive_errors = 0
            
            # Extract status (handle both enum and string)
            if hasattr(status_response, 'status'):
                status_obj = status_response.status
                if hasattr(status_obj, 'name'):
                    current_status = status_obj.name.lower()
                else:
                    current_status = str(status_obj).lower()
            else:
                current_status = str(status_response.get('status', 'unknown')).lower()
            
            # Check for completion
            if current_status == 'complete':
                print(f"\n✓ Kernel completed!")
                print(f"  Total wait time: {elapsed:.0f}s ({elapsed/60:.1f} minutes)")
                return True
            
            # Check for errors
            if current_status in ['error', 'cancelacknowledged', 'cancelled', 'failed']:
                failure_msg = getattr(status_response, 'failure_message', None)
                print(f"\n✗ Kernel execution failed: {current_status}")
                if failure_msg:
                    print(f"  Failure message: {failure_msg}")
                print(f"  Check logs: https://www.kaggle.com/code/{kernel_slug}")
                return False
            
            # Print status updates
            if current_status != last_status or check_count % 3 == 0:
                status_display = current_status.replace('_', ' ').title()
                print(f"[{int(elapsed)}s] Status: {status_display}")
                last_status = current_status
            
            check_count += 1
            
        except Exception as e:
            consecutive_errors += 1
            error_msg = str(e)
            
            # Handle "not found" - this might mean kernel is still initializing
            if 'not' in error_msg.lower() and 'found' in error_msg.lower():
                if elapsed < 60:
                    print(f"[{int(elapsed)}s] Kernel initializing...")
                else:
                    print(f"[{int(elapsed)}s] Warning: Kernel not found. May need manual check.")
            else:
                print(f"[{int(elapsed)}s] API error: {error_msg}")
            
            # If too many consecutive errors, bail out
            if consecutive_errors >= 5:
                print(f"\n✗ Too many consecutive errors ({consecutive_errors}). Aborting.")
                print(f"  Check kernel manually: https://www.kaggle.com/code/{kernel_slug}")
                return False
        
        time.sleep(interval)

def download_outputs(api, tracking_data):
    """Download outputs from completed kernel"""
    kernel_slug = tracking_data['kernel_slug']
    notebook_type = tracking_data['notebook_type']

    temp_dir = Path(f"./temp_output_{notebook_type}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[DOWNLOAD] Downloading outputs from {kernel_slug}...")

    # Step 1: Download – UnicodeEncodeError on Windows is expected when Kaggle
    # writes the notebook log (stdout may contain UTF-8 chars like checkmarks).
    # Binary files (.hdf5 / .pkl) are downloaded BEFORE the log, so we catch
    # the encoding error and proceed with whatever landed in temp_dir.
    try:
        api.kernels_output_cli(kernel_slug, path=str(temp_dir))
    except UnicodeEncodeError:
        print("  (Log file has non-ASCII chars - skipping log, continuing with binary outputs)")
    except Exception as e:
        print(f"  Warning during download: {e}")

    # Step 2: Move downloaded binary files to result/ regardless of log errors
    try:
        result_dir = Path("./result")
        result_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        moved = 0

        for f in temp_dir.glob("*"):
            if f.is_file() and f.suffix != ".log":
                # Determine target filename
                if "embedding" in f.name.lower() and f.suffix in [".npz", ".hdf5"]:
                    new_name = f"{notebook_type}_embeddings_{timestamp}{f.suffix}"
                elif "faiss" in f.name.lower() and f.suffix in [".bin", ".pkl"]:
                    new_name = f"{notebook_type}_faiss_index_{timestamp}{f.suffix}"
                elif f.suffix in [".npz", ".hdf5", ".bin", ".pkl"]:
                    new_name = f"{notebook_type}_{f.stem}_{timestamp}{f.suffix}"
                else:
                    continue

                dest = result_dir / new_name
                shutil.move(str(f), str(dest))
                size_mb = dest.stat().st_size / 1024**2
                print(f"  [OK] {new_name} ({size_mb:.2f} MB)")
                moved += 1

        if moved == 0:
            print(f"  No output files found (only logs)")
            print(f"  Check: https://www.kaggle.com/code/{kernel_slug}")
            return False

        print(f"  [OK] Downloaded {moved} files to result/")
        return True

    except Exception as e:
        print(f"  Download error: {e}")
        return False
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(
        description="Unified Kaggle Pipeline with proper version tracking"
    )
    parser.add_argument("--type", required=True, choices=["siglip2", "dinov3", "dinov3_dense"],
                       help="Notebook type to run")
    parser.add_argument("--dataset", default="chetankv/dogs-cats-images",
                       help="Kaggle dataset slug")
    parser.add_argument("--auto-download", action="store_true",
                       help="Automatically download outputs after completion")
    parser.add_argument("--interval", type=int, default=30,
                       help="Status check interval in seconds (default: 30)")
    parser.add_argument("--max-wait", type=int, default=1200,
                       help="Maximum wait time in seconds (default: 1200 = 20min)")
    parser.add_argument("--push-only", action="store_true",
                       help="Only push notebook, don't monitor")
    
    args = parser.parse_args()
    
    # Initialize Kaggle API
    print("Initializing Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    username = api.get_config_value('username')
    
    print(f"Authenticated as: {username}\n")
    
    # Step 1: Push notebook
    try:
        tracking_data = push_notebook(api, args.type, args.dataset, username)
    except Exception as e:
        print(f"\n✗ Push failed: {e}")
        return 1
    
    # If push-only, exit here
    if args.push_only:
        print(f"\n✓ Push completed. Monitor manually or run:")
        print(f"  python kaggle_pipeline_unified.py --type {args.type} --monitor-only")
        return 0
    
    # Step 2: Monitor execution
    print(f"\nWaiting a few seconds before starting to monitor...")
    time.sleep(5)  # Give Kaggle time to initialize the kernel
    
    success = monitor_kernel(api, tracking_data, args.interval, args.max_wait)
    
    if not success:
        print(f"\n✗ Pipeline incomplete. To retry download later:")
        print(f"  python kaggle_download.py --type {args.type}")
        return 1
    
    # Step 3: Download outputs (if auto-download enabled)
    if args.auto_download:
        download_success = download_outputs(api, tracking_data)
        if download_success:
            print(f"\n✓ Pipeline completed successfully!")
            return 0
        else:
            print(f"\n⚠ Kernel completed but download failed.")
            print(f"  Retry: python kaggle_download.py --type {args.type}")
            return 1
    else:
        print(f"\n✓ Kernel completed! To download outputs:")
        print(f"  python kaggle_download.py --type {args.type}")
        return 0

if __name__ == "__main__":
    exit(main())
