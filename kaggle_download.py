"""
Enhanced download script compatible with unified pipeline
Supports both manual download and auto-retry with tracking
"""
import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi

def download_outputs(api, kernel_slug, notebook_type):
    """Download outputs from completed kernel"""
    temp_dir = Path(f"./temp_output_{notebook_type}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading outputs from {kernel_slug}...")
    
    try:
        # Use API to download outputs
        api.kernels_output_cli(kernel_slug, path=str(temp_dir))
        
        # Move files to result/
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
                print(f"  ✓ {new_name} ({size_mb:.2f} MB)")
                moved += 1
        
        if moved == 0:
            print(f"  ⚠ No output files found (only logs)")
            return False
        
        print(f"\n✓ Downloaded {moved} files to result/")
        return True
        
    except Exception as e:
        print(f"✗ Download error: {e}")
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Download Kaggle kernel outputs")
    parser.add_argument("--type", required=True, choices=["siglip2", "dinov3"],
                       help="Notebook type")
    parser.add_argument("--check-status", action="store_true",
                       help="Check kernel status before downloading")
    
    args = parser.parse_args()
    
    # Initialize API
    api = KaggleApi()
    api.authenticate()
    username = api.get_config_value('username')
    kernel_slug = f"{username}/{args.type}-embed"
    
    # Check tracking file if exists
    tracking_file = Path(f"./temp_kernel_{args.type}_version.json")
    if tracking_file.exists():
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
        print(f"Found tracking data:")
        print(f"  Version: {tracking_data.get('version_number')}")
        print(f"  Pushed at: {tracking_data.get('pushed_at')}")
    
    # Check status if requested
    if args.check_status:
        print(f"\nChecking kernel status...")
        try:
            status_response = api.kernels_status(kernel_slug)
            if hasattr(status_response, 'status'):
                status_obj = status_response.status
                if hasattr(status_obj, 'name'):
                    current_status = status_obj.name.lower()
                else:
                    current_status = str(status_obj).lower()
            else:
                current_status = 'unknown'
            
            print(f"  Status: {current_status}")
            
            if current_status != 'complete':
                print(f"\n⚠ Kernel is not complete yet ({current_status})")
                print(f"  Check: https://www.kaggle.com/code/{kernel_slug}")
                return 1
        except Exception as e:
            print(f"  Error checking status: {e}")
            print(f"  Attempting download anyway...")
    
    # Download outputs
    print(f"\nDownloading from: {kernel_slug}\n")
    success = download_outputs(api, kernel_slug, args.type)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
