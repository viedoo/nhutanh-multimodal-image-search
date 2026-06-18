"""
Unified orchestrator script - Complete pipeline in one command

Usage:
    # Upload dataset + run pipeline + download + start server
    python run.py --type siglip2 --upload-dataset dataset
    
    # Just upload dataset
    python run.py --upload-dataset dataset --upload-only
    
    # Run pipeline without upload
    python run.py --type siglip2 --dataset-slug username/my-dataset
    
    # Run server with existing embeddings
    python run.py --server-only
"""
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import argparse
import subprocess


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"$ {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, shell=False)
    if result.returncode != 0:
        print(f"\n[FAIL] Failed: {description}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Unified Kaggle Pipeline Orchestrator")
    
    # Action modes
    parser.add_argument("--upload-only", action="store_true",
                       help="Only upload dataset, don't run pipeline")
    parser.add_argument("--server-only", action="store_true",
                       help="Only start Flask server with existing embeddings")
    
    # Dataset options
    parser.add_argument("--upload-dataset", default=None,
                       help="Local dataset folder to upload (e.g., 'dataset')")
    parser.add_argument("--dataset-slug", default=None,
                       help="Use existing Kaggle dataset (e.g., 'username/my-dataset')")
    parser.add_argument("--dataset-public", action="store_true",
                       help="Make uploaded dataset public")
    
    # Pipeline options
    parser.add_argument("--type", choices=["siglip2", "dinov3", "dinov3_dense"],
                       help="Model type to run")
    parser.add_argument("--skip-download", action="store_true",
                       help="Don't auto-download outputs")
    
    # Server options
    parser.add_argument("--skip-server", action="store_true",
                       help="Don't start server after pipeline")
    parser.add_argument("--port", type=int, default=5000,
                       help="Server port (default: 5000)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.server_only:
        # Just start server
        cmd = [sys.executable, "app.py"]
        if args.port and args.port != 5000:
            # app.py currently reads port from a constant; forward via env var
            # so that the same script stays the single source of truth.
            import os
            env = os.environ.copy()
            env["APP_PORT"] = str(args.port)
            print(f"\n{'='*60}")
            print(f"Starting Flask Server on port {args.port}")
            print(f"{'='*60}\n")
            result = subprocess.run(cmd, shell=False, env=env)
            sys.exit(result.returncode)
        sys.exit(0 if run_command(cmd, "Starting Flask Server") else 1)
    
    if args.upload_only:
        if not args.upload_dataset:
            print("[FAIL] --upload-dataset required with --upload-only")
            sys.exit(1)
        
        cmd = [sys.executable, "kaggle_dataset.py", "--folder", args.upload_dataset]
        if args.dataset_slug:
            cmd.extend(["--slug", args.dataset_slug])
        if args.dataset_public:
            cmd.append("--public")
        
        sys.exit(0 if run_command(cmd, "Uploading Dataset") else 1)
    
    if not args.type:
        print("[FAIL] --type required (unless using --upload-only or --server-only)")
        sys.exit(1)
    
    if not args.upload_dataset and not args.dataset_slug:
        print("[FAIL] Either --upload-dataset or --dataset-slug required")
        sys.exit(1)
    
    # Step 1: Build pipeline command
    cmd = [sys.executable, "kaggle_pipeline.py", "--type", args.type]
    
    if args.upload_dataset:
        cmd.extend(["--upload-dataset", args.upload_dataset])
        if args.dataset_slug:
            cmd.extend(["--dataset-slug", args.dataset_slug])
        if args.dataset_public:
            cmd.append("--dataset-public")
    elif args.dataset_slug:
        cmd.extend(["--dataset-slug", args.dataset_slug])
    
    if not args.skip_download:
        cmd.append("--auto-download")
    
    # Step 2: Run pipeline
    if not run_command(cmd, "Running Kaggle Pipeline"):
        sys.exit(1)
        
    # Step 3: Run Qdrant Ingestion (if we downloaded new embeddings)
    if not args.skip_download:
        ingest_cmd = [sys.executable, "qdrant_ingest.py"]
        if not run_command(ingest_cmd, "Ingesting embeddings into Qdrant Local"):
            print("[WARN] Qdrant ingestion failed or Qdrant is not running. Server may fail to start.")
    
    # Step 4: Start server if requested
    if not args.skip_server:
        print(f"\n{'='*60}")
        print("Pipeline complete! Starting Flask server...")
        print(f"{'='*60}\n")
        print("Server will be available at: http://localhost:5000")
        print("Press Ctrl+C to stop\n")
        
        server_cmd = [sys.executable, "app.py"]
        subprocess.run(server_cmd)
    else:
        print(f"\n{'='*60}")
        print("[OK] Pipeline complete!")
        print(f"{'='*60}\n")
        print("To start server:")
        print("  python app.py")


if __name__ == "__main__":
    main()
