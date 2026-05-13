import argparse
import logging
from pathlib import Path

from risklens.data.download import download_lending_club, verify_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Download Lending Club dataset from Kaggle")
    parser.add_argument("--force", action="store_true", help="Force re-download if file exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    
    file_path = download_lending_club(RAW_DATA_DIR, force=args.force)
    verify_download(file_path)
    
    size_gb = file_path.stat().st_size / (1024 * 1024 * 1024)
    print(f"\nSUCCESS: Data ready at {file_path} ({size_gb:.2f} GB)")

if __name__ == "__main__":
    main()
