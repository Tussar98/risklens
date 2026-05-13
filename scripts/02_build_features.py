"""CLI entry point for filtering raw Lending Club data to completed loans."""

import argparse
import logging
from pathlib import Path

from risklens.data.load import build_filtered_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter raw Lending Club data to completed loans"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per chunk when reading the raw CSV",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    raw_path = PROJECT_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
    output_path = PROJECT_ROOT / "data" / "interim" / "loans_filtered.parquet"

    result = build_filtered_dataset(raw_path, output_path, chunksize=args.chunksize)
    size_mb = result.stat().st_size / (1024 * 1024)
    print(f"\nSUCCESS: Filtered dataset at {result} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()