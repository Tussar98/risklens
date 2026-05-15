"""CLI: build the LGD dataset from raw Lending Club data."""

import argparse
import logging
from pathlib import Path

from risklens.data.lgd_data import build_lgd_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LGD dataset from raw CSV")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    raw_path = PROJECT_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
    output_path = PROJECT_ROOT / "data" / "interim" / "lgd_data.parquet"

    result = build_lgd_dataset(raw_path, output_path, chunksize=args.chunksize)
    size_mb = result.stat().st_size / (1024 * 1024)
    print(f"\nSUCCESS: LGD dataset at {result} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
