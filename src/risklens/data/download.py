import logging
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger(__name__)

def download_lending_club(
    raw_data_dir: Path,
    dataset_ref: str = "wordsforthewise/lending-club",
    file_name: str = "accepted_2007_to_2018Q4.csv.gz",
    force: bool = False,
) -> Path:
    """
    Downloads the accepted-loans file from Kaggle into raw_data_dir.
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    expected_path = raw_data_dir / file_name

    if expected_path.exists() and not force:
        logger.info("File already exists, skipping download")
        return expected_path

    logger.info(f"Downloading {file_name} from Kaggle dataset {dataset_ref}...")
    api = KaggleApi()
    api.authenticate()
    
    api.dataset_download_file(dataset_ref, file_name, path=str(raw_data_dir))
    
    zip_path = raw_data_dir / f"{file_name}.zip"
    if zip_path.exists() and not expected_path.exists():
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_data_dir)
        zip_path.unlink()
        
    if not expected_path.exists():
        raise FileNotFoundError(f"Failed to find expected file {expected_path} after download.")

    size_mb = expected_path.stat().st_size / (1024 * 1024)
    logger.info(f"Download complete. File size: {size_mb:.2f} MB")
    
    return expected_path


def verify_download(file_path: Path, min_size_mb: int = 300) -> None:
    """
    Raises ValueError if file_path doesn't exist or is smaller than min_size_mb MB.
    """
    if not file_path.exists():
        raise ValueError(f"File {file_path} does not exist.")
        
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb < min_size_mb:
        raise ValueError(f"File {file_path} is too small ({size_mb:.2f} MB < {min_size_mb} MB).")
        
    logger.info(f"Verification passed: {file_path} is {size_mb:.2f} MB.")
