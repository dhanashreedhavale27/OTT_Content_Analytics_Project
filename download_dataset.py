import os
import zipfile
from pathlib import Path

DATASET_SLUG = "shivamb/netflix-shows"
DOWNLOAD_DIR = Path("data")
DOWNLOAD_DIR.mkdir(exist_ok=True)

try:
    import kagglehub
except ImportError:
    raise SystemExit(
        "kagglehub is not installed. Run: pip install -r requirements.txt"
    )

print("Downloading Netflix Movies and TV Shows dataset from Kaggle...")

path = kagglehub.dataset_download(DATASET_SLUG)
print("Kaggle download location:", path)

csv_files = list(Path(path).rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file was found in the downloaded Kaggle dataset.")

# The official dataset contains netflix_titles.csv.
source = next(
    (p for p in csv_files if p.name.lower() == "netflix_titles.csv"),
    csv_files[0]
)

destination = DOWNLOAD_DIR / "netflix_titles.csv"
destination.write_bytes(source.read_bytes())

print(f"Dataset copied to: {destination}")
print("Now run: python train.py")
