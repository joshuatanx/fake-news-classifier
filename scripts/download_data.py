from pathlib import Path
import shutil
import kagglehub

DATASET = "saurabhshahane/fake-news-classification"
TARGET_DIR = Path("data/external/welfake")

def main():
    TARGET_DIR.mkdir(parents = True, exist_ok = True)

    download_path = Path(kagglehub.dataset_download(DATASET))

    for file in download_path.iterdir():
        dest = TARGET_DIR / file.name
        if not dest.exists():
            shutil.copy(file, dest)
            print(f"Copied {file.name}")
        else:
            print(f"Skipped {file.name} (already exists)")

if __name__ == "__main__":
    main()