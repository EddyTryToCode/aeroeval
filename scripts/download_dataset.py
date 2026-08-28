import argparse
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VISDRONE_URLS = {
    "VisDrone2019-DET-train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "VisDrone2019-DET-val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "VisDrone2019-DET-test-dev": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
}

def download_file(url: str, output_path: Path) -> None:
    print(f"Downloading from {url} to {output_path}...")
    import requests

    initial_bytes = output_path.stat().st_size if output_path.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if initial_bytes > 0:
        headers["Range"] = f"bytes={initial_bytes}-"
        print(f"Resuming download from byte {initial_bytes} ({initial_bytes / (1024*1024):.1f} MB)...")

    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)

    if response.status_code == 416:  # Range Not Satisfiable -> already fully downloaded
        print("File already completely downloaded.")
        return

    response.raise_for_status()

    content_range = response.headers.get("Content-Range")
    if content_range:
        total_size = int(content_range.split("/")[-1])
    else:
        total_size = int(response.headers.get('content-length', 0)) + initial_bytes
        initial_bytes = 0  # Server ignored range, overwrite

    mode = 'ab' if initial_bytes > 0 else 'wb'
    downloaded = initial_bytes

    with open(output_path, mode) as f:
        for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.2f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", end="", flush=True)
    print("\nDownload complete.")

def extract_archive(zip_path: Path, target_dir: Path) -> None:
    print(f"Extracting {zip_path} to {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction complete.")

def verify_dataset_structure(data_dir: Path) -> bool:
    splits = ["VisDrone2019-DET-train", "VisDrone2019-DET-val"]
    all_ok = True
    print("\nVerifying dataset structure:")
    for split in splits:
        split_dir = data_dir / split
        img_dir = split_dir / "images"
        ann_dir = split_dir / "annotations"

        if not split_dir.exists():
            print(f"[-] Missing directory: {split_dir}")
            all_ok = False
            continue

        img_count = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        ann_count = len(list(ann_dir.glob("*.txt"))) if ann_dir.exists() else 0

        print(f"[+] {split}:")
        print(f"    - Images: {img_count}")
        print(f"    - Annotations: {ann_count}")

        if img_count == 0 or ann_count == 0:
            all_ok = False
            print("    [!] Warning: Missing images or annotations")

    return all_ok

def main():
    parser = argparse.ArgumentParser(description="Download and prepare VisDrone dataset splits.")
    parser.add_argument("--subset", type=str, default="DET", choices=["DET", "VID", "MOT"], help="Subset type")
    parser.add_argument("--extract-only", action="store_true", help="Skip downloading, extract existing zips in data/")
    parser.add_argument("--include-test", action="store_true", help="Include test-dev split")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    targets = ["VisDrone2019-DET-train", "VisDrone2019-DET-val"]
    if args.include_test:
        targets.append("VisDrone2019-DET-test-dev")

    if not args.extract_only:
        for target in targets:
            target_folder = DATA_DIR / target
            if target_folder.exists():
                print(f"Target folder {target_folder} already exists. Skipping download.")
                continue

            zip_file = DATA_DIR / f"{target}.zip"
            url = VISDRONE_URLS.get(target)
            if url:
                try:
                    download_file(url, zip_file)
                except Exception as e:
                    print(f"Failed to auto-download {target}: {e}")
                    print(f"Please manually download {target}.zip to {DATA_DIR} and run with --extract-only")
            else:
                print(f"No direct URL configured for {target}")

    for zip_file in DATA_DIR.glob("*.zip"):
        extract_archive(zip_file, DATA_DIR)

    verify_dataset_structure(DATA_DIR)

if __name__ == "__main__":
    main()
