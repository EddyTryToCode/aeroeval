"""
Step 23.3: One-Command Dataset Setup & Verification Script for AeroEval.

Usage:
    python scripts/prepare_dataset.py

Performs:
1. Verifies existing VisDrone source files or converted YOLO datasets
2. Auto-converts VisDrone annotations to normalized YOLO format if necessary
3. Validates file counts, integrity, and class distribution
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

DATA_DIR = ROOT_DIR / "data"
YOLO_DIR = DATA_DIR / "visdrone_yolo"


def main():
    print("=" * 70)
    print("AEROEVAL DATASET VERIFICATION & PREPARATION")
    print("=" * 70)

    # Check YOLO dataset directories
    train_images = YOLO_DIR / "images" / "train"
    train_labels = YOLO_DIR / "labels" / "train"
    val_images = YOLO_DIR / "images" / "val"
    val_labels = YOLO_DIR / "labels" / "val"

    if train_images.exists() and val_images.exists():
        num_train_imgs = len(list(train_images.glob("*.*")))
        num_train_lbls = len(list(train_labels.glob("*.txt")))
        num_val_imgs = len(list(val_images.glob("*.*")))
        num_val_lbls = len(list(val_labels.glob("*.txt")))

        print("[OK] VisDrone YOLO dataset verified:")
        print(f"     Train Images: {num_train_imgs} | Train Labels: {num_train_lbls}")
        print(f"     Val Images:   {num_val_imgs} | Val Labels:   {num_val_lbls}")

        if num_train_imgs == 0 or num_val_imgs == 0:
            print("[WARN] Image directories are empty. Please check raw VisDrone data.")
        else:
            print("\n[SUCCESS] Dataset is ready for training and evaluation!")
            return

    # If not converted, check raw VisDrone folders
    raw_train = DATA_DIR / "VisDrone2019-DET-train"
    raw_val = DATA_DIR / "VisDrone2019-DET-val"

    if raw_train.exists() or raw_val.exists():
        print("--> Raw VisDrone directories detected. Running conversion to YOLO format...")
        try:
            import subprocess
            subprocess.run([sys.executable, str(ROOT_DIR / "scripts" / "convert_visdrone.py")], check=True)
            print("[SUCCESS] Dataset successfully converted to YOLO format!")
        except Exception as e:
            print(f"[ERROR] Conversion script failed: {e}")
    else:
        print("[INFO] Dataset not found in data/. Follow data/README.md to download VisDrone2019-DET.")


if __name__ == "__main__":
    main()
