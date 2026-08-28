"""
Convert VisDrone2019-DET dataset annotations to YOLO format.

VisDrone annotation format:
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

Categories in VisDrone:
0: ignored regions (skipped)
1: pedestrian -> YOLO class 0
2: people -> YOLO class 1
3: bicycle -> YOLO class 2
4: car -> YOLO class 3
5: van -> YOLO class 4
6: truck -> YOLO class 5
7: tricycle -> YOLO class 6
8: awning-tricycle -> YOLO class 7
9: bus -> YOLO class 8
10: motor -> YOLO class 9
11: others (skipped)

YOLO annotation format:
<class_id> <x_center> <y_center> <width> <height>  (all normalized to [0, 1])

Output Directory Structure:
data/visdrone_yolo/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
"""

import os
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "visdrone_yolo"

VISDRONE_TO_YOLO_CLASS = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9   # motor
}


def convert_split(split_name: str, src_folder_name: str, use_symlinks: bool = True):
    src_dir = DATA_DIR / src_folder_name
    src_img_dir = src_dir / "images"
    src_ann_dir = src_dir / "annotations"

    dst_img_dir = OUTPUT_DIR / "images" / split_name
    dst_lbl_dir = OUTPUT_DIR / "labels" / split_name

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(src_img_dir.glob("*.jpg")))
    print(f"\nProcessing {split_name} ({len(img_files)} images)...")

    total_boxes = 0
    valid_boxes = 0
    skipped_ignored = 0
    skipped_invalid_geom = 0

    for img_path in tqdm(img_files, desc=f"Converting {split_name}"):
        dst_img_path = dst_img_dir / img_path.name
        if not dst_img_path.exists():
            if use_symlinks:
                try:
                    os.symlink(img_path.resolve(), dst_img_path)
                except OSError:
                    shutil.copy2(img_path, dst_img_path)
            else:
                shutil.copy2(img_path, dst_img_path)

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        ann_path = src_ann_dir / f"{img_path.stem}.txt"
        dst_lbl_path = dst_lbl_dir / f"{img_path.stem}.txt"

        yolo_lines = []

        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]

            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue

                total_boxes += 1
                try:
                    x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    cat_id = int(parts[5])
                except ValueError:
                    skipped_invalid_geom += 1
                    continue

                if cat_id not in VISDRONE_TO_YOLO_CLASS:
                    skipped_ignored += 1
                    continue

                if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
                    skipped_invalid_geom += 1
                    continue

                # Compute normalized coordinates
                x_center = (x + w / 2.0) / img_w
                y_center = (y + h / 2.0) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                # Clamp to [0, 1] range to avoid floating errors
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))

                yolo_cls = VISDRONE_TO_YOLO_CLASS[cat_id]
                yolo_lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                valid_boxes += 1

        with open(dst_lbl_path, "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

    print(f"[+] Finished {split_name}:")
    print(f"    - Images linked/copied: {len(img_files)}")
    print(f"    - Total raw boxes: {total_boxes}")
    print(f"    - Valid YOLO boxes converted: {valid_boxes}")
    print(f"    - Skipped (ignored/others classes): {skipped_ignored}")
    print(f"    - Skipped (invalid geometry/zero area): {skipped_invalid_geom}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", "VisDrone2019-DET-train"),
        ("val", "VisDrone2019-DET-val")
    ]

    for split_name, src_folder in splits:
        src_path = DATA_DIR / src_folder
        if not src_path.exists():
            print(f"Error: {src_path} not found!")
            return
        convert_split(split_name, src_folder, use_symlinks=True)

    print(f"\n[✓] All conversions complete. Output at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
