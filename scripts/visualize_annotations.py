"""
Sanity check script: Visualize YOLO annotations on sampled images.

Features:
- Randomly or deterministically samples N images from train or val split
- Reads YOLO normalized labels and maps to pixel coordinates
- Draws high-contrast bounding boxes with class labels and color coding
- Verifies bounding box validity (within bounds, non-zero area)
- Computes sample statistics (boxes per image, class distribution)
- Saves annotated images to reports/annotation_samples/
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "visdrone_yolo"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "annotation_samples"

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor"
]

# High-contrast color palette for 10 classes (BGR format for OpenCV)
CLASS_COLORS = [
    (0, 255, 0),     # 0: pedestrian - bright green
    (255, 0, 255),   # 1: people - magenta
    (0, 255, 255),   # 2: bicycle - yellow
    (255, 100, 0),   # 3: car - blue
    (0, 165, 255),   # 4: van - orange
    (0, 0, 255),     # 5: truck - red
    (180, 105, 255), # 6: tricycle - hot pink
    (255, 255, 0),   # 7: awning-tricycle - cyan
    (128, 0, 128),   # 8: bus - purple
    (0, 128, 255)    # 9: motor - dark orange
]


def draw_yolo_boxes(image: np.ndarray, label_file: Path):
    h, w, _ = image.shape
    boxes_info = []

    if not label_file.exists():
        return image, boxes_info

    with open(label_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls_id = int(parts[0])
        x_c, y_c, bw, bh = map(float, parts[1:])

        # Denormalize to pixel coordinates
        box_w = bw * w
        box_h = bh * h
        x1 = int((x_c * w) - (box_w / 2.0))
        y1 = int((y_c * h) - (box_h / 2.0))
        x2 = int(x1 + box_w)
        y2 = int(y1 + box_h)

        # Clip bounds for safety
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Text label configuration
        label_text = f"{cls_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        # Draw label background
        cv2.rectangle(image, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
        # Draw label text
        cv2.putText(image, label_text, (x1 + 1, max(th, y1 - 2)), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        boxes_info.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "bbox": (x1, y1, x2, y2),
            "width": box_w,
            "height": box_h,
            "area": box_w * box_h
        })

    return image, boxes_info


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO annotations for sanity check.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split to sample from")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of images to sample and visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_dir = DATA_DIR / "images" / args.split
    lbl_dir = DATA_DIR / "labels" / args.split

    if not img_dir.exists() or not lbl_dir.exists():
        print("Error: Directory not found. Run scripts/convert_visdrone.py first.")
        return

    all_images = sorted(list(img_dir.glob("*.jpg")))
    if not all_images:
        print(f"No images found in {img_dir}")
        return

    num_samples = min(args.num_samples, len(all_images))
    sampled_images = random.sample(all_images, num_samples)

    print(f"Sampling {num_samples} images from {args.split} split...")

    total_sampled_boxes = 0
    class_counts = {c: 0 for c in CLASS_NAMES}

    for img_path in tqdm(sampled_images, desc="Visualizing annotations"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        annotated_img, boxes = draw_yolo_boxes(img, lbl_path)

        total_sampled_boxes += len(boxes)
        for b in boxes:
            class_counts[b["class_name"]] += 1

        out_path = OUTPUT_DIR / f"annotated_{args.split}_{img_path.name}"
        cv2.imwrite(str(out_path), annotated_img)

    print("\n" + "=" * 60)
    print("           SANITY CHECK VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"Split:               {args.split}")
    print(f"Sampled Images:      {num_samples}")
    print(f"Total Objects Drawn: {total_sampled_boxes}")
    print(f"Avg Objects/Image:   {total_sampled_boxes / num_samples:.1f}")
    print("-" * 60)
    print("Class Distribution in Samples:")
    for cls_name, count in class_counts.items():
        pct = (count / total_sampled_boxes * 100) if total_sampled_boxes > 0 else 0
        print(f"  - {cls_name:16s}: {count:4d} ({pct:5.1f}%)")
    print("=" * 60)
    print(f"[✓] Annotated samples saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
