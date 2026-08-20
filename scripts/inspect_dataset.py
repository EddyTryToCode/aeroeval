"""
Comprehensive dataset inspection for VisDrone-DET (Train & Val).

Calculates:
- Image count and resolution statistics
- Object counts per class and class distribution
- Bounding-box dimensions (width, height, aspect ratio, area)
- Size categorization (COCO standard: Small < 32^2, Medium 32^2-96^2, Large > 96^2)
- Scene density (objects per image)
- Truncation and occlusion statistics

Generates:
- reports/dataset_statistics.csv
- reports/class_distribution.png
- reports/object_size_distribution.png
- reports/objects_per_image.png
- reports/bbox_dimensions.png
- reports/image_resolution_distribution.png
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

CATEGORY_MAP = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others"
}

VALID_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def categorize_size(area: float) -> str:
    if area < 32 ** 2:  # < 1024 px^2
        return "small"
    elif area <= 96 ** 2:  # 1024 - 9216 px^2
        return "medium"
    else:  # > 9216 px^2
        return "large"


def parse_visdrone_split(split_name: str, split_dir: Path):
    img_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"

    img_files = sorted(list(img_dir.glob("*.jpg")))
    
    objects_list = []
    images_list = []

    print(f"Inspecting {split_name} ({len(img_files)} images)...")

    for img_path in tqdm(img_files, desc=f"Processing {split_name}"):
        ann_path = ann_dir / f"{img_path.stem}.txt"
        
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        img_obj_count = 0
        img_valid_obj_count = 0
        img_small_count = 0
        img_medium_count = 0
        img_large_count = 0

        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue

                try:
                    x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    score = float(parts[4])
                    cat_id = int(parts[5])
                    truncation = int(parts[6])
                    occlusion = int(parts[7])
                except ValueError:
                    continue

                img_obj_count += 1
                cat_name = CATEGORY_MAP.get(cat_id, "unknown")
                area = max(0.0, w * h)
                size_cat = categorize_size(area)
                aspect_ratio = (w / h) if h > 0 else 0.0

                is_valid = cat_id in VALID_CLASSES and w > 0 and h > 0
                if is_valid:
                    img_valid_obj_count += 1
                    if size_cat == "small":
                        img_small_count += 1
                    elif size_cat == "medium":
                        img_medium_count += 1
                    else:
                        img_large_count += 1

                objects_list.append({
                    "split": split_name,
                    "image_id": img_path.stem,
                    "img_width": img_w,
                    "img_height": img_h,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "bbox_area": area,
                    "aspect_ratio": aspect_ratio,
                    "score": score,
                    "category_id": cat_id,
                    "category_name": cat_name,
                    "is_valid": is_valid,
                    "truncation": truncation,
                    "occlusion": occlusion,
                    "size_category": size_cat
                })

        images_list.append({
            "split": split_name,
            "image_id": img_path.stem,
            "width": img_w,
            "height": img_h,
            "resolution": f"{img_w}x{img_h}",
            "total_objects": img_obj_count,
            "valid_objects": img_valid_obj_count,
            "small_objects": img_small_count,
            "medium_objects": img_medium_count,
            "large_objects": img_large_count
        })

    return pd.DataFrame(objects_list), pd.DataFrame(images_list)


def generate_plots(df_objects: pd.DataFrame, df_images: pd.DataFrame, reports_dir: Path):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    df_valid = df_objects[df_objects["is_valid"]].copy()

    # 1. Class Distribution
    plt.figure(figsize=(12, 6))
    class_order = [CATEGORY_MAP[i] for i in VALID_CLASSES]
    ax = sns.countplot(
        data=df_valid,
        x="category_name",
        hue="split",
        order=class_order,
        palette="viridis"
    )
    plt.title("Object Class Distribution (Train vs Val)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Object Count", fontsize=12)
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Split")
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=45, xytext=(0, 3),
                        textcoords='offset points')
    plt.tight_layout()
    plt.savefig(reports_dir / "class_distribution.png", dpi=300)
    plt.close()

    # 2. Object Size Distribution (COCO Definition)
    plt.figure(figsize=(10, 6))
    size_order = ["small", "medium", "large"]
    ax = sns.countplot(
        data=df_valid,
        x="size_category",
        hue="split",
        order=size_order,
        palette="crest"
    )
    plt.title("Bounding Box Size Categorization (COCO Standard)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Size Category (Small: <32², Medium: 32²-96², Large: >96²)", fontsize=12)
    plt.ylabel("Object Count", fontsize=12)
    plt.legend(title="Split")
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            total_split = len(df_valid[df_valid["split"] == p.get_label()]) if p.get_label() in df_valid["split"].unique() else len(df_valid)
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 5),
                        textcoords='offset points')
    plt.tight_layout()
    plt.savefig(reports_dir / "object_size_distribution.png", dpi=300)
    plt.close()

    # 3. Density: Objects Per Image
    plt.figure(figsize=(11, 5))
    sns.histplot(
        data=df_images,
        x="valid_objects",
        hue="split",
        kde=True,
        bins=50,
        palette="magma",
        element="step"
    )
    plt.title("Density Distribution: Valid Objects per Image", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Objects in Single Image", fontsize=12)
    plt.ylabel("Image Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(reports_dir / "objects_per_image.png", dpi=300)
    plt.close()

    # 4. Bounding Box Dimensions (Width vs Height)
    plt.figure(figsize=(9, 8))
    sample_df = df_valid.sample(n=min(15000, len(df_valid)), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x="bbox_w",
        y="bbox_h",
        hue="size_category",
        alpha=0.4,
        s=15,
        palette="tab10"
    )
    plt.title("Bounding Box Dimensions (Width vs Height, 15k Sample)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("BBox Width (px)", fontsize=12)
    plt.ylabel("BBox Height (px)", fontsize=12)
    plt.xlim(0, max(250, sample_df["bbox_w"].quantile(0.995)))
    plt.ylim(0, max(250, sample_df["bbox_h"].quantile(0.995)))
    plt.tight_layout()
    plt.savefig(reports_dir / "bbox_dimensions.png", dpi=300)
    plt.close()

    # 5. Image Resolution Distribution
    plt.figure(figsize=(12, 6))
    top_resolutions = df_images["resolution"].value_counts().head(10).index
    df_top_res = df_images[df_images["resolution"].isin(top_resolutions)]
    sns.countplot(
        data=df_top_res,
        x="resolution",
        hue="split",
        order=top_resolutions,
        palette="Set2"
    )
    plt.title("Top Image Resolutions in VisDrone-DET", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Resolution (Width x Height)", fontsize=12)
    plt.ylabel("Image Count", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(reports_dir / "image_resolution_distribution.png", dpi=300)
    plt.close()

    print(f"[+] All 5 visualizations saved in {reports_dir}")


def compute_and_save_summary(df_objects: pd.DataFrame, df_images: pd.DataFrame, reports_dir: Path):
    df_valid = df_objects[df_objects["is_valid"]].copy()

    stats_list = []

    for split in ["Train", "Val", "Combined"]:
        if split == "Combined":
            sub_objs = df_valid
            sub_imgs = df_images
        else:
            sub_objs = df_valid[df_valid["split"] == split]
            sub_imgs = df_images[df_images["split"] == split]

        total_images = len(sub_imgs)
        total_valid_objs = len(sub_objs)
        avg_objs_per_img = sub_imgs["valid_objects"].mean() if total_images > 0 else 0
        median_objs_per_img = sub_imgs["valid_objects"].median() if total_images > 0 else 0
        max_objs_in_img = sub_imgs["valid_objects"].max() if total_images > 0 else 0
        min_objs_in_img = sub_imgs["valid_objects"].min() if total_images > 0 else 0

        # Size ratios
        small_objs = (sub_objs["size_category"] == "small").sum()
        med_objs = (sub_objs["size_category"] == "medium").sum()
        large_objs = (sub_objs["size_category"] == "large").sum()

        small_pct = (small_objs / total_valid_objs * 100) if total_valid_objs > 0 else 0
        med_pct = (med_objs / total_valid_objs * 100) if total_valid_objs > 0 else 0
        large_pct = (large_objs / total_valid_objs * 100) if total_valid_objs > 0 else 0

        # Bbox geometry
        mean_area = sub_objs["bbox_area"].mean() if total_valid_objs > 0 else 0
        median_area = sub_objs["bbox_area"].median() if total_valid_objs > 0 else 0
        mean_w = sub_objs["bbox_w"].mean() if total_valid_objs > 0 else 0
        mean_h = sub_objs["bbox_h"].mean() if total_valid_objs > 0 else 0

        stats_list.append({
            "Split": split,
            "Total_Images": total_images,
            "Total_Valid_Objects": total_valid_objs,
            "Avg_Objects_Per_Image": round(avg_objs_per_img, 2),
            "Median_Objects_Per_Image": median_objs_per_img,
            "Min_Objects_In_Image": min_objs_in_img,
            "Max_Objects_In_Image": max_objs_in_img,
            "Small_Objects_Count": small_objs,
            "Small_Objects_Pct": round(small_pct, 2),
            "Medium_Objects_Count": med_objs,
            "Medium_Objects_Pct": round(med_pct, 2),
            "Large_Objects_Count": large_objs,
            "Large_Objects_Pct": round(large_pct, 2),
            "Mean_BBox_Area_px2": round(mean_area, 2),
            "Median_BBox_Area_px2": round(median_area, 2),
            "Mean_BBox_Width_px": round(mean_w, 2),
            "Mean_BBox_Height_px": round(mean_h, 2)
        })

    summary_df = pd.DataFrame(stats_list)
    summary_df.to_csv(reports_dir / "dataset_statistics.csv", index=False)
    print(f"[+] Dataset summary saved to {reports_dir / 'dataset_statistics.csv'}")

    # Print nicely to terminal
    print("\n" + "=" * 80)
    print("                      VISDRONE-DET DATASET SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("                     PER-CLASS OBJECT COUNTS")
    print("=" * 80)
    class_summary = pd.crosstab(
        df_valid["category_name"],
        df_valid["split"],
        margins=True,
        margins_name="Total"
    )
    class_summary["% of Total"] = (class_summary["Total"] / len(df_valid) * 100).round(2)
    print(class_summary.to_string())
    print("=" * 80 + "\n")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_dir = DATA_DIR / "VisDrone2019-DET-train"
    val_dir = DATA_DIR / "VisDrone2019-DET-val"

    if not train_dir.exists() or not val_dir.exists():
        print(f"Error: Missing dataset folders in {DATA_DIR}")
        return

    train_objs, train_imgs = parse_visdrone_split("Train", train_dir)
    val_objs, val_imgs = parse_visdrone_split("Val", val_dir)

    all_objs = pd.concat([train_objs, val_objs], ignore_index=True)
    all_imgs = pd.concat([train_imgs, val_imgs], ignore_index=True)

    print("\nGenerating statistical plots...")
    generate_plots(all_objs, all_imgs, REPORTS_DIR)

    print("Computing metrics summary...")
    compute_and_save_summary(all_objs, all_imgs, REPORTS_DIR)


if __name__ == "__main__":
    main()
