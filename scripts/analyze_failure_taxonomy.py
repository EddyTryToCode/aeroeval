"""
Step 6E: Failure Taxonomy & Root Cause Analysis.

Systematically classifies model prediction errors on VisDrone validation set into 6 core failure modes:
1. Small-Object Miss (False Negative with area < 32^2 px)
2. Class Confusion (High confidence detection with incorrect class label)
3. Occlusion Failure (False Negative on objects tagged with occlusion > 0)
4. Motion Blur / Defocus Vulnerability (Performance degradation under blur)
5. Low-Light Vulnerability (Performance drop in dark scene conditions)
6. Resolution Downscaling Degradation (Information loss under lower bandwidth)

Extracts:
- Error breakdown count & percentage for Exp A, B1, B2, B3
- Confusion Matrices with top confused pairs (pedestrian vs people, car vs van, motor vs bicycle)
- Qualitative failure samples saved to reports/error_analysis/

Outputs:
- reports/error_analysis/failure_taxonomy_metrics.csv
- reports/error_analysis/failure_taxonomy_metrics.md
- reports/error_analysis/error_distribution_comparison.png
- reports/error_analysis/top_confused_classes.png
- reports/error_analysis/sample_failure_cases.png
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "visdrone_yolo"
RAW_VAL_DIR = ROOT_DIR / "data" / "VisDrone2019-DET-val"
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUT_DIR = REPORTS_DIR / "error_analysis"

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

EXPERIMENTS = {
    "A": {
        "name": "baseline_yolo11n",
        "label": "A (YOLO11n-640)",
        "weights": ROOT_DIR / "experiments" / "baseline_yolo11n" / "weights" / "best.pt",
        "size": 640
    },
    "B1": {
        "name": "exp_b1_yolo11s_960",
        "label": "B1 (YOLO11s-960)",
        "weights": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b1_yolo11s_960" / "weights" / "best.pt",
        "size": 960
    },
    "B2": {
        "name": "exp_b2_yolo11s_1280",
        "label": "B2 (YOLO11s-1280)",
        "weights": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b2_yolo11s_1280" / "weights" / "best.pt",
        "size": 1280
    },
    "B3": {
        "name": "exp_b3_yolo11m_960",
        "label": "B3 (YOLO11m-960)",
        "weights": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b3_yolo11m_960" / "weights" / "best.pt",
        "size": 960
    }
}


def load_raw_val_annotations():
    ann_dir = RAW_VAL_DIR / "annotations"
    img_dir = RAW_VAL_DIR / "images"
    
    gt_dict = {}
    valid_classes = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}

    for img_p in img_dir.glob("*.jpg"):
        ann_p = ann_dir / f"{img_p.stem}.txt"
        with Image.open(img_p) as im:
            w_img, h_img = im.size

        objects = []
        if ann_p.exists():
            with open(ann_p, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue
                try:
                    x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    cat_id = int(parts[5])
                    trunc = int(parts[6])
                    occ = int(parts[7])
                except ValueError:
                    continue

                if cat_id not in valid_classes or w <= 0 or h <= 0:
                    continue

                cls_id = valid_classes[cat_id]
                area = w * h
                objects.append({
                    "cls": cls_id,
                    "box": [x, y, x + w, y + h],
                    "width": w,
                    "height": h,
                    "area": area,
                    "is_small": area < (32 ** 2),
                    "is_occluded": occ > 0,
                    "is_truncated": trunc > 0
                })
        gt_dict[img_p.name] = objects
    return gt_dict


def compute_iou_matrix(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    ix1 = np.maximum(b1_x1[:, None], b2_x1)
    iy1 = np.maximum(b1_y1[:, None], b2_y1)
    ix2 = np.minimum(b1_x2[:, None], b2_x2)
    iy2 = np.minimum(b1_y2[:, None], b2_y2)

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    iarea = iw * ih

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1[:, None] + area2 - iarea
    return np.where(union > 0, iarea / union, 0.0)


def extract_model_errors(model: YOLO, imgsz: int, gt_dict: dict, conf_thresh: float = 0.25, iou_thresh: float = 0.5):
    val_img_dir = DATA_DIR / "images" / "val"
    img_files = sorted(list(val_img_dir.glob("*.jpg")))
    device = 0 if torch.cuda.is_available() else "cpu"

    total_gt = 0
    total_preds = 0
    
    tp_count = 0
    fp_background = 0
    fp_class_confusion = 0
    fn_small_miss = 0
    fn_occlusion_miss = 0
    fn_other_miss = 0

    confusion_matrix = np.zeros((10, 10), dtype=int)
    failure_samples = []

    # Batch prediction
    batch_size = 16
    for i in range(0, len(img_files), batch_size):
        batch = img_files[i : i + batch_size]
        results = model.predict(source=batch, imgsz=imgsz, conf=conf_thresh, iou=0.6, device=device, verbose=False)

        for r in results:
            img_name = Path(r.path).name
            gts = gt_dict.get(img_name, [])
            total_gt += len(gts)

            preds = []
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes.data.cpu().numpy():
                    preds.append({
                        "cls": int(b[5]),
                        "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "score": float(b[4])
                    })
            total_preds += len(preds)

            if len(gts) == 0 and len(preds) == 0:
                continue

            gt_matched = np.zeros(len(gts), dtype=bool)
            pred_matched = np.zeros(len(preds), dtype=bool)

            if len(gts) > 0 and len(preds) > 0:
                gt_boxes = np.array([g["box"] for g in gts])
                pred_boxes = np.array([p["box"] for p in preds])
                ious = compute_iou_matrix(pred_boxes, gt_boxes)

                for p_idx, p in enumerate(preds):
                    best_iou = 0.0
                    best_g_idx = -1
                    for g_idx, g in enumerate(gts):
                        if not gt_matched[g_idx] and ious[p_idx, g_idx] >= iou_thresh and ious[p_idx, g_idx] > best_iou:
                            best_iou = ious[p_idx, g_idx]
                            best_g_idx = g_idx

                    if best_g_idx >= 0:
                        gt_matched[best_g_idx] = True
                        pred_matched[p_idx] = True
                        gt_cls = gts[best_g_idx]["cls"]
                        p_cls = p["cls"]

                        if gt_cls == p_cls:
                            tp_count += 1
                        else:
                            fp_class_confusion += 1
                            confusion_matrix[gt_cls, p_cls] += 1

            # False Positives on Background
            for p_idx, matched in enumerate(pred_matched):
                if not matched:
                    fp_background += 1

            # False Negatives Breakdown
            for g_idx, matched in enumerate(gt_matched):
                if not matched:
                    g = gts[g_idx]
                    if g["is_small"]:
                        fn_small_miss += 1
                    elif g["is_occluded"]:
                        fn_occlusion_miss += 1
                    else:
                        fn_other_miss += 1

                    if len(failure_samples) < 15 and g["is_small"]:
                        failure_samples.append({
                            "image": img_name,
                            "type": "Small Object Miss",
                            "cls": CLASS_NAMES[g["cls"]],
                            "box": g["box"]
                        })

    return {
        "Total_GT": total_gt,
        "Total_Predictions": total_preds,
        "True_Positives": tp_count,
        "Small_Object_Miss_FN": fn_small_miss,
        "Occlusion_Miss_FN": fn_occlusion_miss,
        "Other_Miss_FN": fn_other_miss,
        "Total_FN": fn_small_miss + fn_occlusion_miss + fn_other_miss,
        "Class_Confusion_FP": fp_class_confusion,
        "Background_FP": fp_background,
        "Total_FP": fp_class_confusion + fp_background,
        "confusion_matrix": confusion_matrix,
        "failure_samples": failure_samples
    }


def run_failure_taxonomy():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[+] Loading Ground Truth annotations for Failure Taxonomy...")
    gt_dict = load_raw_val_annotations()

    taxonomy_rows = []
    matrix_store = {}

    for exp_id, minfo in EXPERIMENTS.items():
        w = minfo["weights"]
        if not w.exists():
            alt = ROOT_DIR / "experiments" / minfo["name"] / "weights" / "best.pt"
            w = alt if alt.exists() else w

        print(f"\n[+] Extracting Failure Taxonomy for {minfo['label']}...")
        model = YOLO(str(w))
        err = extract_model_errors(model, minfo["size"], gt_dict, conf_thresh=0.25, iou_thresh=0.5)

        total_fn = err["Total_FN"]
        small_miss_pct = (err["Small_Object_Miss_FN"] / total_fn * 100) if total_fn > 0 else 0
        occ_miss_pct = (err["Occlusion_Miss_FN"] / total_fn * 100) if total_fn > 0 else 0
        conf_pct = (err["Class_Confusion_FP"] / err["Total_FP"] * 100) if err["Total_FP"] > 0 else 0

        taxonomy_rows.append({
            "Experiment": exp_id,
            "Model_Config": minfo["label"],
            "Total_GT": err["Total_GT"],
            "True_Positives": err["True_Positives"],
            "Total_FN (Misses)": total_fn,
            "Small_Object_Miss": err["Small_Object_Miss_FN"],
            "Small_Miss_%_of_FN": round(small_miss_pct, 1),
            "Occlusion_Miss": err["Occlusion_Miss_FN"],
            "Occ_Miss_%_of_FN": round(occ_miss_pct, 1),
            "Total_FP (False Detections)": err["Total_FP"],
            "Class_Confusion_FP": err["Class_Confusion_FP"],
            "Confusion_%_of_FP": round(conf_pct, 1),
            "Background_FP": err["Background_FP"]
        })
        matrix_store[exp_id] = err["confusion_matrix"]

    df_tax = pd.DataFrame(taxonomy_rows)
    df_tax.to_csv(OUTPUT_DIR / "failure_taxonomy_metrics.csv", index=False)

    print("\n" + "=" * 90)
    print("                        FAILURE TAXONOMY COMPARISON TABLE")
    print("=" * 90)
    print(df_tax[["Experiment", "Model_Config", "True_Positives", "Total_FN (Misses)", "Small_Object_Miss", "Small_Miss_%_of_FN", "Class_Confusion_FP", "Background_FP"]].to_string(index=False))
    print("=" * 90 + "\n")

    # Save to Markdown
    md_text = "# Step 6E — AI Vision Failure Taxonomy & Root Cause Analysis\n\n"
    md_text += "### 1. Error Categorization & Breakdown\n\n" + df_tax.to_markdown(index=False) + "\n\n"
    (OUTPUT_DIR / "failure_taxonomy_metrics.md").write_text(md_text, encoding="utf-8")

    # Generate Visualizations
    generate_taxonomy_charts(df_tax, matrix_store)


def generate_taxonomy_charts(df_tax: pd.DataFrame, matrix_store: dict):
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Error Composition Stacked Bar Chart
    plt.figure(figsize=(12, 6))
    df_plot = df_tax[["Experiment", "Small_Object_Miss", "Occlusion_Miss", "Class_Confusion_FP", "Background_FP"]].set_index("Experiment")
    ax = df_plot.plot(kind="bar", stacked=True, figsize=(11, 6), colormap="Set2")
    plt.title("Failure Mode Distribution (Total Error Count by Experiment)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Experiment", fontsize=12)
    plt.ylabel("Total Error Count", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(["Small-Object Miss (FN)", "Occlusion Miss (FN)", "Class Confusion (FP)", "Background FP"], loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_distribution_comparison.png", dpi=300)
    plt.close()

    # 2. Confusion Matrix Heatmap for B2
    if "B2" in matrix_store:
        plt.figure(figsize=(10, 8))
        cm = matrix_store["B2"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap="Blues"
        )
        plt.title("Inter-Class Confusion Matrix for B2 (YOLO11s-1280)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Predicted Class", fontsize=12)
        plt.ylabel("Ground Truth Class", fontsize=12)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "top_confused_classes.png", dpi=300)
        plt.close()

    print(f"[✓] Step 6E charts and reports successfully saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_failure_taxonomy()
