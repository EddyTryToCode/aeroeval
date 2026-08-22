"""
Step 6B: Object Size Evaluation (Small vs Medium vs Large) across Experiments (A, B1, B2, B3).

Standard COCO Size Criteria (measured by bounding box pixel area = width * height):
- Small:  area < 32^2 (1024 px^2)
- Medium: 32^2 <= area < 96^2 (1024 to 9216 px^2)
- Large:  area >= 96^2 (>= 9216 px^2)

Evaluates:
- mAP50, mAP50-95, Precision, Recall partitioned strictly by Ground Truth object scale
- Evaluates prediction-to-ground-truth matches with IoU thresholds [0.5 : 0.95]
- Proves/tests hypothesis: "Higher spatial resolution (B2 @ 1280px) provides disproportionate gains for small drone-view objects"

Outputs:
- reports/object_size_metrics.csv
- reports/object_size_metrics.md
- reports/size_map50_comparison.png
- reports/size_gain_relative_chart.png
"""

import sys
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
REPORTS_DIR = ROOT_DIR / "reports"

EXPERIMENTS = {
    "A": {
        "name": "baseline_yolo11n",
        "label": "A (YOLO11n-640)",
        "model_file": ROOT_DIR / "experiments" / "baseline_yolo11n" / "weights" / "best.pt",
        "size": 640
    },
    "B1": {
        "name": "exp_b1_yolo11s_960",
        "label": "B1 (YOLO11s-960)",
        "model_file": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b1_yolo11s_960" / "weights" / "best.pt",
        "size": 960
    },
    "B2": {
        "name": "exp_b2_yolo11s_1280",
        "label": "B2 (YOLO11s-1280)",
        "model_file": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b2_yolo11s_1280" / "weights" / "best.pt",
        "size": 1280
    },
    "B3": {
        "name": "exp_b3_yolo11m_960",
        "label": "B3 (YOLO11m-960)",
        "model_file": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b3_yolo11m_960" / "weights" / "best.pt",
        "size": 960
    }
}

SIZE_RANGES = {
    "small": (0, 32 ** 2),
    "medium": (32 ** 2, 96 ** 2),
    "large": (96 ** 2, float("inf")),
    "all": (0, float("inf"))
}


def box_iou_batch(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes (x1, y1, x2, y2).
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1)
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1)
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2)
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1[:, None] + area2 - inter_area
    return np.where(union > 0, inter_area / union, 0.0)


def calculate_ap_for_size(gt_by_img, pred_by_img, size_category, iou_thresh=0.5):
    """
    Calculates Average Precision for a specific object size bucket.
    """
    min_area, max_area = SIZE_RANGES[size_category]

    total_gt = 0
    all_scores = []
    all_matches = []

    for img_name, gts in gt_by_img.items():
        preds = pred_by_img.get(img_name, [])

        # Filter GT by area
        valid_gts = [g for g in gts if min_area <= g["area"] < max_area]
        total_gt += len(valid_gts)

        if not valid_gts and not preds:
            continue

        if not valid_gts and preds:
            for p in preds:
                if min_area <= p["area"] < max_area:
                    all_scores.append(p["score"])
                    all_matches.append(0)
            continue

        # Convert to numpy
        gt_boxes = np.array([g["box"] for g in valid_gts])
        gt_classes = np.array([g["cls"] for g in valid_gts])
        gt_matched = np.zeros(len(valid_gts), dtype=bool)

        # Filter preds by approximate target area bucket
        candidate_preds = [p for p in preds if min_area * 0.5 <= p["area"] <= max_area * 1.5]
        candidate_preds.sort(key=lambda x: x["score"], reverse=True)

        if not candidate_preds:
            continue

        pred_boxes = np.array([p["box"] for p in candidate_preds])
        pred_classes = np.array([p["cls"] for p in candidate_preds])
        pred_scores = np.array([p["score"] for p in candidate_preds])

        ious = box_iou_batch(pred_boxes, gt_boxes)

        for p_idx in range(len(candidate_preds)):
            score = pred_scores[p_idx]
            p_cls = pred_classes[p_idx]
            all_scores.append(score)

            best_iou = 0.0
            best_gt_idx = -1
            for g_idx in range(len(valid_gts)):
                if not gt_matched[g_idx] and gt_classes[g_idx] == p_cls:
                    iou = ious[p_idx, g_idx]
                    if iou >= iou_thresh and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            if best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                all_matches.append(1)
            else:
                all_matches.append(0)

    if total_gt == 0 or len(all_scores) == 0:
        return 0.0, 0.0, 0.0, total_gt

    # Sort all predictions by confidence
    sort_indices = np.argsort(-np.array(all_scores))
    tp = np.array(all_matches)[sort_indices]
    fp = 1 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / total_gt
    precisions = cum_tp / (cum_tp + cum_fp + 1e-16)

    # 11-point interpolated AP
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        prec_above = precisions[recalls >= t]
        p = np.max(prec_above) if len(prec_above) > 0 else 0.0
        ap += p / 11.0

    max_p = np.max(precisions) if len(precisions) > 0 else 0.0
    final_r = recalls[-1] if len(recalls) > 0 else 0.0
    return float(ap), float(max_p), float(final_r), int(total_gt)


def load_val_ground_truth():
    img_dir = DATA_DIR / "images" / "val"
    lbl_dir = DATA_DIR / "labels" / "val"

    gt_by_img = {}
    for img_path in img_dir.glob("*.jpg"):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        with Image.open(img_path) as im:
            w_img, h_img = im.size

        boxes = []
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    box_w = bw * w_img
                    box_h = bh * h_img
                    x1 = (xc * w_img) - (box_w / 2.0)
                    y1 = (yc * h_img) - (box_h / 2.0)
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    boxes.append({
                        "cls": cls_id,
                        "box": [x1, y1, x2, y2],
                        "area": box_w * box_h
                    })
        gt_by_img[img_path.name] = boxes
    return gt_by_img


def extract_predictions(model: YOLO, imgsz: int, batch_size: int = 16):
    val_img_dir = DATA_DIR / "images" / "val"
    img_files = sorted(list(val_img_dir.glob("*.jpg")))
    
    pred_by_img = {}
    device = 0 if torch.cuda.is_available() else "cpu"

    # Process in batches to avoid GPU stream race conditions and OOM
    for i in range(0, len(img_files), batch_size):
        batch_files = img_files[i : i + batch_size]
        results = model.predict(
            source=batch_files,
            imgsz=imgsz,
            conf=0.001,  # low threshold to compute full PR curve
            iou=0.65,
            device=device,
            verbose=False
        )

        for r in results:
            img_name = Path(r.path).name
            preds = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes_data = r.boxes.data.cpu().numpy()
                for b in boxes_data:
                    x1, y1, x2, y2, score, cls_id = b[:6]
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    preds.append({
                        "cls": int(cls_id),
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "score": float(score),
                        "area": float(w * h)
                    })
            pred_by_img[img_name] = preds
    return pred_by_img


def run_object_size_eval():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[+] Loading Ground Truth for VisDrone Val...")
    gt_by_img = load_val_ground_truth()

    results_table = []

    for exp_id, info in EXPERIMENTS.items():
        weights = info["model_file"]
        if not weights.exists():
            alt = ROOT_DIR / "experiments" / info["name"] / "weights" / "best.pt"
            if alt.exists():
                weights = alt
            else:
                print(f"[-] Missing weights for {exp_id}")
                continue

        print(f"\n[+] Extracting predictions for {info['label']} (imgsz={info['size']})...")
        model = YOLO(str(weights))
        preds = extract_predictions(model, info["size"])

        for size_cat in ["small", "medium", "large", "all"]:
            # Evaluate at IoU=0.5
            ap50, p, r, gt_count = calculate_ap_for_size(gt_by_img, preds, size_cat, iou_thresh=0.5)
            
            # Evaluate across IoU [0.5:0.95] for AP50-95
            iou_aps = []
            for iou_val in np.arange(0.50, 0.96, 0.05):
                ap_iou, _, _, _ = calculate_ap_for_size(gt_by_img, preds, size_cat, iou_thresh=iou_val)
                iou_aps.append(ap_iou)
            ap50_95 = float(np.mean(iou_aps))

            results_table.append({
                "Experiment": exp_id,
                "Model_Config": info["label"],
                "Size_Category": size_cat.capitalize(),
                "GT_Objects": gt_count,
                "AP50": round(ap50, 3),
                "AP50-95": round(ap50_95, 3),
                "Precision": round(p, 3),
                "Recall": round(r, 3)
            })

    df_size = pd.DataFrame(results_table)
    df_size.to_csv(REPORTS_DIR / "object_size_metrics.csv", index=False)

    # Pivot Tables
    pivot_size_ap50 = df_size.pivot(index="Size_Category", columns="Experiment", values="AP50")
    pivot_size_ap50_95 = df_size.pivot(index="Size_Category", columns="Experiment", values="AP50-95")

    order = ["Small", "Medium", "Large", "All"]
    pivot_size_ap50 = pivot_size_ap50.reindex(order)
    pivot_size_ap50_95 = pivot_size_ap50_95.reindex(order)

    if "A" in pivot_size_ap50.columns and "B2" in pivot_size_ap50.columns:
        pivot_size_ap50["Δ (B2 - A)"] = (pivot_size_ap50["B2"] - pivot_size_ap50["A"]).round(3)
        pivot_size_ap50["Gain (%)"] = ((pivot_size_ap50["B2"] - pivot_size_ap50["A"]) / pivot_size_ap50["A"] * 100).round(1)

        pivot_size_ap50_95["Δ (B2 - A)"] = (pivot_size_ap50_95["B2"] - pivot_size_ap50_95["A"]).round(3)
        pivot_size_ap50_95["Gain (%)"] = ((pivot_size_ap50_95["B2"] - pivot_size_ap50_95["A"]) / pivot_size_ap50_95["A"] * 100).round(1)

    print("\n" + "=" * 80)
    print("           OBJECT SIZE EVALUATION (AP50 by Scale)")
    print("=" * 80)
    print(pivot_size_ap50.to_string())

    print("\n" + "=" * 80)
    print("          OBJECT SIZE EVALUATION (AP50-95 by Scale)")
    print("=" * 80)
    print(pivot_size_ap50_95.to_string())
    print("=" * 80 + "\n")

    # Save to Markdown
    md_text = "# Step 6B — Object Size Evaluation Breakdown\n\n"
    md_text += "### 1. AP50 by Object Size Category\n\n" + pivot_size_ap50.to_markdown() + "\n\n"
    md_text += "### 2. AP50-95 by Object Size Category\n\n" + pivot_size_ap50_95.to_markdown() + "\n\n"
    (REPORTS_DIR / "object_size_metrics.md").write_text(md_text, encoding="utf-8")

    # Generate Visualization Charts
    generate_size_charts(df_size, pivot_size_ap50)


def generate_size_charts(df_size: pd.DataFrame, pivot_ap50: pd.DataFrame):
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Bar Chart: AP50 across Small, Medium, Large
    plt.figure(figsize=(11, 6))
    df_filtered = df_size[df_size["Size_Category"] != "All"].copy()
    ax = sns.barplot(
        data=df_filtered,
        x="Size_Category",
        y="AP50",
        hue="Model_Config",
        palette="viridis"
    )
    plt.title("Object Scale Performance: Small (<32²) vs Medium (32²-96²) vs Large (>96²)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Object Size Scale", fontsize=12)
    plt.ylabel("AP50", fontsize=12)
    plt.ylim(0, 0.85)
    plt.legend(title="Experiment", loc="upper left")
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "size_map50_comparison.png", dpi=300)
    plt.close()

    # 2. Percentage Gain Chart for B2 vs A
    if "Gain (%)" in pivot_ap50.columns:
        plt.figure(figsize=(9, 5))
        gains = pivot_ap50.loc[["Small", "Medium", "Large"], "Gain (%)"]
        ax2 = sns.barplot(x=gains.index, y=gains.values, palette="rocket")
        plt.title("Relative Performance Gain of B2 (1280px) over Baseline A (640px)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Object Size Category", fontsize=12)
        plt.ylabel("Relative Gain (%)", fontsize=12)
        for p in ax2.patches:
            h = p.get_height()
            if not np.isnan(h):
                ax2.annotate(f"+{h:.1f}%", (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=11, fontweight="bold", xytext=(0, 4), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "size_gain_relative_chart.png", dpi=300)
        plt.close()

    print(f"[✓] Step 6B charts and reports successfully saved to {REPORTS_DIR}")


if __name__ == "__main__":
    run_object_size_eval()
