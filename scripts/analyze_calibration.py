"""
Step 10: Confidence & Calibration Analysis across Models (A, B1, B2, B3).

Analyzes:
1. Confidence distribution (Correct vs Incorrect detections)
2. Precision-Recall-F1 vs Confidence Threshold curve sweep
3. Optimal F1-maximizing operating threshold
4. Reliability Diagram & Expected Calibration Error (ECE)

Outputs:
- reports/calibration/optimal_thresholds.json
- reports/calibration/threshold_sweep_metrics.csv
- reports/calibration/threshold_sweep_curves.png
- reports/calibration/confidence_distribution_tp_fp.png
- reports/calibration/reliability_diagram.png
- reports/calibration/calibration_report.md
"""

import json
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
OUTPUT_DIR = REPORTS_DIR / "calibration"

import sys
sys.path.append(str(ROOT_DIR / "src"))
from aeroeval.metrics.calibration import evaluate_calibration

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


def load_val_gt():
    img_dir = DATA_DIR / "images" / "val"
    lbl_dir = DATA_DIR / "labels" / "val"

    gt_by_img = {}
    for img_p in img_dir.glob("*.jpg"):
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        with Image.open(img_p) as im:
            w_img, h_img = im.size

        boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            for l in lines:
                p = l.split()
                if len(p) == 5:
                    cls_id = int(p[0])
                    xc, yc, bw, bh = map(float, p[1:])
                    bw_px, bh_px = bw * w_img, bh * h_img
                    x1 = (xc * w_img) - (bw_px / 2.0)
                    y1 = (yc * h_img) - (bh_px / 2.0)
                    boxes.append({
                        "cls": cls_id,
                        "box": [x1, y1, x1 + bw_px, y1 + bh_px]
                    })
        gt_by_img[img_p.name] = boxes
    return gt_by_img


def extract_raw_predictions(model: YOLO, imgsz: int):
    val_img_dir = DATA_DIR / "images" / "val"
    img_files = sorted(list(val_img_dir.glob("*.jpg")))
    device = 0 if torch.cuda.is_available() else "cpu"

    pred_by_img = {}
    batch_size = 16
    for i in range(0, len(img_files), batch_size):
        batch = img_files[i : i + batch_size]
        results = model.predict(
            source=batch,
            imgsz=imgsz,
            conf=0.01,  # capture wide range for calibration
            iou=0.60,
            device=device,
            verbose=False
        )
        for r in results:
            img_name = Path(r.path).name
            preds = []
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes.data.cpu().numpy():
                    preds.append({
                        "cls": int(b[5]),
                        "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "score": float(b[4])
                    })
            pred_by_img[img_name] = preds
    return pred_by_img


def run_calibration_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[+] Loading Ground Truth for VisDrone Val...")
    gt_by_img = load_val_gt()

    all_sweeps = []
    optimal_dict = {}
    reliability_dict = {}
    preds_dict = {}

    for exp_id, minfo in EXPERIMENTS.items():
        w = minfo["weights"]
        if not w.exists():
            alt = ROOT_DIR / "experiments" / minfo["name"] / "weights" / "best.pt"
            w = alt if alt.exists() else w

        print(f"\n[+] Processing Calibration Analysis for {minfo['label']}...")
        model = YOLO(str(w))
        preds = extract_raw_predictions(model, minfo["size"])

        calib_out = evaluate_calibration(gt_by_img, preds, iou_thresh=0.5)

        df_sw = calib_out["df_sweep"]
        df_sw["Experiment"] = exp_id
        df_sw["Model_Config"] = minfo["label"]
        all_sweeps.append(df_sw)

        opt = calib_out["best_operating_point"]
        opt["ECE"] = calib_out["ece"]
        optimal_dict[exp_id] = {
            "Model": minfo["label"],
            **opt
        }

        reliability_dict[exp_id] = calib_out["reliability_table"]
        preds_dict[exp_id] = calib_out["df_predictions"]

        print(f"    * Optimal Threshold: {opt['optimal_threshold']:.2f} -> Max F1 = {opt['max_f1']:.3f} (P={opt['precision_at_optimal']:.3f}, R={opt['recall_at_optimal']:.3f}, ECE={opt['ECE']:.4f})")

    # Combine Sweeps
    df_all_sweeps = pd.concat(all_sweeps, ignore_index=True)
    df_all_sweeps.to_csv(OUTPUT_DIR / "threshold_sweep_metrics.csv", index=False)

    # Save Optimal JSON
    with open(OUTPUT_DIR / "optimal_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(optimal_dict, f, indent=4)

    df_opt = pd.DataFrame.from_dict(optimal_dict, orient="index")
    print("\n" + "=" * 85)
    print("                     OPTIMAL OPERATING THRESHOLDS (F1 Maximizing)")
    print("=" * 85)
    print(df_opt.to_string())
    print("=" * 85 + "\n")

    # Save to Markdown
    md_content = "# Step 10 — Confidence & Calibration Analysis\n\n"
    md_content += "### 1. Optimal Confidence Thresholds & Calibration Errors\n\n"
    md_content += df_opt.to_markdown() + "\n\n"
    (OUTPUT_DIR / "calibration_report.md").write_text(md_content, encoding="utf-8")

    # Generate Visualizations
    generate_calibration_charts(df_all_sweeps, preds_dict, reliability_dict)


def generate_calibration_charts(df_sweeps: pd.DataFrame, preds_dict: dict, reliability_dict: dict):
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Precision-Recall-F1 vs Confidence Threshold Curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (exp, sub_df) in enumerate(df_sweeps.groupby("Experiment")):
        ax = axes[idx]
        ax.plot(sub_df["Threshold"], sub_df["Precision"], label="Precision", color="blue", lw=2)
        ax.plot(sub_df["Threshold"], sub_df["Recall"], label="Recall", color="green", lw=2)
        ax.plot(sub_df["Threshold"], sub_df["F1_Score"], label="F1-Score", color="crimson", lw=2.5, linestyle="--")

        # Find best F1
        best_r = sub_df.loc[sub_df["F1_Score"].idxmax()]
        ax.axvline(x=best_r["Threshold"], color="black", linestyle=":", alpha=0.8, label=f"Best Thresh ({best_r['Threshold']:.2f})")
        ax.scatter([best_r["Threshold"]], [best_r["F1_Score"]], color="crimson", s=70, zorder=5)

        ax.set_title(f"Exp {exp} ({sub_df['Model_Config'].iloc[0]})", fontweight="bold", fontsize=12)
        ax.set_xlabel("Confidence Threshold", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=9)

    plt.suptitle("Threshold Sensitivity & Optimal Operating Point Selection", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_sweep_curves.png", dpi=300)
    plt.close()

    # 2. Confidence Distribution of True Positives (Correct) vs False Positives (Incorrect)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, exp in enumerate(["B1", "B2"]):
        ax = axes[idx]
        df_p = preds_dict[exp]
        sns.histplot(
            data=df_p,
            x="score",
            hue="is_correct",
            bins=30,
            palette={1: "#2ecc71", 0: "#e74c3c"},
            kde=True,
            element="step",
            ax=ax
        )
        ax.set_title(f"Confidence Profile: Exp {exp} (1=Correct TP, 0=Incorrect FP)", fontweight="bold", fontsize=12)
        ax.set_xlabel("Predicted Confidence Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(["Correct (TP)", "Incorrect (FP)"], loc="upper center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confidence_distribution_tp_fp.png", dpi=300)
    plt.close()

    # 3. Reliability Diagram (Calibration Curve) for B1 and B2
    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration (Ideal)")

    colors = {"A": "orange", "B1": "blue", "B2": "green", "B3": "purple"}
    for exp, rel_df in reliability_dict.items():
        valid_b = rel_df[rel_df["Count"] > 0]
        plt.plot(valid_b["Avg_Confidence"], valid_b["Accuracy"], marker="o", lw=2, label=f"Exp {exp}", color=colors.get(exp))

    plt.title("Reliability Diagram (Model Calibration)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Confidence (Predicted Probability)", fontsize=12)
    plt.ylabel("Accuracy (Empirical Precision)", fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reliability_diagram.png", dpi=300)
    plt.close()

    print(f"[✓] Step 10 charts and report successfully saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_calibration_analysis()
