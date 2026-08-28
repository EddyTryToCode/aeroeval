"""
Step 6A: Detailed Per-Class Evaluation across Experiments (A, B1, B2, B3).

Extracts:
- Per-class AP50, AP50-95, Precision, Recall for all 10 VisDrone categories
- Delta improvement calculations (B2 vs A, B2 vs B1, B2 vs B3)
- Focus group analysis: pedestrian, people, bicycle, tricycle, awning-tricycle, motor

Outputs:
- reports/per_class_metrics.csv
- reports/per_class_metrics.md
- reports/per_class_ap50_comparison.png
- reports/per_class_ap50_95_comparison.png
- reports/weak_classes_improvement.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"
REPORTS_DIR = ROOT_DIR / "reports"

EXPERIMENTS = {
    "A": {
        "name": "baseline_yolo11n",
        "label": "A (YOLO11n-640)",
        "model_file": ROOT_DIR / "experiments" / "baseline_yolo11n" / "weights" / "best.pt",
        "size": 640,
        "batch": 8
    },
    "B1": {
        "name": "exp_b1_yolo11s_960",
        "label": "B1 (YOLO11s-960)",
        "model_file": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b1_yolo11s_960" / "weights" / "best.pt",
        "size": 960,
        "batch": 16
    },
    "B2": {
        "name": "exp_b2_yolo11s_1280",
        "label": "B2 (YOLO11s-1280)",
        "model_file": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b2_yolo11s_1280" / "weights" / "best.pt",
        "size": 1280,
        "batch": 8
    },
    "B3": {
        "name": "exp_b3_yolo11m_960",
        "label": "B3 (YOLO11m-960)",
        "model_file": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b3_yolo11m_960" / "weights" / "best.pt",
        "size": 960,
        "batch": 16
    }
}

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

WEAK_CLASSES = ["pedestrian", "people", "bicycle", "tricycle", "awning-tricycle", "motor"]


def run_per_class_eval():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[+] Starting Per-Class Evaluation on device: {device}")

    all_results = []

    for exp_id, info in EXPERIMENTS.items():
        weights = info["model_file"]
        if not weights.exists():
            # Alternative path check
            alt = ROOT_DIR / "experiments" / info["name"] / "weights" / "best.pt"
            if alt.exists():
                weights = alt
            else:
                print(f"[-] Missing weights for {exp_id}: {weights}")
                continue

        print(f"\nEvaluating {exp_id} ({info['label']}) at imgsz={info['size']}...")
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(CONFIGS_DIR / "visdrone.yaml"),
            imgsz=info["size"],
            batch=info["batch"],
            device=device,
            split="val",
            verbose=False
        )

        ap50_per_class = metrics.box.ap50
        ap50_95_per_class = metrics.box.ap
        p_per_class = metrics.box.p
        r_per_class = metrics.box.r

        for idx, cls_name in enumerate(CLASS_NAMES):
            ap50_val = float(ap50_per_class[idx]) if idx < len(ap50_per_class) else 0.0
            ap50_95_val = float(ap50_95_per_class[idx]) if idx < len(ap50_95_per_class) else 0.0
            p_val = float(p_per_class[idx]) if idx < len(p_per_class) else 0.0
            r_val = float(r_per_class[idx]) if idx < len(r_per_class) else 0.0

            all_results.append({
                "Experiment": exp_id,
                "Model_Config": info["label"],
                "Class_ID": idx,
                "Class_Name": cls_name,
                "Is_Weak_Group": cls_name in WEAK_CLASSES,
                "AP50": round(ap50_val, 3),
                "AP50-95": round(ap50_95_val, 3),
                "Precision": round(p_val, 3),
                "Recall": round(r_val, 3)
            })

    df = pd.DataFrame(all_results)
    df.to_csv(REPORTS_DIR / "per_class_metrics.csv", index=False)

    # Pivot Tables for Easy Reading
    pivot_ap50 = df.pivot(index="Class_Name", columns="Experiment", values="AP50").reindex(CLASS_NAMES)
    pivot_ap50_95 = df.pivot(index="Class_Name", columns="Experiment", values="AP50-95").reindex(CLASS_NAMES)

    # Calculate Deltas
    if "A" in pivot_ap50.columns and "B2" in pivot_ap50.columns:
        pivot_ap50["Δ (B2 - A)"] = (pivot_ap50["B2"] - pivot_ap50["A"]).round(3)
        pivot_ap50["Gain (%)"] = ((pivot_ap50["B2"] - pivot_ap50["A"]) / pivot_ap50["A"] * 100).round(1)
        pivot_ap50_95["Δ (B2 - A)"] = (pivot_ap50_95["B2"] - pivot_ap50_95["A"]).round(3)
        pivot_ap50_95["Gain (%)"] = ((pivot_ap50_95["B2"] - pivot_ap50_95["A"]) / pivot_ap50_95["A"] * 100).round(1)

    print("\n" + "=" * 85)
    print("                     PER-CLASS AP50 COMPARISON TABLE")
    print("=" * 85)
    print(pivot_ap50.to_string())

    print("\n" + "=" * 85)
    print("                    PER-CLASS AP50-95 COMPARISON TABLE")
    print("=" * 85)
    print(pivot_ap50_95.to_string())
    print("=" * 85 + "\n")

    # Save to Markdown
    md_content = "# Step 6A — Per-Class Evaluation Breakdown\n\n"
    md_content += "### 1. Per-Class AP50\n\n" + pivot_ap50.to_markdown() + "\n\n"
    md_content += "### 2. Per-Class AP50-95\n\n" + pivot_ap50_95.to_markdown() + "\n\n"
    (REPORTS_DIR / "per_class_metrics.md").write_text(md_content, encoding="utf-8")

    # Visualizations
    generate_per_class_charts(df, pivot_ap50, pivot_ap50_95)


def generate_per_class_charts(df: pd.DataFrame, pivot_ap50: pd.DataFrame, pivot_ap50_95: pd.DataFrame):
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Bar Chart: AP50 across all classes
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=df,
        x="Class_Name",
        y="AP50",
        hue="Model_Config",
        palette="crest"
    )
    plt.title("Per-Class AP50 Comparison (A vs B1 vs B2 vs B3)", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("AP50", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.legend(title="Experiment", loc="upper right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "per_class_ap50_comparison.png", dpi=300)
    plt.close()

    # 2. Bar Chart: AP50-95 across all classes
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=df,
        x="Class_Name",
        y="AP50-95",
        hue="Model_Config",
        palette="viridis"
    )
    plt.title("Per-Class AP50-95 Comparison (A vs B1 vs B2 vs B3)", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("AP50-95", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 0.7)
    plt.legend(title="Experiment", loc="upper right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "per_class_ap50_95_comparison.png", dpi=300)
    plt.close()

    # 3. Weak Classes Specific Gain Focus
    df_weak = df[df["Is_Weak_Group"]].copy()
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_weak,
        x="Class_Name",
        y="AP50",
        hue="Model_Config",
        palette="magma"
    )
    plt.title("Performance on Drone Small/Weak Classes (Pedestrians, 2-Wheelers)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Challenging Category", fontsize=12)
    plt.ylabel("AP50", fontsize=12)
    plt.ylim(0, 0.8)
    plt.legend(title="Experiment", loc="upper left")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "weak_classes_improvement.png", dpi=300)
    plt.close()

    print(f"[✓] Step 6A charts and reports successfully saved to {REPORTS_DIR}")


if __name__ == "__main__":
    run_per_class_eval()
