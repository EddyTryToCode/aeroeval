"""
Step 6C: Comprehensive Robustness Benchmark across Candidate Models (B1 vs B2 vs A vs B3).

Applies controlled environmental corruptions:
1. Gaussian Blur (Atmospheric haze / defocus)
2. Motion Blur (Drone velocity / vibration)
3. Low Light (Dawn / dusk / night flight)
4. Overexposure (Direct sunlight glare)
5. Gaussian Noise (Low-cost sensor noise)
6. JPEG Compression (Video downlink compression artifacts)
7. Occlusion (Drone propeller / sensor lens dust)
8. Resolution Degradation (Bandwidth constraint downscaling)

Computes:
- mAP50 under each corruption across 3 severity levels (Mild, Medium, Heavy)
- Relative mAP drop: Δ = Clean_mAP - Corrupted_mAP
- Robustness Retention Score (%): (Corrupted_mAP / Clean_mAP) * 100
- Robustness degradation curves and comparison heatmap

Outputs:
- reports/robustness_benchmark_metrics.csv
- reports/robustness_benchmark_metrics.md
- reports/robustness_heatmap.png
- reports/robustness_degradation_curves.png
- reports/robustness_comparison_b1_b2.png
"""

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from aeroeval.robustness.corruptions import CORRUPTIONS

DATA_DIR = ROOT_DIR / "data" / "visdrone_yolo"
REPORTS_DIR = ROOT_DIR / "reports"
CONFIGS_DIR = ROOT_DIR / "configs"

MODELS_TO_EVALUATE = {
    "A": {
        "label": "A (YOLO11n-640)",
        "weights": ROOT_DIR / "experiments" / "baseline_yolo11n" / "weights" / "best.pt",
        "size": 640,
        "batch": 8
    },
    "B1": {
        "label": "B1 (YOLO11s-960)",
        "weights": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b1_yolo11s_960" / "weights" / "best.pt",
        "size": 960,
        "batch": 16
    },
    "B2": {
        "label": "B2 (YOLO11s-1280)",
        "weights": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b2_yolo11s_1280" / "weights" / "best.pt",
        "size": 1280,
        "batch": 8
    },
    "B3": {
        "label": "B3 (YOLO11m-960)",
        "weights": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b3_yolo11m_960" / "weights" / "best.pt",
        "size": 960,
        "batch": 16
    }
}


def create_corrupted_dataset(temp_dir: Path, corruption_name: str, severity: int):
    val_img_dir = DATA_DIR / "images" / "val"
    val_lbl_dir = DATA_DIR / "labels" / "val"

    corrupt_img_dir = temp_dir / "images" / "val"
    corrupt_lbl_dir = temp_dir / "labels" / "val"

    corrupt_img_dir.mkdir(parents=True, exist_ok=True)
    corrupt_lbl_dir.mkdir(parents=True, exist_ok=True)

    transform_fn = CORRUPTIONS[corruption_name]

    for img_p in val_img_dir.glob("*.jpg"):
        img = cv2.imread(str(img_p))
        if img is not None:
            corrupted = transform_fn(img, severity)
            cv2.imwrite(str(corrupt_img_dir / img_p.name), corrupted)

        # Labels stay unchanged
        src_lbl = val_lbl_dir / f"{img_p.stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, corrupt_lbl_dir / src_lbl.name)

    # Create temporary dataset yaml
    dataset_yaml = {
        "path": str(temp_dir),
        "train": str(temp_dir / "images" / "val"),  # dummy
        "val": "images/val",
        "nc": 10,
        "names": {
            0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
            5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"
        }
    }
    yaml_path = temp_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f)

    return yaml_path


def run_robustness_benchmark(sample_val_size: int = 50):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 80)
    print(f"       STARTING COMPREHENSIVE ROBUSTNESS BENCHMARK (Sample {sample_val_size} imgs)")
    print("=" * 80)

    val_img_dir = DATA_DIR / "images" / "val"
    val_lbl_dir = DATA_DIR / "labels" / "val"
    all_val_imgs = sorted(list(val_img_dir.glob("*.jpg")))

    np.random.seed(42)
    selected_imgs = list(np.random.choice(all_val_imgs, min(sample_val_size, len(all_val_imgs)), replace=False))
    print(f"[+] Using deterministic sample of {len(selected_imgs)} val images.")

    # Load models
    loaded_models = {}
    for exp_id, minfo in MODELS_TO_EVALUATE.items():
        w = minfo["weights"]
        if not w.exists():
            alt = ROOT_DIR / "experiments" / minfo["weights"].parent.parent.name / "weights" / "best.pt"
            w = alt if alt.exists() else w
        loaded_models[exp_id] = YOLO(str(w))

    # Clean Baseline
    clean_baselines = {}
    with tempfile.TemporaryDirectory() as tmp_clean:
        tmp_clean_dir = Path(tmp_clean)
        clean_img_dir = tmp_clean_dir / "images" / "val"
        clean_lbl_dir = tmp_clean_dir / "labels" / "val"
        clean_img_dir.mkdir(parents=True, exist_ok=True)
        clean_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_p in selected_imgs:
            shutil.copy2(img_p, clean_img_dir / img_p.name)
            src_lbl = val_lbl_dir / f"{img_p.stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, clean_lbl_dir / src_lbl.name)

        clean_yaml = {
            "path": str(tmp_clean_dir),
            "train": str(clean_img_dir),
            "val": "images/val",
            "nc": 10,
            "names": {
                0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
                5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"
            }
        }
        clean_yaml_path = tmp_clean_dir / "data.yaml"
        with open(clean_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(clean_yaml, f)

        for exp_id, minfo in MODELS_TO_EVALUATE.items():
            model = loaded_models[exp_id]
            val_res = model.val(
                data=str(clean_yaml_path),
                imgsz=minfo["size"],
                batch=minfo["batch"],
                device=device,
                split="val",
                verbose=False
            )
            clean_baselines[exp_id] = {
                "mAP50": float(val_res.box.map50),
                "mAP50-95": float(val_res.box.map)
            }
            print(f"[+] {minfo['label']} Clean Sample mAP50 = {clean_baselines[exp_id]['mAP50']:.3f}")

    results_list = []
    severities = [
        (0, "Mild (Level 1)"),
        (1, "Medium (Level 2)"),
        (2, "Heavy (Level 3)")
    ]

    for corr_name, transform_fn in CORRUPTIONS.items():
        print(f"\n---> Testing Corruption: {corr_name}")
        for sev_idx, sev_label in severities:
            with tempfile.TemporaryDirectory() as tmp_d:
                temp_dir = Path(tmp_d)
                corrupt_img_dir = temp_dir / "images" / "val"
                corrupt_lbl_dir = temp_dir / "labels" / "val"
                corrupt_img_dir.mkdir(parents=True, exist_ok=True)
                corrupt_lbl_dir.mkdir(parents=True, exist_ok=True)

                for img_p in selected_imgs:
                    img = cv2.imread(str(img_p))
                    if img is not None:
                        corrupted = transform_fn(img, sev_idx)
                        cv2.imwrite(str(corrupt_img_dir / img_p.name), corrupted)
                    src_lbl = val_lbl_dir / f"{img_p.stem}.txt"
                    if src_lbl.exists():
                        shutil.copy2(src_lbl, corrupt_lbl_dir / src_lbl.name)

                dataset_yaml = {
                    "path": str(temp_dir),
                    "train": str(corrupt_img_dir),
                    "val": "images/val",
                    "nc": 10,
                    "names": {
                        0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
                        5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"
                    }
                }
                yaml_path = temp_dir / "data.yaml"
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(dataset_yaml, f)

                for exp_id, minfo in MODELS_TO_EVALUATE.items():
                    model = loaded_models[exp_id]
                    res = model.val(
                        data=str(yaml_path),
                        imgsz=minfo["size"],
                        batch=minfo["batch"],
                        device=device,
                        split="val",
                        verbose=False
                    )

                    c_map50 = float(res.box.map50)
                    c_map = float(res.box.map)
                    clean_m50 = clean_baselines[exp_id]["mAP50"]
                    drop50 = clean_m50 - c_map50
                    retention_pct = (c_map50 / clean_m50 * 100.0) if clean_m50 > 0 else 0.0

                    results_list.append({
                        "Experiment": exp_id,
                        "Model_Config": minfo["label"],
                        "Corruption": corr_name,
                        "Severity_Index": sev_idx + 1,
                        "Severity": sev_label,
                        "Clean_mAP50": round(clean_m50, 3),
                        "Corrupted_mAP50": round(c_map50, 3),
                        "mAP50_Drop": round(drop50, 3),
                        "Retention_Rate_%": round(retention_pct, 1),
                        "Corrupted_mAP50-95": round(c_map, 3)
                    })

    df_rob = pd.DataFrame(results_list)
    df_rob.to_csv(REPORTS_DIR / "robustness_benchmark_metrics.csv", index=False)

    avg_retention = df_rob.groupby(["Experiment", "Model_Config"])["Retention_Rate_%"].mean().round(1)
    print("\n" + "=" * 80)
    print("              OVERALL MODEL ROBUSTNESS RETENTION SCORE (%)")
    print("=" * 80)
    print(avg_retention.to_string())
    print("=" * 80)

    pivot_corr = df_rob.pivot_table(
        index="Corruption",
        columns="Experiment",
        values="Corrupted_mAP50",
        aggfunc="mean"
    ).round(3)

    print("\n" + "=" * 80)
    print("          MEAN CORRUPTED mAP50 BY CORRUPTION TYPE (Across Severities)")
    print("=" * 80)
    print(pivot_corr.to_string())
    print("=" * 80 + "\n")

    md_content = "# Step 6C — Drone Vision Robustness Benchmark\n\n"
    md_content += "### 1. Overall Average Retention Rate (% of Clean Performance Retained)\n\n"
    md_content += avg_retention.to_frame().to_markdown() + "\n\n"
    md_content += "### 2. Mean Corrupted mAP50 under Environmental Shifts\n\n"
    md_content += pivot_corr.to_markdown() + "\n\n"
    (REPORTS_DIR / "robustness_benchmark_metrics.md").write_text(md_content, encoding="utf-8")

    generate_robustness_charts(df_rob, pivot_corr)


def generate_robustness_charts(df_rob: pd.DataFrame, pivot_corr: pd.DataFrame):
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Heatmap: Corruption vs Model (Mean mAP50)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_corr,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={'label': 'Mean Corrupted mAP50'}
    )
    plt.title("Robustness Matrix: Mean mAP50 under Environmental Perturbations", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Corruption Perturbation", fontsize=12)
    plt.xlabel("Model Experiment", fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "robustness_heatmap.png", dpi=300)
    plt.close()

    # 2. Side-by-Side Comparison of B1 (960px) vs B2 (1280px)
    df_b1_b2 = df_rob[df_rob["Experiment"].isin(["B1", "B2"])].copy()
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=df_b1_b2,
        x="Corruption",
        y="Corrupted_mAP50",
        hue="Model_Config",
        palette="crest"
    )
    plt.title("Candidate Comparison: B1 (YOLO11s-960) vs B2 (YOLO11s-1280) under Corruptions", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Corruption Type", fontsize=12)
    plt.ylabel("Corrupted mAP50", fontsize=12)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 0.6)
    plt.legend(title="Candidate Model", loc="upper right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "robustness_comparison_b1_b2.png", dpi=300)
    plt.close()

    # 3. Degradation Curves across Severity (1 -> 2 -> 3)
    plt.figure(figsize=(14, 8))
    g = sns.FacetGrid(df_rob, col="Corruption", col_wrap=4, height=3.2, sharey=True)
    g.map_dataframe(
        sns.lineplot,
        x="Severity_Index",
        y="Corrupted_mAP50",
        hue="Experiment",
        marker="o",
        palette="tab10"
    )
    g.set_axis_labels("Severity (1:Mild → 3:Heavy)", "mAP50")
    g.add_legend(title="Model")
    g.fig.subplots_adjust(top=0.90)
    g.fig.suptitle("Performance Degradation Curves across Severity Levels", fontsize=15, fontweight="bold")
    plt.savefig(REPORTS_DIR / "robustness_degradation_curves.png", dpi=300)
    plt.close()

    print(f"[✓] Step 6C charts and reports successfully saved to {REPORTS_DIR}")


if __name__ == "__main__":
    run_robustness_benchmark()
