"""
Step 6F: Deployment Trade-off & Multi-Criteria Decision Analysis (MCDA).

Synthesizes:
1. Detection Accuracy (mAP50, mAP50-95)
2. Environmental Robustness (Mean Retention Score % across 8 corruptions)
3. Inference Latency (ms) & FPS
4. Memory Footprint (Model Parameters & Storage Size MB)

Evaluates 4 Realistic Deployment Profiles:
- Profile 1: "Real-Time Embedded Drone" (Prioritizes FPS >= 30, Low Latency, Small Footprint)
- Profile 2: "High-Altitude Precision Inspection" (Prioritizes Accuracy, Robustness, Small Object Detection)
- Profile 3: "Edge AI / Low-Power Compute" (Prioritizes Memory, Energy/Latency efficiency)
- Profile 4: "Balanced General UAV Platform" (Harmonic compromise between all dimensions)

Outputs:
- reports/deployment_tradeoff_matrix.csv
- reports/deployment_tradeoff_matrix.md
- reports/deployment_profile_rankings.csv
- reports/deployment_profile_rankings.md
- reports/pareto_frontier_accuracy_vs_latency.png
- reports/deployment_profile_radar_comparison.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"

# Verified metrics from earlier benchmark phases
EXPERIMENT_SPECS = {
    "A": {
        "model": "YOLO11n",
        "size": 640,
        "mAP50": 0.291,
        "mAP50_95": 0.164,
        "robustness_retention": 87.7,  # %
        "latency_ms": 9.82,
        "fps": 101.9,
        "params_m": 2.58,
        "size_mb": 5.4
    },
    "B1": {
        "model": "YOLO11s",
        "size": 960,
        "mAP50": 0.473,
        "mAP50_95": 0.285,
        "robustness_retention": 83.5,
        "latency_ms": 19.19,
        "fps": 52.1,
        "params_m": 9.42,
        "size_mb": 18.4
    },
    "B2": {
        "model": "YOLO11s",
        "size": 1280,
        "mAP50": 0.533,
        "mAP50_95": 0.329,
        "robustness_retention": 80.7,
        "latency_ms": 34.05,
        "fps": 29.4,
        "params_m": 9.42,
        "size_mb": 18.4
    },
    "B3": {
        "model": "YOLO11m",
        "size": 960,
        "mAP50": 0.531,
        "mAP50_95": 0.326,
        "robustness_retention": 82.7,
        "latency_ms": 45.78,
        "fps": 21.8,
        "params_m": 20.04,
        "size_mb": 39.5
    }
}

DEPLOYMENT_PROFILES = {
    "Real-Time Embedded Drone": {
        "desc": "High FPS (>=30 FPS), strict latency constraint on onboard chip",
        "weights": {"accuracy": 0.25, "robustness": 0.20, "fps": 0.35, "memory": 0.20}
    },
    "High-Altitude Precision Inspection": {
        "desc": "Offline / high-res inspection where accuracy & robustness outweigh speed",
        "weights": {"accuracy": 0.45, "robustness": 0.30, "fps": 0.15, "memory": 0.10}
    },
    "Edge AI / Low-Power Compute": {
        "desc": "Extremely resource constrained edge hardware (Jetson Nano / OAK-D)",
        "weights": {"accuracy": 0.20, "robustness": 0.20, "fps": 0.20, "memory": 0.40}
    },
    "Balanced General UAV": {
        "desc": "Harmonic balance across all operational dimensions",
        "weights": {"accuracy": 0.30, "robustness": 0.25, "fps": 0.30, "memory": 0.15}
    }
}


def normalize_min_max(val, min_v, max_v, higher_is_better=True):
    if max_v == min_v:
        return 1.0
    if higher_is_better:
        return (val - min_v) / (max_v - min_v)
    else:
        return (max_v - val) / (max_v - min_v)


def run_deployment_analysis():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = pd.DataFrame.from_dict(EXPERIMENT_SPECS, orient="index")
    df_raw.index.name = "Experiment"
    df_raw.reset_index(inplace=True)

    # 1. Compute Min-Max bounds for normalization
    map_min, map_max = df_raw["mAP50"].min(), df_raw["mAP50"].max()
    rob_min, rob_max = df_raw["robustness_retention"].min(), df_raw["robustness_retention"].max()
    fps_min, fps_max = df_raw["fps"].min(), df_raw["fps"].max()
    mem_min, mem_max = df_raw["size_mb"].min(), df_raw["size_mb"].max()

    # 2. Compute Utility Scores (0 to 1) for each dimension
    df_norm = pd.DataFrame()
    df_norm["Experiment"] = df_raw["Experiment"]
    df_norm["Model"] = df_raw["model"] + "@" + df_raw["size"].astype(str)
    df_norm["Accuracy_Score"] = df_raw["mAP50"].apply(lambda v: normalize_min_max(v, map_min, map_max, True))
    df_norm["Robustness_Score"] = df_raw["robustness_retention"].apply(lambda v: normalize_min_max(v, rob_min, rob_max, True))
    df_norm["FPS_Score"] = df_raw["fps"].apply(lambda v: normalize_min_max(v, fps_min, fps_max, True))
    df_norm["Memory_Score"] = df_raw["size_mb"].apply(lambda v: normalize_min_max(v, mem_min, mem_max, False))

    # 3. Multi-Criteria Scoring across Profiles
    profile_rankings = []

    for profile_name, p_info in DEPLOYMENT_PROFILES.items():
        w = p_info["weights"]
        scores = (
            df_norm["Accuracy_Score"] * w["accuracy"] +
            df_norm["Robustness_Score"] * w["robustness"] +
            df_norm["FPS_Score"] * w["fps"] +
            df_norm["Memory_Score"] * w["memory"]
        ) * 100.0

        for idx, exp in enumerate(df_norm["Experiment"]):
            profile_rankings.append({
                "Deployment_Profile": profile_name,
                "Experiment": exp,
                "Model_Config": df_norm.loc[idx, "Model"],
                "Composite_Score": round(scores[idx], 1),
                "mAP50": df_raw.loc[idx, "mAP50"],
                "FPS": df_raw.loc[idx, "fps"],
                "Latency_ms": df_raw.loc[idx, "latency_ms"],
                "Size_MB": df_raw.loc[idx, "size_mb"]
            })

    df_rank = pd.DataFrame(profile_rankings)
    df_rank.sort_values(by=["Deployment_Profile", "Composite_Score"], ascending=[True, False], inplace=True)
    df_rank["Rank"] = df_rank.groupby("Deployment_Profile")["Composite_Score"].rank(ascending=False, method="min").astype(int)

    # Save to CSV and MD
    df_raw.to_csv(REPORTS_DIR / "deployment_tradeoff_matrix.csv", index=False)
    df_rank.to_csv(REPORTS_DIR / "deployment_profile_rankings.csv", index=False)

    pivot_rank = df_rank.pivot(index="Deployment_Profile", columns="Experiment", values="Composite_Score")
    
    print("\n" + "=" * 90)
    print("                 DEPLOYMENT PROFILE SELECTION MATRIX (Score: 0-100)")
    print("=" * 90)
    print(pivot_rank.to_string())
    print("=" * 90 + "\n")

    print("=" * 90)
    print("                     RECOMMENDED WINNER PER DEPLOYMENT PROFILE")
    print("=" * 90)
    for p_name in DEPLOYMENT_PROFILES.keys():
        winner = df_rank[(df_rank["Deployment_Profile"] == p_name) & (df_rank["Rank"] == 1)].iloc[0]
        print(f"  * {p_name:36s} -> WINNER: Exp {winner['Experiment']} ({winner['Model_Config']}) [Score: {winner['Composite_Score']:.1f}/100]")
    print("=" * 90 + "\n")

    # Export to markdown
    md_text = "# Step 6F — Deployment Trade-off & Model Selection Analysis\n\n"
    md_text += "### 1. Raw Engineering Benchmark Matrix\n\n" + df_raw.to_markdown(index=False) + "\n\n"
    md_text += "### 2. Composite Decision Scores across Deployment Profiles\n\n" + pivot_rank.to_markdown() + "\n\n"
    md_text += "### 3. Detailed Profile Rankings\n\n" + df_rank.to_markdown(index=False) + "\n\n"
    (REPORTS_DIR / "deployment_tradeoff_matrix.md").write_text(md_text, encoding="utf-8")

    # Visualizations
    generate_deployment_charts(df_raw, df_norm)


def generate_deployment_charts(df_raw: pd.DataFrame, df_norm: pd.DataFrame):
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Pareto Frontier: mAP50 vs Inference Latency
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(
        data=df_raw,
        x="latency_ms",
        y="mAP50",
        hue="Experiment",
        size="size_mb",
        sizes=(150, 450),
        palette="tab10"
    )
    # Annotate points
    for _, row in df_raw.iterrows():
        plt.text(
            row["latency_ms"] + 0.8,
            row["mAP50"] - 0.008,
            f"{row['Experiment']} ({row['model']}@{row['size']})\n{row['fps']:.1f} FPS",
            fontsize=10,
            fontweight="bold"
        )

    # Draw real-time threshold boundary (30 FPS -> 33.3ms)
    plt.axvline(x=33.33, color="red", linestyle="--", alpha=0.7, label="Real-Time Boundary (30 FPS / 33.3ms)")
    plt.title("Pareto Efficiency Frontier: Detection Accuracy vs Latency", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Pure Inference Latency (ms) — Lower is Better", fontsize=12)
    plt.ylabel("mAP50 Accuracy — Higher is Better", fontsize=12)
    plt.xlim(5, 55)
    plt.ylim(0.25, 0.58)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pareto_frontier_accuracy_vs_latency.png", dpi=300)
    plt.close()

    # 2. Multi-Dimensional Radar Chart
    labels = ["Accuracy (mAP)", "Robustness", "Throughput (FPS)", "Lightweight (1/Size)"]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax_rad = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], labels, size=11, fontweight="bold")
    ax_rad.set_rlabel_position(25)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
    plt.ylim(0, 1.1)

    colors = {"A": "#e74c3c", "B1": "#3498db", "B2": "#2ecc71", "B3": "#9b59b6"}

    for idx, row in df_norm.iterrows():
        exp = row["Experiment"]
        values = [row["Accuracy_Score"], row["Robustness_Score"], row["FPS_Score"], row["Memory_Score"]]
        values += values[:1]
        ax_rad.plot(angles, values, linewidth=2.5, linestyle='solid', label=f"Exp {exp} ({row['Model']})", color=colors.get(exp))
        ax_rad.fill(angles, values, alpha=0.12, color=colors.get(exp))

    plt.title("Multi-Dimensional Deployment Evaluation Profile", size=14, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "deployment_profile_radar_comparison.png", dpi=300)
    plt.close()

    print(f"[✓] Step 6F charts and reports successfully saved to {REPORTS_DIR}")


if __name__ == "__main__":
    run_deployment_analysis()
