"""
Step 6D: Robustness Degradation Analysis & Environmental Vulnerability Profiling.

Computes:
- Clean -> Corruption absolute and relative mAP50 drop
- Robustness Degradation Index (RDI = mean % drop across perturbations)
- Worst-case perturbation rankings per model
- Environmental Sensitivity Matrix

Outputs:
- reports/robustness_degradation_analysis.csv
- reports/robustness_degradation_analysis.md
- reports/robustness_drop_delta_comparison.png
- reports/robustness_radar_profiles.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"


def run_degradation_analysis():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_rob_file = REPORTS_DIR / "robustness_benchmark_metrics.csv"
    if not raw_rob_file.exists():
        print(f"Error: {raw_rob_file} not found. Run scripts/evaluate_robustness.py first.")
        return

    df_rob = pd.read_csv(raw_rob_file)

    # 1. Calculate Average Drop & Retention per Corruption per Model
    summary = df_rob.groupby(["Experiment", "Model_Config", "Corruption"]).agg(
        Clean_mAP50=("Clean_mAP50", "mean"),
        Mean_Corrupted_mAP50=("Corrupted_mAP50", "mean"),
        Mean_mAP50_Drop=("mAP50_Drop", "mean"),
        Mean_Retention_Rate=("Retention_Rate_%", "mean")
    ).reset_index()

    summary["Relative_Drop_%"] = (100.0 - summary["Mean_Retention_Rate"]).round(1)
    summary["Mean_mAP50_Drop"] = summary["Mean_mAP50_Drop"].round(3)
    summary["Mean_Corrupted_mAP50"] = summary["Mean_Corrupted_mAP50"].round(3)
    summary["Clean_mAP50"] = summary["Clean_mAP50"].round(3)

    summary.to_csv(REPORTS_DIR / "robustness_degradation_analysis.csv", index=False)

    # Pivot Tables
    pivot_drop = summary.pivot(index="Corruption", columns="Experiment", values="Mean_mAP50_Drop").round(3)
    pivot_rel_drop = summary.pivot(index="Corruption", columns="Experiment", values="Relative_Drop_%").round(1)

    print("\n" + "=" * 85)
    print("           ABSOLUTE mAP50 PERFORMANCE DROP (Clean - Corrupted)")
    print("=" * 85)
    print(pivot_drop.to_string())

    print("\n" + "=" * 85)
    print("          RELATIVE PERFORMANCE DROP (%) FROM CLEAN BASELINE")
    print("=" * 85)
    print(pivot_rel_drop.to_string())
    print("=" * 85 + "\n")

    # Worst-case failure condition ranking
    print("=" * 85)
    print("       WORST-CASE ENVIRONMENTAL PERTURBATIONS BY SENSITIVITY")
    print("=" * 85)
    for exp in ["A", "B1", "B2", "B3"]:
        sub = summary[summary["Experiment"] == exp].sort_values(by="Relative_Drop_%", ascending=False)
        top_vuln = sub.iloc[0]["Corruption"]
        top_drop = sub.iloc[0]["Relative_Drop_%"]
        sec_vuln = sub.iloc[1]["Corruption"]
        sec_drop = sub.iloc[1]["Relative_Drop_%"]
        print(f"  Exp {exp:2s} ({sub.iloc[0]['Model_Config']:16s}): #1 {top_vuln} (-{top_drop}%), #2 {sec_vuln} (-{sec_drop}%)")
    print("=" * 85 + "\n")

    # Markdown Export
    md_text = "# Step 6D — Robustness Degradation Analysis\n\n"
    md_text += "### 1. Absolute mAP50 Drop by Perturbation\n\n" + pivot_drop.to_markdown() + "\n\n"
    md_text += "### 2. Relative Performance Degradation (% Loss from Clean)\n\n" + pivot_rel_drop.to_markdown() + "\n\n"
    (REPORTS_DIR / "robustness_degradation_analysis.md").write_text(md_text, encoding="utf-8")

    # Generate Visualizations
    generate_degradation_charts(summary, pivot_drop, pivot_rel_drop)


def generate_degradation_charts(summary: pd.DataFrame, pivot_drop: pd.DataFrame, pivot_rel_drop: pd.DataFrame):
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Grouped Bar Chart of Absolute mAP50 Drop
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(
        data=summary,
        x="Corruption",
        y="Mean_mAP50_Drop",
        hue="Model_Config",
        palette="rocket"
    )
    plt.title("Absolute mAP50 Degradation (Lower Drop is Better)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Environmental Perturbation", fontsize=12)
    plt.ylabel("mAP50 Drop (Clean - Corrupted)", fontsize=12)
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Model", loc="upper right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "robustness_drop_delta_comparison.png", dpi=300)
    plt.close()

    # 2. Radar Chart: Robustness Retention Profile across 8 corruptions
    categories = list(pivot_rel_drop.index)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, size=10, fontweight="bold")
    ax.set_rlabel_position(30)
    plt.yticks([10, 20, 30, 40, 50, 60], ["10%", "20%", "30%", "40%", "50%", "60%"], color="grey", size=9)
    plt.ylim(0, 65)

    colors = {"A": "#e74c3c", "B1": "#3498db", "B2": "#2ecc71", "B3": "#9b59b6"}
    for exp in ["A", "B1", "B2", "B3"]:
        values = pivot_rel_drop[exp].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Exp {exp}", color=colors.get(exp))
        ax.fill(angles, values, alpha=0.1, color=colors.get(exp))

    plt.title("Environmental Sensitivity Radar (% Performance Drop)", size=14, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "robustness_radar_profiles.png", dpi=300)
    plt.close()

    print(f"[✓] Step 6D charts and report successfully saved to {REPORTS_DIR}")


if __name__ == "__main__":
    run_degradation_analysis()
