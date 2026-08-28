"""
Step 14: Real-Time Performance & Efficiency Benchmark Script.

Measures inference latency, FPS throughput, and system resource consumption across
UAV object detection models (PyTorch .pt and ONNX .onnx).

Usage:
    python scripts/benchmark.py \
        --model experiments/baseline_yolo11n/weights/best.pt \
        --imgsz 640 \
        --device 0 \
        --warmup 50 \
        --iterations 200 \
        --output-dir reports/benchmark

Outputs:
    - reports/benchmark/latency_breakdown.png
    - reports/benchmark/fps_over_time.png
    - reports/benchmark/resource_usage.png
    - reports/benchmark/benchmark_summary.json
    - reports/benchmark/benchmark_summary.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from aeroeval.metrics.efficiency import benchmark_model_efficiency


def plot_latency_breakdown(bench_results: List[Dict], output_path: Path):
    """Stacked horizontal bar chart of Preprocess, Inference, and Postprocess."""
    plt.figure(figsize=(10, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    names = [r["model_name"] for r in bench_results]
    prep = [r["preprocess_mean_ms"] for r in bench_results]
    inf = [r["inference_mean_ms"] for r in bench_results]
    post = [r["postprocess_mean_ms"] for r in bench_results]

    y_pos = np.arange(len(names))
    height = 0.55

    plt.barh(y_pos, prep, height, label="Preprocessing (ms)", color="#3b82f6")
    plt.barh(y_pos, inf, height, left=prep, label="Pure Inference (ms)", color="#10b981")
    plt.barh(y_pos, post, height, left=np.array(prep) + np.array(inf), label="Postprocessing / NMS (ms)", color="#f59e0b")

    for i in range(len(names)):
        total_e2e = bench_results[i]["e2e_latency_mean_ms"]
        fps = bench_results[i]["fps_e2e"]
        plt.text(total_e2e + 0.5, y_pos[i], f"{total_e2e:.1f} ms ({fps:.0f} FPS)", va="center", fontweight="bold", fontsize=10)

    plt.yticks(y_pos, names, fontweight="bold")
    plt.xlabel("Latency per Frame (ms)", fontsize=11, fontweight="bold")
    plt.title("Real-Time Latency Breakdown & End-to-End Throughput", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fps_over_time(time_series: Dict, model_name: str, output_path: Path):
    """Line chart showing FPS throughput stability over frame iterations."""
    plt.figure(figsize=(10, 4.5), dpi=300)
    sns.set_theme(style="whitegrid")

    e2e_lat = np.array(time_series["e2e_ms"])
    fps_series = 1000.0 / np.maximum(e2e_lat, 0.001)

    # Moving average window
    window = 10
    fps_smooth = pd.Series(fps_series).rolling(window=window, min_periods=1).mean()

    iterations = np.arange(1, len(fps_series) + 1)

    plt.plot(iterations, fps_series, alpha=0.35, color="#0284c7", label="Instantaneous FPS")
    plt.plot(iterations, fps_smooth, color="#0369a1", linewidth=2.2, label=f"Moving Avg (window={window})")

    mean_fps = np.mean(fps_series)
    plt.axhline(mean_fps, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mean: {mean_fps:.1f} FPS")

    plt.title(f"Throughput Stability Over Frame Stream ({model_name})", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Frame Iteration", fontsize=10, fontweight="bold")
    plt.ylabel("Throughput (Frames Per Second)", fontsize=10, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_resource_usage(time_series: Dict, model_name: str, output_path: Path):
    """Dual-axis plot showing CPU % and VRAM/RAM consumption over benchmark stream."""
    fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=300)
    sns.set_theme(style="whitegrid")

    iterations = np.arange(1, len(time_series["cpu_percent"]) + 1)

    color_cpu = "#ea580c"
    ax1.set_xlabel("Frame Iteration", fontsize=10, fontweight="bold")
    ax1.set_ylabel("CPU Usage (%)", color=color_cpu, fontsize=10, fontweight="bold")
    line1 = ax1.plot(iterations, time_series["cpu_percent"], color=color_cpu, linewidth=1.8, label="CPU %")
    ax1.tick_params(axis="y", labelcolor=color_cpu)

    ax2 = ax1.twinx()
    color_mem = "#7c3aed"
    vram_data = time_series.get("vram_mb", [])
    has_vram = any(v > 0 for v in vram_data)

    if has_vram:
        ax2.set_ylabel("VRAM Allocated (MB)", color=color_mem, fontsize=10, fontweight="bold")
        line2 = ax2.plot(iterations, vram_data, color=color_mem, linewidth=2.0, label="GPU VRAM (MB)")
    else:
        ax2.set_ylabel("Process RAM (MB)", color=color_mem, fontsize=10, fontweight="bold")
        line2 = ax2.plot(iterations, time_series["ram_mb"], color=color_mem, linewidth=2.0, label="Host RAM (MB)")
    ax2.tick_params(axis="y", labelcolor=color_mem)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    plt.title(f"System Resource Profile During Streaming Inference ({model_name})", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Benchmarking for AeroEval Models")
    parser.add_argument("--model", type=str, default="experiments/baseline_yolo11n/weights/best.pt", help="Model path")
    parser.add_argument("--models", nargs="+", type=str, default=None, help="Optional multiple model paths")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("--output-dir", type=str, default="reports/benchmark", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = args.models if args.models else [args.model]

    all_summaries = []
    first_time_series = None
    first_model_name = ""

    print("=" * 70)
    print("AEROEVAL REAL-TIME BENCHMARKING ENGINE")
    print(f"Device: {args.device} | Image Size: {args.imgsz} | Iterations: {args.iterations} | Warmup: {args.warmup}")
    print("=" * 70)

    for m_str in model_paths:
        m_path = ROOT_DIR / m_str if not Path(m_str).is_absolute() else Path(m_str)
        if not m_path.exists():
            print(f"[WARN] Model file not found: {m_path}, skipping...")
            continue

        print(f"\n--> Profiling: {m_path.name} (Format: {m_path.suffix.upper()})...")
        res = benchmark_model_efficiency(
            model_path=m_path,
            imgsz=args.imgsz,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations
        )

        ts = res.pop("time_series")
        if first_time_series is None:
            first_time_series = ts
            first_model_name = res["model_name"]

        all_summaries.append(res)

        print(f"    Inference Mean Latency: {res['inference_mean_ms']} ms (P95: {res['inference_p95_ms']} ms)")
        print(f"    E2E Mean Latency:       {res['e2e_latency_mean_ms']} ms (FPS: {res['fps_e2e']})")
        print(f"    Pure Model Throughput:  {res['fps_model']} FPS")
        print(f"    Resource Profile:       CPU {res['avg_cpu_percent']}%, Peak VRAM: {res['peak_vram_mb']} MB")

    if not all_summaries:
        print("[ERROR] No valid models were benchmarked.")
        return

    # Save summary JSON
    json_path = output_dir / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[SAVED] Benchmark summary JSON -> {json_path}")

    # Save summary CSV
    df = pd.DataFrame(all_summaries)
    csv_path = output_dir / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] Benchmark summary CSV  -> {csv_path}")

    # Generate visual plots
    bar_path = output_dir / "latency_breakdown.png"
    plot_latency_breakdown(all_summaries, bar_path)
    print(f"[SAVED] Latency breakdown plot -> {bar_path}")

    if first_time_series:
        fps_path = output_dir / "fps_over_time.png"
        plot_fps_over_time(first_time_series, first_model_name, fps_path)
        print(f"[SAVED] FPS stability plot     -> {fps_path}")

        res_path = output_dir / "resource_usage.png"
        plot_resource_usage(first_time_series, first_model_name, res_path)
        print(f"[SAVED] Resource usage plot    -> {res_path}")

    print("\n[SUCCESS] Stage 14 Benchmarking Completed Successfully!")


if __name__ == "__main__":
    main()
