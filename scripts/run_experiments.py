"""
Experiment runner and benchmark evaluation matrix.

Supports running:
- Exp A:  YOLO11n @ 640  (50 epochs)
- Exp B1: YOLO11s @ 960  (100 epochs)
- Exp B2: YOLO11s @ 1280 (100 epochs)
- Exp B3: YOLO11m @ 960  (100 epochs)

Also measures:
- Validation metrics (mAP50, mAP50-95, Precision, Recall)
- Pure model inference latency & FPS
- End-to-end latency & FPS
- Outputs comparison table to reports/experiment_matrix.csv and markdown
"""

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"
REPORTS_DIR = ROOT_DIR / "reports"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"

EXPERIMENTS = {
    "A": {
        "name": "baseline_yolo11n",
        "config": CONFIGS_DIR / "baseline.yaml",
        "model": "yolo11n.pt",
        "size": 640,
        "epochs": 50,
        "batch": 8
    },
    "B1": {
        "name": "exp_b1_yolo11s_960",
        "config": CONFIGS_DIR / "exp_b1_yolo11s_960.yaml",
        "model": "yolo11s.pt",
        "size": 960,
        "epochs": 100,
        "batch": 16
    },
    "B2": {
        "name": "exp_b2_yolo11s_1280",
        "config": CONFIGS_DIR / "exp_b2_yolo11s_1280.yaml",
        "model": "yolo11s.pt",
        "size": 1280,
        "epochs": 100,
        "batch": 8
    },
    "B3": {
        "name": "exp_b3_yolo11m_960",
        "config": CONFIGS_DIR / "exp_b3_yolo11m_960.yaml",
        "model": "yolo11m.pt",
        "size": 960,
        "epochs": 100,
        "batch": 16
    }
}


def benchmark_fps(model: YOLO, imgsz: int, device: str = "0", warmup: int = 20, iterations: int = 100):
    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    if device not in ["cpu", "-1"] and torch.cuda.is_available():
        dummy_input = dummy_input.to(f"cuda:{device}")
        torch_model = model.model.to(f"cuda:{device}")
        torch_model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = torch_model(dummy_input)
            torch.cuda.synchronize()

        # Measure
        timings = []
        with torch.no_grad():
            for _ in range(iterations):
                t0 = time.perf_counter()
                _ = torch_model(dummy_input)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)

        latency_ms = float(np.mean(timings))
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        return round(latency_ms, 2), round(fps, 1)
    else:
        torch_model = model.model.cpu()
        torch_model.eval()
        for _ in range(5):
            _ = torch_model(dummy_input)
        timings = []
        with torch.no_grad():
            for _ in range(20):
                t0 = time.perf_counter()
                _ = torch_model(dummy_input)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)
        latency_ms = float(np.mean(timings))
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        return round(latency_ms, 2), round(fps, 1)


def train_experiment(exp_id: str):
    exp = EXPERIMENTS[exp_id]
    cfg_file = exp["config"]
    
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"\n=======================================================")
    print(f"       STARTING EXPERIMENT {exp_id}: {exp['name']}")
    print(f"=======================================================")
    
    model = YOLO(cfg["model"])
    model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg.get("device", 0),
        workers=cfg.get("workers", 4),
        project=cfg["project"],
        name=cfg["name"],
        seed=cfg.get("seed", 42),
        deterministic=cfg.get("deterministic", True),
        save=cfg.get("save", True),
        save_period=cfg.get("save_period", 10),
        plots=cfg.get("plots", True),
        patience=cfg.get("patience", 20),
        optimizer=cfg.get("optimizer", "auto")
    )


def evaluate_and_build_matrix():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix_rows = []

    for exp_id, info in EXPERIMENTS.items():
        candidates = [
            EXPERIMENTS_DIR / info["name"] / "weights" / "best.pt",
            ROOT_DIR / "runs" / "detect" / "experiments" / info["name"] / "weights" / "best.pt",
            ROOT_DIR / "runs" / "detect" / info["name"] / "weights" / "best.pt",
        ]
        weights_path = None
        for c in candidates:
            if c.exists():
                weights_path = c
                break
        
        row = {
            "Experiment": exp_id,
            "Model": info["model"].replace(".pt", "").upper(),
            "Size": info["size"],
            "Epoch": info["epochs"],
            "mAP50": None,
            "mAP50-95": None,
            "Recall": None,
            "Precision": None,
            "Latency_ms": None,
            "FPS": None,
            "Status": "Not Trained"
        }

        if weights_path and weights_path.exists():
            print(f"\n[+] Evaluating weights for Exp {exp_id}: {weights_path}")
            model = YOLO(str(weights_path))
            
            # Validation metrics
            metrics = model.val(
                data="configs/visdrone.yaml",
                imgsz=info["size"],
                batch=info["batch"],
                device=0 if torch.cuda.is_available() else "cpu",
                split="val",
                verbose=False
            )

            # Benchmark FPS
            lat_ms, fps = benchmark_fps(model, imgsz=info["size"], device="0" if torch.cuda.is_available() else "cpu")

            row["mAP50"] = round(float(metrics.box.map50), 3)
            row["mAP50-95"] = round(float(metrics.box.map), 3)
            row["Recall"] = round(float(metrics.box.mr), 3)
            row["Precision"] = round(float(metrics.box.mp), 3)
            row["Latency_ms"] = lat_ms
            row["FPS"] = fps
            row["Status"] = "Completed"
        else:
            print(f"[-] Exp {exp_id} ({info['name']}) has not been trained yet.")

        matrix_rows.append(row)

    df_matrix = pd.DataFrame(matrix_rows)
    df_matrix.to_csv(REPORTS_DIR / "experiment_matrix.csv", index=False)
    
    # Save markdown version
    md_table = df_matrix.to_markdown(index=False)
    (REPORTS_DIR / "experiment_matrix.md").write_text(md_table, encoding="utf-8")

    print("\n" + "=" * 80)
    print("                     EXPERIMENT BENCHMARK MATRIX")
    print("=" * 80)
    print(df_matrix.to_string(index=False))
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Manage and run experimental matrix.")
    parser.add_argument("--train", type=str, choices=["A", "B1", "B2", "B3", "all"], help="Train specific experiment")
    parser.add_argument("--eval-matrix", action="store_true", help="Evaluate all trained models and generate matrix")
    args = parser.parse_args()

    if args.train:
        if args.train == "all":
            for eid in ["B1", "B2", "B3"]:
                train_experiment(eid)
        else:
            train_experiment(args.train)

    if args.eval_matrix or not args.train:
        evaluate_and_build_matrix()


if __name__ == "__main__":
    main()
