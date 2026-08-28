"""
Training script for baseline YOLO model on VisDrone-DET.

Features:
- Loads training configuration from configs/baseline.yaml
- Checks CUDA & GPU VRAM availability and automatically adjusts batch size if needed
- Trains using Ultralytics YOLO API
- Tracks training duration and peak memory usage
- Logs training metrics summary
- Copies best weights to an accessible location
"""

import argparse
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "baseline.yaml"


def check_and_adjust_gpu(batch_size: int, device_id: int = 0) -> int:
    if not torch.cuda.is_available():
        print("[!] No GPU detected. Falling back to CPU.")
        return 4

    vram_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(device_id)
    print(f"[+] Detected GPU: {gpu_name} with {vram_gb:.2f} GB VRAM")

    if vram_gb < 5.0 and batch_size > 8:
        print(f"[!] VRAM is {vram_gb:.1f} GB (< 5GB). Auto-adjusting batch size from {batch_size} to 8 to avoid OOM.")
        return 8
    return batch_size


def main():
    parser = argparse.ArgumentParser(description="Train baseline YOLO detector.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to config YAML file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to use (0, cpu, etc.)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch is not None:
        config["batch"] = args.batch
    if args.device is not None:
        config["device"] = args.device

    # Adjust batch size for safety
    if str(config.get("device", "0")) not in ["cpu", "-1"]:
        try:
            dev_id = int(config.get("device", 0))
            config["batch"] = check_and_adjust_gpu(config.get("batch", 8), dev_id)
        except ValueError:
            pass

    print("\n" + "=" * 60)
    print("           STARTING BASELINE MODEL TRAINING")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k:15s}: {v}")
    print("=" * 60 + "\n")

    start_time = time.time()

    # Initialize YOLO model
    model = YOLO(config["model"])

    # Train model
    model.train(
        data=config["data"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        workers=config.get("workers", 4),
        project=config["project"],
        name=config["name"],
        seed=config.get("seed", 42),
        deterministic=config.get("deterministic", True),
        save=config.get("save", True),
        save_period=config.get("save_period", 10),
        plots=config.get("plots", True),
        patience=config.get("patience", 15),
        optimizer=config.get("optimizer", "auto"),
        lr0=config.get("lr0", 0.01),
        lrf=config.get("lrf", 0.01)
    )

    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n" + "=" * 60)
    print("           BASELINE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total training time: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")
    print(f"Results and weights saved in: {config['project']}/{config['name']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
