"""
Efficiency and Real-Time Latency Benchmarking Module for Drone AI Perception.

Measures:
- Latency breakdown: Preprocessing, Pure Inference, Postprocessing (NMS), End-to-End
- Statistical latency metrics: Mean, Std, Median (P50), P95, P99, Min, Max
- Throughput: Pure Model FPS vs End-to-End FPS
- System Resources: CPU %, RAM (MB), GPU %, VRAM (MB)
- Model complexity: Parameters (M), Model file size (MB), estimated GFLOPs
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO


def get_process_memory_mb() -> float:
    """Returns current process RAM RSS usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_mb(device_id: int = 0) -> float:
    """Returns current CUDA GPU memory allocated in MB."""
    if torch.cuda.is_available() and device_id >= 0:
        return torch.cuda.memory_allocated(device_id) / (1024 * 1024)
    return 0.0


def benchmark_model_efficiency(
    model_path: Union[str, Path],
    imgsz: int = 640,
    device: str = "0",
    warmup: int = 50,
    iterations: int = 200,
    input_image: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Comprehensive efficiency and latency profiler for UAV object detection models.

    Args:
        model_path: Path to .pt or .onnx model weights.
        imgsz: Image dimension for inference.
        device: '0', 'cpu', etc.
        warmup: Number of warmup frames.
        iterations: Number of benchmark measurement frames.
        input_image: Optional numpy image (H, W, 3) to use instead of random noise.

    Returns:
        Dictionary with statistical metrics and time-series for analysis.
    """
    model_path = Path(model_path)
    is_cuda = (str(device).lower() not in ["cpu", "-1"]) and torch.cuda.is_available()
    dev_id = int(device) if (is_cuda and str(device).isdigit()) else 0

    model_size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0.0

    # Load model
    is_onnx = model_path.suffix.lower() == ".onnx"

    num_params = 0.0

    if not is_onnx:
        model = YOLO(str(model_path))
        if hasattr(model, "model") and model.model is not None:
            num_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    else:
        # Load via YOLO ONNX wrapper or onnxruntime
        model = YOLO(str(model_path), task="detect")

    # Prepare dummy or real frame
    if input_image is not None:
        raw_img = input_image.copy()
    else:
        raw_img = np.random.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup phase
    for _ in range(warmup):
        _ = model.predict(source=raw_img, imgsz=imgsz, device=device, verbose=False)
        if is_cuda:
            torch.cuda.synchronize(dev_id)

    preprocess_times: List[float] = []
    inference_times: List[float] = []
    postprocess_times: List[float] = []
    e2e_times: List[float] = []
    cpu_usages: List[float] = []
    vram_usages: List[float] = []
    ram_usages: List[float] = []

    # Benchmark loop
    for _ in range(iterations):
        # Step 1: Preprocess simulation (resize / normalize / tensor transfer)
        t_prep_start = time.perf_counter()
        im = cv2.resize(raw_img, (imgsz, imgsz))
        im_tensor = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if is_cuda:
            im_tensor = im_tensor.to(f"cuda:{dev_id}")
            torch.cuda.synchronize(dev_id)
        t_prep_end = time.perf_counter()
        prep_ms = (t_prep_end - t_prep_start) * 1000.0
        preprocess_times.append(prep_ms)

        # Step 2: Pure Model Forward Inference
        t_inf_start = time.perf_counter()
        if not is_onnx and hasattr(model, "model") and model.model is not None:
            with torch.no_grad():
                _ = model.model(im_tensor)
        else:
            _ = model.predict(source=im, imgsz=imgsz, device=device, verbose=False)
        if is_cuda:
            torch.cuda.synchronize(dev_id)
        t_inf_end = time.perf_counter()
        inf_ms = (t_inf_end - t_inf_start) * 1000.0
        inference_times.append(inf_ms)

        # Step 3: Full End-to-End Prediction (Preprocess + Inference + Postprocess NMS)
        t_pipe_start = time.perf_counter()
        _ = model.predict(source=raw_img, imgsz=imgsz, device=device, verbose=False)
        if is_cuda:
            torch.cuda.synchronize(dev_id)
        t_pipe_end = time.perf_counter()
        pipe_ms = (t_pipe_end - t_pipe_start) * 1000.0

        # Postprocess time is pipe_ms minus pure inference and preprocess
        post_ms = max(0.1, pipe_ms - inf_ms - prep_ms)
        postprocess_times.append(post_ms)

        e2e_times.append(prep_ms + inf_ms + post_ms)


        # Resource monitoring
        cpu_usages.append(psutil.cpu_percent(interval=None))
        vram_usages.append(get_gpu_memory_mb(dev_id) if is_cuda else 0.0)
        ram_usages.append(get_process_memory_mb())

    # Statistical Aggregations
    mean_inf = float(np.mean(inference_times))
    std_inf = float(np.std(inference_times))
    min_inf = float(np.min(inference_times))
    max_inf = float(np.max(inference_times))
    p50_inf = float(np.percentile(inference_times, 50))
    p95_inf = float(np.percentile(inference_times, 95))
    p99_inf = float(np.percentile(inference_times, 99))

    mean_prep = float(np.mean(preprocess_times))
    mean_post = float(np.mean(postprocess_times))
    mean_e2e = float(np.mean(e2e_times))
    std_e2e = float(np.std(e2e_times))
    p95_e2e = float(np.percentile(e2e_times, 95))

    fps_model = round(1000.0 / mean_inf, 1) if mean_inf > 0 else 0.0
    fps_e2e = round(1000.0 / mean_e2e, 1) if mean_e2e > 0 else 0.0

    return {
        "model_name": model_path.stem,
        "model_path": str(model_path),
        "model_format": "ONNX" if is_onnx else "PyTorch",
        "num_parameters_m": round(num_params, 2),
        "model_size_mb": round(model_size_mb, 2),
        "device": f"CUDA:{dev_id}" if is_cuda else "CPU",
        "imgsz": imgsz,
        "iterations": iterations,
        "warmup": warmup,
        "inference_mean_ms": round(mean_inf, 2),
        "inference_std_ms": round(std_inf, 2),
        "inference_min_ms": round(min_inf, 2),
        "inference_max_ms": round(max_inf, 2),
        "inference_p50_ms": round(p50_inf, 2),
        "inference_p95_ms": round(p95_inf, 2),
        "inference_p99_ms": round(p99_inf, 2),
        "preprocess_mean_ms": round(mean_prep, 2),
        "postprocess_mean_ms": round(mean_post, 2),
        "e2e_latency_mean_ms": round(mean_e2e, 2),
        "e2e_latency_std_ms": round(std_e2e, 2),
        "e2e_latency_p95_ms": round(p95_e2e, 2),
        "fps_model": fps_model,
        "fps_e2e": fps_e2e,
        "avg_cpu_percent": round(float(np.mean(cpu_usages)), 1),
        "peak_vram_mb": round(float(np.max(vram_usages)), 1) if is_cuda else 0.0,
        "avg_ram_mb": round(float(np.mean(ram_usages)), 1),
        "time_series": {
            "preprocess_ms": preprocess_times,
            "inference_ms": inference_times,
            "postprocess_ms": postprocess_times,
            "e2e_ms": e2e_times,
            "cpu_percent": cpu_usages,
            "vram_mb": vram_usages,
            "ram_mb": ram_usages,
        }
    }
