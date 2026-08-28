# AeroEval: Real-Time UAV Vision & AI Evaluation Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Ultralytics-YOLOv11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![API](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED.svg)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AeroEval** is an open-source, production-grade benchmarking and evaluation platform designed specifically for Computer Vision and AI Perception models operating in Unmanned Aerial Vehicle (UAV) environments.

---

## 📌 System Architecture

```text
Aerial Drone Video / High-Resolution Stream
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│             OBJECT DETECTION & INFERENCE ENGINE              │
│       YOLOv11 (n / s / m)  •  PyTorch & ONNX Backends        │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                    AI EVALUATION ENGINE                      │
│                                                              │
│  1. Multi-Scale Accuracy (COCO Small / Medium / Large AP)    │
│  2. Environmental Robustness (Blur, Noise, Weather, Cutout)  │
│  3. Real-Time Hardware Profiling (Latency, FPS, VRAM, CPU)   │
│  4. Failure Taxonomy & Root Cause Analysis (Error Breakdown) │
│  5. Confidence Calibration & Reliability (ECE Analysis)      │
│  6. Multi-Object Tracking Evaluation (MOTA, IDF1, IDSW)      │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│        MULTI-CRITERIA DECISION ANALYSIS (MCDA)               │
│  Mission Profiles: real_time_uav | high_accuracy | edge_dev  │
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│     FastAPI REST Service     │ │  Streamlit Analytics App     │
│   (Models, Evaluate, Report) │ │ (Overview, Compare, Deploy)  │
│      http://localhost:8000   │ │    http://localhost:8501     │
└──────────────────────────────┘ └──────────────────────────────┘
```

---

## 📊 Benchmark Summary Matrix

| Experiment | Model Config | Resolution | mAP50 | mAP50-95 | Recall | Precision | E2E Latency | FPS | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | YOLO11n (Baseline) | 640px | 0.374 | 0.221 | 0.358 | 0.442 | **13.4 ms** | **74.6 FPS** | 46.7 MB |
| **Exp B1** | YOLO11s | 960px | 0.431 | 0.268 | 0.412 | 0.485 | 23.8 ms | 42.0 FPS | 82.3 MB |
| **Exp B2** | YOLO11s (High-Res) | 1280px | **0.468** | 0.295 | **0.448** | **0.518** | 38.5 ms | 26.0 FPS | 134.1 MB |
| **Exp B3** | YOLO11m (Capacity) | 960px | **0.472** | **0.301** | 0.445 | 0.514 | 42.1 ms | 23.8 FPS | 188.5 MB |

### Key Empirical Findings
1. **Small-Object Spatial Hypothesis**: Increasing input resolution from 640px to 1280px yields a **+68.4% relative gain in small-object Recall**, overcoming severe information loss for objects $< 32^2\text{ px}$.
2. **Failure Taxonomy Breakdown**: **>50% of model detection failures** are small-target misses, while class confusion is concentrated in visually adjacent pairs (`pedestrian` $\leftrightarrow$ `people`, `car` $\leftrightarrow$ `van`).
3. **Environmental Vulnerability**: Models degrade most severely under **Motion Blur (-48.4% drop)** and **Sensor Noise (-40.0% drop)**, while maintaining strong stability under **Low-Light / Glare conditions**.
4. **PyTorch vs ONNX Optimization**: Exported ONNX models achieve **96.8% detection concordance** and match PyTorch FP32 speeds with a standardized format for edge hardware.

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
git clone https://github.com/EddyTryToCode/aeroeval.git
cd aeroeval

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

### 2. Dataset Preparation
```bash
# One-command verification and setup
python scripts/prepare_dataset.py
```

### 3. Unified CLI Usage
```bash
# Full multi-modal model evaluation
python -m aeroeval evaluate --model experiments/baseline_yolo11n/weights/best.pt --dataset configs/visdrone.yaml

# Real-time hardware latency & FPS benchmark
python -m aeroeval benchmark --model experiments/baseline_yolo11n/weights/best.pt --imgsz 640 --device 0

# Multi-criteria deployment profile recommendation
python -m aeroeval recommend --profile real_time_uav --output reports
```

### 4. Running the Web Services

#### Option A: FastAPI REST API
```bash
uvicorn aeroeval.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation available at: **http://localhost:8000/docs**

#### Option B: Streamlit Analytics Dashboard
```bash
streamlit run src/aeroeval/dashboard/app.py
```
Open interactive dashboard at: **http://localhost:8501**

#### Option C: Docker Compose (All Services)
```bash
docker compose up --build
```

---

## 🧪 Testing

Execute the comprehensive test suite with pytest:
```bash
pytest -v
```
All 21+ unit and integration tests validate metric accuracy, tracking logic, corruption operators, ModelRegistry, and FastAPI routes.

---

## 📁 Repository Structure

```
aeroeval/
├── configs/               # YAML configuration files (dataset, baseline, robustness, benchmark, deployment)
├── data/                  # VisDrone data storage (git-ignored)
├── docker-compose.yml     # Multi-service containerization (API + Dashboard)
├── Dockerfile             # Production container definition
├── experiments/           # Training weights and experiment logs
├── pyproject.toml         # Package metadata and tool configs
├── reports/               # Evaluation artifacts, HTML reports, CSVs, and charts
│   ├── benchmark/         # Efficiency and ONNX benchmark comparisons
│   ├── final_report.md    # 15-section comprehensive technical evaluation report
│   └── run_baseline/      # Standalone HTML and JSON evaluation outputs
├── scripts/               # Step-by-step CLI runners and benchmarking utilities
├── src/
│   └── aeroeval/
│       ├── api/           # FastAPI service, routers, schemas, dependencies
│       ├── dashboard/     # Streamlit 5-page multi-modal dashboard app
│       ├── metrics/       # Detection, Small-Object, Tracking, Calibration, Error Taxonomy
│       ├── models/        # ModelRegistry and unified PyTorch/ONNX runner
│       ├── pipeline/      # EvaluationPipeline orchestrator and ExperimentLogger
│       ├── reporting/     # HTML report generator and RecommendationEngine
│       └── robustness/    # 8 environmental corruption simulation operators
└── tests/                 # Full Pytest unit and integration test suite
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
