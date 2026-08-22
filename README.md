# AeroEval: Real-Time UAV Vision & AI Evaluation Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Ultralytics-YOLOv11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Evaluation](https://img.shields.io/badge/Task-Object%20Detection%20%26%20Robustness-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AeroEval** là nền tảng đánh giá toàn diện mô hình thị giác máy tính trên dữ liệu Drone/UAV (VisDrone Dataset). Dự án tập trung vào: **Độ chính xác (Accuracy) → Đánh giá vật thể nhỏ (Small-object) → Độ bền bỉ môi trường (Robustness) → Đánh giá tài nguyên (Efficiency) → Lựa chọn mô hình triển khai (Model Selection)**.

---

## 📌 1. Kiến trúc hệ thống

```text
Drone Aerial Imagery / Video Streams
                │
                ▼
┌──────────────────────────────────────────┐
│      Object Detection & Tracking         │
│  YOLOv11 (n/s/m) / Multi-Scale Inference │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│                  AI EVALUATION ENGINE                        │
│                                                              │
│  1. Multi-Scale Accuracy (COCO Small / Med / Large AP)       │
│  2. Environmental Robustness (Blur, Noise, Light, Occlusion) │
│  3. Failure Taxonomy & Root Cause Analysis                   │
│  4. Real-time Inference Profiling (Latency, FPS, Memory)     │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│  Multi-Criteria Decision Analysis (MCDA) │
│   Deployment Profile Model Selection     │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│         FastAPI + Web Dashboard          │
└──────────────────────────────────────────┘
```

---

## 📊 2. Kết quả thực nghiệm chính (Benchmark Matrix)

| Experiment | Model Config | Resolution | Epochs | mAP50 | mAP50-95 | Recall | Precision | Latency (ms) | Throughput (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | YOLO11n | 640px | 50 | 0.291 | 0.164 | 0.322 | 0.414 | **9.82 ms** | **101.9 FPS** |
| **Exp B1** | YOLO11s | 960px | 100 | 0.473 | 0.285 | 0.472 | 0.590 | 19.19 ms | 52.1 FPS |
| **Exp B2** | YOLO11s | 1280px | 100 | **0.533** | **0.329** | **0.523** | **0.635** | 34.05 ms | 29.4 FPS |
| **Exp B3** | YOLO11m | 960px | 100 | 0.531 | 0.326 | 0.523 | 0.627 | 45.78 ms | 21.8 FPS |

### Phát hiện quan trọng:
1. **Small-Object Hypothesis Confirmed**: Tăng độ phân giải lên 1280px (Exp B2) giúp tăng vọt **+78.4% AP50 trên Small Objects** (< 32² px), kéo theo các lớp yếu như Xe đạp (**+484.5%**), Người đi bộ (**+102.6%**), Xe ba bánh (**+150.6%**).
2. **Failure Mode Breakdown**: **>90% lỗi bỏ sót (False Negatives)** của mô hình thị giác trên Drone xuất phát từ kích thước vật thể siêu nhỏ.
3. **Environmental Vulnerability**: Mô hình nhạy cảm nhất với **Motion Blur (-48.4% rớt điểm)** và **Gaussian Sensor Noise (-40.0%)**, nhưng duy trì độ chính xác cực tốt dưới điều kiện **Low-Light / Glare** (< 5% suy giảm).

---

## 🚀 3. Cấu trúc thư mục

```text
aeroeval/
├── configs/               # Cấu hình dataset và các thí nghiệm (YAML)
├── data/                  # Dữ liệu VisDrone (không commit lên git)
├── notebooks/             # Notebook phân tích dữ liệu khám phá (EDA)
├── src/                   # Mã nguồn cốt lõi (aeroeval framework)
│   └── aeroeval/
│       ├── metrics/       # Đánh giá mAP, size, calibration, taxonomy
│       ├── robustness/    # 8 toán tử biến dạng môi trường
│       ├── models/        # Model runner & registry
│       ├── reporting/     # Báo cáo tự động & model recommendation
│       └── pipeline/      # Pipeline đánh giá tập trung
├── scripts/               # Scripts thực thi từng bước (Data -> Train -> Eval)
├── tests/                 # Unit tests với pytest
├── evaluation/            # Các module đánh giá độc lập
├── deployment/            # Docker, ONNX / TensorRT deployment
└── reports/               # Bảng biểu CSV, Markdown và biểu đồ PNG
```

---

## 🛠️ 4. Hướng dẫn cài đặt & Thực thi nhanh

### 4.1. Khởi tạo môi trường
```bash
git clone https://github.com/<username>/aeroeval.git
cd aeroeval

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2. Tải và chuẩn bị dữ liệu VisDrone
```bash
# Tải dataset tự động
python scripts/download_dataset.py

# Phân tích thống kê dataset
python scripts/inspect_dataset.py

# Chuyển đổi nhãn sang chuẩn YOLO
python scripts/convert_visdrone.py
```

### 4.3. Chạy đánh giá toàn diện (Evaluation Pipeline)
```bash
# 1. Đánh giá chi tiết từng Class (Step 6A)
python scripts/evaluate_per_class.py

# 2. Đánh giá theo kích thước Small/Med/Large (Step 6B)
python scripts/evaluate_by_size.py

# 3. Đánh giá độ bền bỉ môi trường Robustness (Step 6C & 6D)
python scripts/evaluate_robustness.py
python scripts/analyze_degradation.py

# 4. Phân tích nguyên nhân lỗi Failure Taxonomy (Step 6E)
python scripts/analyze_failure_taxonomy.py

# 5. Phân tích tối ưu triển khai Model Selection (Step 6F)
python scripts/analyze_deployment_tradeoff.py
```

---

## 📜 License
Dự án được phát hành dưới giấy phép [MIT License](LICENSE).
