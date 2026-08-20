# AeroEval — Real-Time UAV Vision & AI Evaluation Platform
## Project Roadmap

> **Goal:** Build a professional Computer Vision + AI Evaluation + Edge/Real-time deployment project that can strengthen applications for **AI Engineer Intern, Computer Vision Intern, AI Evaluation/AI Quality, Edge AI, UAV/Robotics** roles.

The project should not be a simple “train YOLO on VisDrone” demo. The final product should demonstrate:

**AI model → rigorous evaluation → robustness testing → efficiency benchmarking → deployment → automated reporting**

---

# 1. Final Project Vision

## Proposed name

**AeroEval — Real-Time UAV Vision & AI Evaluation Platform**

## Core architecture

```text
Drone Video / Images
        │
        ▼
┌─────────────────────┐
│ Object Detection    │
│ YOLO / RT-DETR      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Multi-Object Track  │
│ ByteTrack / BoT-SORT│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────┐
│         AI EVALUATION ENGINE        │
│                                     │
│ Accuracy                            │
│ Robustness                          │
│ Calibration                         │
│ Small-object performance            │
│ Latency / FPS                       │
│ Memory / Model size                 │
│ Distribution shift                  │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────┐
│ Model Recommendation│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ FastAPI + Dashboard │
└─────────────────────┘
```

---

# 2. Why This Project

The portfolio should ultimately contain three complementary projects:

| Project | Main capability |
|---|---|
| Legal Agentic RAG | NLP / LLM / Retrieval / Agentic AI |
| Fraud Detection | Business ML / Responsible AI / Deployment |
| AeroEval | Computer Vision / UAV / AI Evaluation / Edge AI |

The third project is intended to replace the highly academic PET-MRI project and broaden the types of AI roles the portfolio can target.

---

# 3. Target Job Categories

The completed project should support applications to:

- AI Engineer Intern
- Computer Vision Intern
- AI Evaluation / AI Quality roles
- ML Engineer Intern
- Edge AI Intern
- UAV / Robotics AI Intern
- AI Testing / Model Validation roles
- MLOps-oriented AI Intern roles

---

# 4. Dataset: VisDrone

## Primary dataset

Use **VisDrone** as the main dataset.

VisDrone is designed for drone-based computer vision and provides multiple tasks including:

- Object Detection (DET)
- Video Object Detection (VID)
- Single Object Tracking (SOT)
- Multi-Object Tracking (MOT)

### Official source

https://github.com/VisDrone/VisDrone-Dataset

## Important rule

Do **not** download every VisDrone dataset at once.

Start with:

> **VisDrone-DET train + val**

Only download VID/MOT after the detection pipeline is working reliably.

---

# 5. Phase 0 — Environment Setup

Recommended baseline:

- Python 3.11
- PyTorch
- Ultralytics
- OpenCV
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- FastAPI
- Uvicorn
- Pydantic
- pytest
- ONNX / ONNX Runtime
- TensorRT when NVIDIA hardware is available
- Docker

## Initial installation

```bash
python -m venv .venv
source .venv/bin/activate
```

Then:

```bash
pip install --upgrade pip

pip install ultralytics \
    opencv-python \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    fastapi \
    uvicorn \
    pydantic \
    pytest
```

Check the environment:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

---

# 6. Phase 1 — Repository Structure

Create the repository immediately.

```text
aeroeval/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
├── data/
├── notebooks/
├── src/
├── scripts/
├── tests/
├── evaluation/
├── deployment/
├── reports/
└── experiments/
```

## Rules

- Never commit the dataset to GitHub.
- Keep all experiments reproducible.
- Keep configuration files separate from code.
- Move reusable functionality into `src/`.
- Use notebooks for exploration, not as the main application.

---

# 7. Phase 2 — Download VisDrone DET

Download only:

- VisDrone-DET train
- VisDrone-DET val

Start with the official VisDrone dataset repository:

https://github.com/VisDrone/VisDrone-Dataset

The repository provides the official download information and annotations.

After extraction, aim for a structure similar to:

```text
data/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
│
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
```

---

# 8. Phase 3 — Dataset Inspection

Before training anything, create:

```text
scripts/inspect_dataset.py
```

The script should report:

```text
Images:
Train: XXXX
Val: XXXX

Objects:
pedestrian: XXXX
people: XXXX
bicycle: XXXX
car: XXXX
van: XXXX
truck: XXXX
tricycle: XXXX
awning-tricycle: XXXX
bus: XXXX
motor: XXXX
```

Also calculate:

- number of images
- number of objects
- class distribution
- bounding-box width
- bounding-box height
- bounding-box area
- image resolution distribution
- objects per image

Outputs:

```text
reports/dataset_statistics.csv
reports/class_distribution.png
reports/object_size_distribution.png
```

## Why

This establishes evidence about:

- class imbalance
- crowded scenes
- small objects
- challenging image conditions

Do not make claims before calculating these statistics.

---

# 9. Phase 4 — Convert VisDrone Annotations to YOLO

VisDrone annotations are not in native YOLO label format.

Create:

```text
scripts/convert_visdrone.py
```

Convert annotations into:

```text
class x_center y_center width height
```

with normalized coordinates in `[0,1]`.

Target structure:

```text
data/visdrone_yolo/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Create:

```text
configs/visdrone.yaml
```

with the ten detection classes.

---

# 10. Phase 5 — Annotation Sanity Check

Create:

```text
scripts/visualize_annotations.py
```

Randomly sample annotated images and render bounding boxes.

Check:

- class mapping
- coordinate conversion
- box boundaries
- invalid boxes
- image-label correspondence
- resize correctness

Save samples to:

```text
reports/annotation_samples/
```

Do not start model training until the annotation visualization looks correct.

---

# 11. Phase 6 — First Baseline

Start with one small pretrained detector.

Recommended initial direction:

> YOLO small/nano model

Example:

```bash
yolo detect train \
    model=yolo11n.pt \
    data=configs/visdrone.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16
```

Adjust `batch` to available GPU memory.

## Record

- training loss
- validation loss
- mAP50
- mAP50-95
- Precision
- Recall
- per-class AP

Do not optimize the system before establishing a reproducible baseline.

---

# 12. Phase 7 — Baseline Evaluation Report

Create:

```text
reports/baseline_report.md
```

Include:

```text
Model:
Dataset:
Image Size:
Epochs:
Batch:
Hardware:

mAP50:
mAP50-95:
Precision:
Recall:

Per-class AP:
...
```

Generate:

```text
reports/baseline/
├── confusion_matrix.png
├── pr_curve.png
├── f1_curve.png
├── per_class_ap.png
└── qualitative_results/
```

The baseline must be reproducible.

---

# 13. Phase 8 — Small-Object Evaluation

This is one of the most important evaluation components.

Group objects by size:

```text
Small
Medium
Large
```

Measure:

```text
mAP_small
mAP_medium
mAP_large
```

Expected analysis:

```text
Small object AP:
Medium object AP:
Large object AP:
```

The purpose is to identify performance degradation on small drone-view objects.

Do not invent a standardized threshold definition. Document the exact rule used by the project.

---

# 14. Phase 9 — Robustness Evaluation

Build a dedicated robustness subsystem.

Create:

```text
evaluation/
└── robustness/
    ├── blur.py
    ├── brightness.py
    ├── noise.py
    ├── compression.py
    ├── resize.py
    └── occlusion.py
```

Test controlled corruption conditions.

## A. Gaussian blur

Example levels:

```text
kernel = 3
kernel = 5
kernel = 7
```

## B. Motion blur

Simulate camera/drone motion.

## C. Brightness

Example:

```text
0.5x
0.7x
1.0x
1.3x
```

## D. Gaussian noise

Example:

```text
sigma = 5
sigma = 10
sigma = 20
```

## E. JPEG compression

Example:

```text
quality = 90
quality = 70
quality = 50
quality = 30
```

## F. Resolution degradation

Example:

```text
1280 → 640 → 320
```

## G. Occlusion

Example:

```text
5%
10%
20%
```

---

# 15. Phase 10 — Robustness Metrics

For every corruption condition, record:

- mAP
- precision
- recall
- per-class performance
- performance drop from clean data

A simple project-specific metric can be:

```text
Robustness Drop =
Clean mAP - Corrupted mAP
```

Example report format:

```text
Clean           42.3
Blur            39.1   Δ -3.2
Low-light       34.8   Δ -7.5
Compression     37.5   Δ -4.8
Occlusion       29.4   Δ -12.9
```

If a custom aggregate score is created, clearly label it as a project-specific metric, not a community-standard benchmark.

---

# 16. Phase 11 — Model Comparison

Only after the baseline is stable, introduce a second model.

Possible comparison:

- YOLO small/nano
- larger YOLO variant
- RT-DETR

The exact choice depends on available hardware.

Compare:

| Model | mAP50-95 | Latency | FPS | Params |
|---|---:|---:|---:|---:|
| Model A | measured | measured | measured | measured |
| Model B | measured | measured | measured | measured |

The goal is:

> Accuracy vs efficiency trade-off

Do not automatically select the highest-mAP model.

---

# 17. Phase 12 — Confidence / Calibration Analysis

Analyze confidence thresholds.

Example report:

| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.20 | measured | measured |
| 0.30 | measured | measured |
| 0.40 | measured | measured |
| 0.50 | measured | measured |

Analyze:

- confidence distribution
- correct vs incorrect confidence
- precision-recall trade-off
- threshold sensitivity

The objective is to understand:

> What confidence threshold is appropriate for the intended deployment profile?

---

# 18. Phase 13 — Error Taxonomy

Create automatic or semi-automatic error categorization:

```text
False Positive
False Negative
Small-object miss
Occlusion miss
Crowded-scene miss
Low-confidence detection
Class confusion
```

Output:

```text
reports/error_analysis/
```

Include example images.

Create a final table:

```text
Top failure modes:

1. ...
2. ...
3. ...
4. ...
5. ...
```

This is a major component of the AI Evaluation aspect.

---

# 19. Phase 14 — Video Object Detection

After image detection works, move to VisDrone VID.

Do not download VID until:

- DET training works
- evaluation works
- robustness evaluation works

VisDrone provides official VID train/val/test data.

Goal:

```text
Video
 ↓
Detector
 ↓
Per-frame results
 ↓
Real-time benchmark
```

---

# 20. Phase 15 — Multi-Object Tracking

Add a tracker such as:

- ByteTrack
- BoT-SORT

Pipeline:

```text
Frame
 ↓
Detector
 ↓
Tracker
 ↓
Track IDs
 ↓
Trajectories
```

Evaluate tracking using appropriate metrics such as:

- MOTA
- IDF1
- HOTA
- ID switches

Do not add tracking only for visual effect. Use it to demonstrate temporal perception.

---

# 21. Phase 16 — Real-Time Benchmarking

Measure separately:

## Model latency

```text
image → model → output
```

## End-to-end latency

```text
read frame
→ preprocessing
→ inference
→ tracking
→ rendering
→ output
```

Measure:

- model latency
- preprocessing time
- postprocessing time
- tracking time
- end-to-end latency
- FPS
- CPU usage
- GPU usage
- VRAM
- RAM
- model size

Avoid claiming “real-time” unless the actual measurement supports it.

---

# 22. Phase 17 — ONNX

Export the model:

```bash
yolo export \
    model=best.pt \
    format=onnx
```

Compare:

```text
PyTorch FP32
vs
ONNX FP32
```

Measure:

- accuracy
- latency
- FPS
- memory

Document any accuracy changes.

---

# 23. Phase 18 — FP16 / TensorRT

If an NVIDIA GPU is available, extend the benchmark:

```text
FP32
 ↓
FP16
 ↓
INT8
```

For every version record:

- mAP
- latency
- FPS
- memory
- model size

Example table structure:

| Engine | Precision | mAP | Latency | FPS |
|---|---|---:|---:|---:|
| PyTorch | FP32 | measured | measured | measured |
| ONNX | FP32 | measured | measured | measured |
| TensorRT | FP16 | measured | measured | measured |
| TensorRT | INT8 | measured | measured | measured |

Do not pre-fill results.

---

# 24. Phase 19 — AI Evaluation Engine

Now turn the individual experiments into a reusable framework.

Suggested structure:

```text
src/aeroeval/
├── metrics/
│   ├── detection.py
│   ├── tracking.py
│   ├── calibration.py
│   └── efficiency.py
│
├── robustness/
│   ├── blur.py
│   ├── noise.py
│   ├── low_light.py
│   ├── compression.py
│   └── occlusion.py
│
├── models/
│   ├── registry.py
│   └── runner.py
│
├── reporting/
│   ├── report.py
│   └── recommendation.py
│
└── pipeline/
    └── evaluate.py
```

A unified command could eventually look like:

```bash
aeroeval evaluate \
    --model models/yolo11n.pt \
    --dataset configs/visdrone.yaml \
    --robustness full \
    --benchmark
```

Expected output:

```text
reports/run_001/
├── summary.json
├── metrics.csv
├── robustness.csv
├── efficiency.csv
├── errors.csv
├── figures/
└── evaluation_report.html
```

This is the point at which the project becomes an actual **AI Evaluation Platform** rather than a model training project.

---

# 25. Phase 20 — Model Recommendation

Create a deployment profile.

Example:

```text
Deployment Profile:
Real-time UAV
```

Example weighting:

```text
accuracy  = 0.30
latency   = 0.30
robustness = 0.25
memory    = 0.15
```

Use measured results to rank models.

The output should answer:

> Which model is the best choice for this deployment requirement?

This is decision support, not just metric visualization.

---

# 26. Phase 21 — Evaluation API

Use FastAPI.

Suggested endpoints:

```text
POST /models/register
POST /evaluate
POST /robustness
POST /benchmark
GET  /results/{run_id}
GET  /models
```

Example request:

```json
{
  "model": "yolo11n",
  "dataset": "visdrone",
  "profile": "real_time"
}
```

Example response:

```json
{
  "mAP50": "...",
  "mAP50_95": "...",
  "latency_ms": "...",
  "fps": "...",
  "robustness_score": "...",
  "recommendation": "..."
}
```

All final numbers must come from actual experiments.

---

# 27. Phase 22 — Dashboard

Use Streamlit.

## Page 1 — Overview

Display:

- best model
- mAP
- FPS
- latency
- robustness
- model size

## Page 2 — Model Comparison

Compare multiple models.

## Page 3 — Robustness

Show:

- blur
- brightness
- noise
- compression
- occlusion

## Page 4 — Error Analysis

Display real failure cases.

## Page 5 — Deployment

Compare:

- PyTorch
- ONNX
- TensorRT

---

# 28. Phase 23 — Docker

Create:

```text
Dockerfile
docker-compose.yml
```

Target architecture:

```text
FastAPI
   │
   ├── Evaluation Engine
   ├── Model Runner
   └── Results
        │
        ▼
    Streamlit
```

Run:

```bash
docker compose up --build
```

The README should explain:

- installation
- dataset setup
- training
- evaluation
- benchmarking
- API
- dashboard
- Docker

---

# 29. Phase 24 — Testing

Use pytest.

Suggested structure:

```text
tests/
├── test_metrics.py
├── test_robustness.py
├── test_dataset.py
├── test_api.py
└── test_model_registry.py
```

Examples:

```python
def test_iou():
    ...
```

```python
def test_robustness_drop():
    ...
```

```python
def test_api_evaluation():
    ...
```

Run:

```bash
pytest -q
```

Do not claim a test count until it exists.

---

# 30. Phase 25 — CI/CD

Add GitHub Actions:

```text
Push
 ↓
Install dependencies
 ↓
Lint
 ↓
Unit tests
 ↓
Build Docker
```

GPU training does not need to run in GitHub Actions.

The purpose is to show:

> reproducible engineering workflow

---

# 31. Phase 26 — Data Versioning / Reproducibility

Do not commit the dataset.

Create documentation:

```text
data/
README.md
```

Record:

```text
Dataset:
VisDrone2019 DET

Source:
Official VisDrone repository

Train:
...

Validation:
...

License:
...

Checksum:
...
```

Create a preparation script:

```text
scripts/prepare_dataset.py
```

or equivalent.

---

# 32. Phase 27 — Experiment Reproducibility

Store experiment configuration:

```text
configs/
├── baseline.yaml
├── robustness.yaml
├── benchmark.yaml
└── deployment.yaml
```

Record for each experiment:

```text
experiment ID
model
dataset
commit
seed
hardware
parameters
metrics
```

MLflow or Weights & Biases can be added later if useful, but are not required for the MVP.

---

# 33. Phase 28 — Final Report

Recommended report length:

> Approximately 15–25 pages.

Suggested structure:

```text
1. Problem Definition
2. Dataset
3. Baseline
4. Detection Performance
5. Small-object Analysis
6. Robustness Evaluation
7. Tracking
8. Model Comparison
9. Runtime Benchmark
10. ONNX/TensorRT Optimization
11. Evaluation Framework
12. API & Dashboard
13. Error Analysis
14. Limitations
15. Conclusion
```

The report should show measured evidence rather than marketing claims.

---

# 34. Required Figures

At minimum:

1. System architecture
2. Dataset examples
3. Ground truth vs prediction
4. Model comparison
5. Robustness curve
6. Latency vs mAP
7. FP32 vs FP16 vs INT8
8. Tracking visualization
9. Evaluation dashboard
10. Automatic evaluation report

---

# 35. Demo Video

Create a 60–90 second demo.

Suggested flow:

### 0–15 sec
Drone/UAV video

### 15–30 sec
Detection + tracking

### 30–45 sec
Robustness evaluation

### 45–60 sec
Model comparison

### 60–75 sec
Dashboard

### 75–90 sec
Deployment benchmark

Put the video/GIF in the GitHub README.

---

# 36. README Structure

Top of README:

```text
AeroEval
Real-Time UAV Vision & AI Evaluation Platform

A computer vision evaluation framework for drone imagery that
benchmarks detection, tracking, robustness and deployment efficiency.
```

Then:

```text
Features
- Object Detection
- Multi-object Tracking
- Robustness Testing
- Small-object Evaluation
- Runtime Benchmarking
- ONNX/TensorRT Optimization
- FastAPI
- Streamlit Dashboard
- Automated Evaluation Reports
```

Then:

- architecture diagram
- quick start
- dataset setup
- training
- evaluation
- benchmarking
- API
- dashboard
- results
- limitations

---

# 37. Evaluation Integrity

These rules are mandatory.

## Never

- use test data for model selection
- tune on the final benchmark
- report fabricated metrics
- call a custom metric a standard benchmark
- claim “real-time” without measuring it
- report only the best-case result

## Recommended split

```text
Training
   ↓
Training / model development

Validation
   ↓
Model selection / threshold tuning

Test-dev / appropriate final test set
   ↓
Final benchmark
```

The official VisDrone repository provides separate datasets and annotations. Follow the official split definitions carefully.

---

# 38. Expansion Plan

Do not download all data at the beginning.

## Milestone A — MVP

Use DET only.

Complete:

- dataset pipeline
- training
- evaluation
- robustness
- error analysis
- API
- dashboard

## Milestone B — Advanced

Add VID.

Complete:

- video inference
- tracking
- temporal analysis
- FPS benchmark

## Milestone C — Professional

Add MOT.

Complete:

- IDF1
- HOTA
- MOTA
- ID switches
- trajectories

VisDrone provides separate VID and MOT data, which are significantly larger than the initial DET subset, so they should be added only after the core project is stable.

---

# 39. Recommended 8–10 Week Roadmap

## Week 1

- Environment
- Repository
- Dataset download
- Dataset inspection
- Annotation conversion

## Week 2

- YOLO baseline
- Metrics
- Visualization
- Error analysis

## Week 3

- Small-object evaluation
- Robustness pipeline

## Week 4

- Model comparison
- Calibration
- Threshold analysis

## Week 5

- VID
- Tracking
- FPS benchmark

## Week 6

- ONNX
- FP16
- TensorRT if available

## Week 7

- Evaluation engine
- Automated reports
- FastAPI

## Week 8

- Streamlit
- Docker
- Pytest
- GitHub Actions

## Week 9

- Final experiments
- Report
- Demo video

## Week 10

- README
- Portfolio
- CV update
- Interview preparation

If the CV submission deadline is earlier, stop after **Milestone A** and upgrade later.

---

# 40. Minimum Requirements Before Putting the Project on the CV

The project should have, at minimum:

## Detection

- reproducible baseline
- quantitative metrics
- per-class analysis

## Evaluation

- robustness testing
- error taxonomy
- model comparison

## Engineering

- FastAPI
- dashboard
- tests
- Docker

## Deployment

- latency/FPS benchmark
- ONNX or TensorRT optimization

Only after reaching this level should the project become one of the three main projects on the CV.

---

# 41. Final CV Positioning

After completion, the portfolio should communicate:

### Legal Agentic RAG
Generative AI / NLP / Retrieval / Agentic AI

### Fraud Detection
Business ML / Responsible AI / Explainability / Deployment

### AeroEval
Computer Vision / UAV / AI Evaluation / Edge AI

The overall message should become:

> **I can build AI systems, evaluate them rigorously, and deploy them.**

This is more valuable for an AI Engineer Intern profile than simply showing a list of AI models.

---

# 42. First Immediate Action

Do **not** start by training YOLO.

The first task is:

1. Create the `aeroeval` repository.
2. Create the folder structure.
3. Download **VisDrone-DET train + val** from the official source.
4. Extract the files.
5. Create `scripts/inspect_dataset.py`.
6. Generate:
   - class distribution
   - object counts
   - image statistics
   - bbox statistics
   - object-size statistics.
7. Commit the initial repository structure.

Only after these checks are complete should the baseline model and training configuration be finalized.

---

# 43. Project Principle

The project should always follow:

```text
Data
  ↓
Understand
  ↓
Baseline
  ↓
Measure
  ↓
Find Failure Modes
  ↓
Improve
  ↓
Stress Test
  ↓
Benchmark
  ↓
Deploy
  ↓
Evaluate Again
```

The purpose is not to build the model with the highest number.

The purpose is to build a system that can answer:

> **How good is this AI model, when does it fail, how robust is it, how fast is it, and is it suitable for deployment?**
