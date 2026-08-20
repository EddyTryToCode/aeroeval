# AeroEval — Bản Kế Hoạch Hành Động Chi Tiết (Step-by-Step)

> Tài liệu này chia nhỏ toàn bộ dự án AeroEval thành từng bước hành động cụ thể,
> từ khởi tạo dự án đến khi hoàn thành và đưa lên CV.

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 0: KHỞI TẠO DỰ ÁN & MÔI TRƯỜNG
# ═══════════════════════════════════════════════

## Bước 0.1 — Tạo GitHub Repository

1. Truy cập https://github.com/new
2. Tên repo: `aeroeval`
3. Mô tả: `Real-Time UAV Vision & AI Evaluation Platform`
4. Chọn **Public**
5. Chọn **Add a README file**
6. Chọn License: **MIT**
7. Nhấn **Create repository**
8. Clone repo về máy:
   ```bash
   git clone https://github.com/<username>/aeroeval.git
   cd aeroeval
   ```

## Bước 0.2 — Tạo file .gitignore

Tạo file `.gitignore` với nội dung sau:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
*.egg
dist/
build/
*.whl

# Virtual environment
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data — KHÔNG BAO GIỜ commit dataset
data/VisDrone*/
data/visdrone_yolo/
data/*.zip
data/*.tar
data/*.tar.gz
data/*.rar

# Model weights
*.pt
*.pth
*.onnx
*.engine
*.trt
runs/
weights/

# Reports (giữ code tạo report, không commit ảnh nặng)
reports/**/*.png
reports/**/*.jpg
reports/**/*.jpeg
reports/**/*.gif
reports/annotation_samples/
reports/baseline/qualitative_results/
reports/error_analysis/

# Experiment outputs
experiments/*/
mlruns/
wandb/

# OS
.DS_Store
Thumbs.db

# Docker
*.log

# Environment variables
.env
.env.local
```

## Bước 0.3 — Tạo cấu trúc thư mục dự án

Chạy lần lượt:

```bash
mkdir -p configs
mkdir -p data
mkdir -p notebooks
mkdir -p src/aeroeval/metrics
mkdir -p src/aeroeval/robustness
mkdir -p src/aeroeval/models
mkdir -p src/aeroeval/reporting
mkdir -p src/aeroeval/pipeline
mkdir -p scripts
mkdir -p tests
mkdir -p evaluation/robustness
mkdir -p deployment
mkdir -p reports/baseline
mkdir -p reports/annotation_samples
mkdir -p reports/error_analysis
mkdir -p experiments
```

Tạo các file `__init__.py` cho Python package:

```bash
touch src/__init__.py
touch src/aeroeval/__init__.py
touch src/aeroeval/metrics/__init__.py
touch src/aeroeval/robustness/__init__.py
touch src/aeroeval/models/__init__.py
touch src/aeroeval/reporting/__init__.py
touch src/aeroeval/pipeline/__init__.py
```

Tạo các file `.gitkeep` cho thư mục trống:

```bash
touch data/.gitkeep
touch notebooks/.gitkeep
touch experiments/.gitkeep
touch deployment/.gitkeep
touch reports/baseline/.gitkeep
touch reports/annotation_samples/.gitkeep
touch reports/error_analysis/.gitkeep
```

## Bước 0.4 — Tạo Python Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## Bước 0.5 — Tạo requirements.txt

```text
# Core ML
torch>=2.0
torchvision>=0.15
ultralytics>=8.0

# Computer Vision
opencv-python>=4.8
opencv-python-headless>=4.8

# Data & Scientific
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3

# Visualization
matplotlib>=3.7
seaborn>=0.12

# API
fastapi>=0.100
uvicorn>=0.23
pydantic>=2.0

# Dashboard
streamlit>=1.28

# Testing
pytest>=7.4
pytest-cov>=4.1
httpx>=0.24

# Export & Optimization
onnx>=1.14
onnxruntime>=1.16

# Utilities
pyyaml>=6.0
tqdm>=4.65
Pillow>=10.0

# Linting
ruff>=0.1.0
```

## Bước 0.6 — Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Bước 0.7 — Kiểm tra môi trường

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics YOLO: OK')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
```

## Bước 0.8 — Tạo file cấu hình Ruff (linting)

Tạo `pyproject.toml`:

```toml
[project]
name = "aeroeval"
version = "0.1.0"
description = "Real-Time UAV Vision & AI Evaluation Platform"
requires-python = ">=3.11"

[tool.ruff]
target-version = "py311"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "W", "I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

## Bước 0.9 — Commit khởi tạo

```bash
git add .
git commit -m "chore: initial project structure with environment setup"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 1: TẢI VÀ CHUẨN BỊ DATASET
# ═══════════════════════════════════════════════

## Bước 1.1 — Tạo file hướng dẫn dataset

Tạo `data/README.md`:

```markdown
# Dataset: VisDrone2019-DET

## Source
Official VisDrone Dataset Repository:
https://github.com/VisDrone/VisDrone-Dataset

## Required Downloads (Phase 1 — MVP)
- VisDrone-DET train set
- VisDrone-DET val set

## Download Links
Tham khảo official repo để lấy link Google Drive / Baidu Pan mới nhất.

## After Download
Place files in this directory:
```
data/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
```

## License
VisDrone dataset is for academic/research use only.

## DO NOT commit dataset files to git.
```

## Bước 1.2 — Tạo script tải dataset

Tạo `scripts/download_dataset.py`:

```python
"""
Download VisDrone-DET train and val datasets.

Usage:
    python scripts/download_dataset.py

Notes:
    - VisDrone dataset is hosted on Google Drive / University server.
    - If automatic download fails, download manually from:
      https://github.com/VisDrone/VisDrone-Dataset
    - Place zip files in data/ then run this script with --extract-only
"""
```

Script nên thực hiện:
1. Kiểm tra xem data/ đã có dataset chưa
2. Tải VisDrone2019-DET-train.zip (~1.4GB)
3. Tải VisDrone2019-DET-val.zip (~0.15GB)
4. Giải nén vào data/
5. Xác nhận cấu trúc thư mục đúng
6. In số lượng files: images và annotations

Nếu link tải tự động không hoạt động (VisDrone hay thay đổi link):
- Tải thủ công từ official repo
- Đặt file zip vào `data/`
- Chạy: `python scripts/download_dataset.py --extract-only`

## Bước 1.3 — Giải nén và xác nhận cấu trúc

Sau khi tải xong, kiểm tra:

```bash
ls data/VisDrone2019-DET-train/images/ | wc -l    # Expected: ~6471 images
ls data/VisDrone2019-DET-train/annotations/ | wc -l # Expected: ~6471 annotation files
ls data/VisDrone2019-DET-val/images/ | wc -l       # Expected: ~548 images
ls data/VisDrone2019-DET-val/annotations/ | wc -l  # Expected: ~548 annotation files
```

## Bước 1.4 — Commit script và docs

```bash
git add scripts/download_dataset.py data/README.md
git commit -m "feat: add dataset download script and documentation"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 2: KHÁM PHÁ & PHÂN TÍCH DATASET
# ═══════════════════════════════════════════════

## Bước 2.1 — Tạo script inspect dataset

Tạo `scripts/inspect_dataset.py`

Script phải thực hiện:

### 2.1.1 — Đếm cơ bản
- Tổng số images trong train
- Tổng số images trong val
- Tổng số annotation files

### 2.1.2 — Phân tích VisDrone annotation format
VisDrone annotation format cho mỗi dòng:
```
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

Object categories:
```
0: ignored regions
1: pedestrian
2: people
3: bicycle
4: car
5: van
6: truck
7: tricycle
8: awning-tricycle
9: bus
10: motor
11: others
```

Lưu ý: category 0 (ignored regions) và 11 (others) sẽ bị loại khi train.

### 2.1.3 — Thống kê class distribution
- Đếm số objects cho mỗi class (1-10)
- Tính phần trăm mỗi class
- Xuất ra bảng và biểu đồ

### 2.1.4 — Thống kê bounding box
- Phân bố bbox width (pixel)
- Phân bố bbox height (pixel)
- Phân bố bbox area (pixel²)
- Tính min, max, mean, median, std cho mỗi metric
- Phân loại kích thước: Small (<32²), Medium (32²-96²), Large (>96²)

### 2.1.5 — Thống kê image
- Phân bố resolution (width x height)
- Số objects trung bình per image
- Min/max objects per image

### 2.1.6 — Outputs
Lưu kết quả vào:
```
reports/dataset_statistics.csv
reports/class_distribution.png
reports/object_size_distribution.png
reports/objects_per_image.png
reports/bbox_dimensions.png
reports/image_resolution_distribution.png
```

## Bước 2.2 — Chạy inspect script

```bash
python scripts/inspect_dataset.py
```

## Bước 2.3 — Tạo notebook khám phá (tùy chọn)

Tạo `notebooks/01_dataset_exploration.ipynb`

Dùng để trực quan hóa thêm:
- Hiển thị một số ảnh mẫu
- Vẽ heatmap vị trí objects
- Phân tích tương quan giữa kích thước ảnh và số objects
- So sánh train vs val distribution

## Bước 2.4 — Commit

```bash
git add scripts/inspect_dataset.py reports/dataset_statistics.csv
git commit -m "feat: dataset inspection script with statistics"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 3: CHUYỂN ĐỔI ANNOTATIONS
# ═══════════════════════════════════════════════

## Bước 3.1 — Tạo script chuyển đổi VisDrone → YOLO format

Tạo `scripts/convert_visdrone.py`

### Quy tắc chuyển đổi:

VisDrone format:
```
bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
```

YOLO format:
```
class_id x_center y_center width height
```
(tất cả normalized [0, 1])

### Mapping class:
```
VisDrone category → YOLO class_id
1 (pedestrian)     → 0
2 (people)         → 1
3 (bicycle)        → 2
4 (car)            → 3
5 (van)            → 4
6 (truck)          → 5
7 (tricycle)       → 6
8 (awning-tricycle)→ 7
9 (bus)            → 8
10 (motor)         → 9
```

### Logic chuyển đổi cho mỗi annotation line:
```python
x_center = (bbox_left + bbox_width / 2) / image_width
y_center = (bbox_top + bbox_height / 2) / image_height
w = bbox_width / image_width
h = bbox_height / image_height
```

### Lọc bỏ:
- category 0 (ignored regions)
- category 11 (others)
- bbox width <= 0 hoặc bbox height <= 0
- score == 0 (nếu muốn lọc thêm)

### Cấu trúc output:
```
data/visdrone_yolo/
├── images/
│   ├── train/   (symlink hoặc copy từ VisDrone2019-DET-train/images/)
│   └── val/     (symlink hoặc copy từ VisDrone2019-DET-val/images/)
└── labels/
    ├── train/   (YOLO format .txt files)
    └── val/     (YOLO format .txt files)
```

### Xử lý edge cases:
- Đọc kích thước ảnh thực tế bằng PIL/OpenCV (không giả định)
- Clamp coordinates vào [0, 1]
- Log cảnh báo cho boxes bất thường
- Đếm số boxes bị skip và lý do

## Bước 3.2 — Tạo file cấu hình YOLO dataset

Tạo `configs/visdrone.yaml`:

```yaml
path: ../data/visdrone_yolo
train: images/train
val: images/val

nc: 10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

## Bước 3.3 — Chạy chuyển đổi

```bash
python scripts/convert_visdrone.py
```

Kiểm tra output:
```bash
ls data/visdrone_yolo/images/train/ | wc -l
ls data/visdrone_yolo/labels/train/ | wc -l
ls data/visdrone_yolo/images/val/ | wc -l
ls data/visdrone_yolo/labels/val/ | wc -l
```

Số images phải bằng số label files.

## Bước 3.4 — Commit

```bash
git add scripts/convert_visdrone.py configs/visdrone.yaml
git commit -m "feat: VisDrone to YOLO annotation converter"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 4: KIỂM TRA ANNOTATIONS (SANITY CHECK)
# ═══════════════════════════════════════════════

## Bước 4.1 — Tạo script visualize annotations

Tạo `scripts/visualize_annotations.py`

Script phải:
1. Random chọn N images (mặc định N=20)
2. Đọc ảnh gốc
3. Đọc YOLO label tương ứng
4. Chuyển YOLO normalized coords → pixel coords
5. Vẽ bounding boxes lên ảnh với màu theo class
6. Ghi tên class bên cạnh mỗi box
7. Lưu ảnh annotated vào `reports/annotation_samples/`
8. In thống kê: số boxes per image, phân bố class trong sample

### Checklist kiểm tra bằng mắt:
- [ ] Boxes nằm đúng vị trí trên objects
- [ ] Class labels đúng (car là car, không phải truck)
- [ ] Không có boxes nằm ngoài ảnh
- [ ] Không có boxes kích thước bất thường (quá lớn/quá nhỏ vô lý)
- [ ] Mỗi ảnh đều có label file tương ứng
- [ ] Tên file ảnh khớp tên file label

## Bước 4.2 — Chạy visualization

```bash
python scripts/visualize_annotations.py --num-samples 20
```

## Bước 4.3 — Kiểm tra kết quả

Mở các ảnh trong `reports/annotation_samples/` và kiểm tra thủ công.

**QUAN TRỌNG**: KHÔNG bắt đầu train model cho đến khi annotation visualization trông chính xác.

## Bước 4.4 — Commit

```bash
git add scripts/visualize_annotations.py
git commit -m "feat: annotation visualization for sanity check"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 5: TRAIN BASELINE MODEL
# ═══════════════════════════════════════════════

## Bước 5.1 — Tạo training config

Tạo `configs/baseline.yaml`:

```yaml
model: yolo11n.pt
data: configs/visdrone.yaml
epochs: 50
imgsz: 640
batch: 16
device: 0
project: experiments
name: baseline_yolo11n
seed: 42
deterministic: true
save: true
save_period: 10
plots: true
patience: 10
```

## Bước 5.2 — Tạo training script

Tạo `scripts/train_baseline.py`

Script phải:
1. Load config từ `configs/baseline.yaml`
2. Kiểm tra GPU availability
3. Điều chỉnh batch size nếu cần (dựa trên VRAM)
4. Bắt đầu training
5. Log thời gian training
6. Sau training, in best metrics
7. Copy best.pt vào vị trí rõ ràng

### Xử lý memory:
- Nếu OOM: giảm batch từ 16 → 8 → 4
- Nếu không có GPU: `device: cpu` (sẽ rất chậm, khuyến nghị dùng Colab)

## Bước 5.3 — Chạy training

```bash
python scripts/train_baseline.py
```

Hoặc sử dụng CLI trực tiếp:
```bash
yolo detect train \
    model=yolo11n.pt \
    data=configs/visdrone.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    project=experiments \
    name=baseline_yolo11n \
    seed=42
```

## Bước 5.4 — Theo dõi training

Kiểm tra trong quá trình training:
- Loss có giảm đều không
- Không có NaN loss
- Validation metrics có cải thiện không

## Bước 5.5 — Ghi nhận kết quả

Sau training, thu thập từ `experiments/baseline_yolo11n/`:
- `results.csv` — metrics qua từng epoch
- `confusion_matrix.png`
- `PR_curve.png`
- `F1_curve.png`
- `results.png` — training curves
- `weights/best.pt` — best checkpoint
- `weights/last.pt` — last checkpoint
- `args.yaml` — training arguments

## Bước 5.6 — Commit

```bash
git add scripts/train_baseline.py configs/baseline.yaml
git commit -m "feat: baseline YOLO training pipeline"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 6: ĐÁNH GIÁ BASELINE
# ═══════════════════════════════════════════════

## Bước 6.1 — Chạy validation chính thức

```bash
yolo detect val \
    model=experiments/baseline_yolo11n/weights/best.pt \
    data=configs/visdrone.yaml \
    imgsz=640 \
    split=val
```

## Bước 6.2 — Thu thập metrics

Ghi nhận:
- mAP50
- mAP50-95
- Precision
- Recall
- Per-class AP (cho 10 classes)

## Bước 6.3 — Tạo baseline report

Tạo `reports/baseline_report.md`:

```markdown
# Baseline Evaluation Report

## Configuration
- Model: YOLOv11n (pretrained on COCO)
- Dataset: VisDrone2019-DET
- Image Size: 640
- Epochs: 50
- Batch Size: 16
- Hardware: [ghi rõ GPU model]
- Training Time: [ghi rõ]

## Overall Metrics
| Metric    | Value |
|-----------|-------|
| mAP50     | X.XX  |
| mAP50-95  | X.XX  |
| Precision | X.XX  |
| Recall    | X.XX  |

## Per-Class AP
| Class           | AP50  | AP50-95 |
|-----------------|-------|---------|
| pedestrian      | X.XX  | X.XX    |
| people          | X.XX  | X.XX    |
| bicycle         | X.XX  | X.XX    |
| car             | X.XX  | X.XX    |
| van             | X.XX  | X.XX    |
| truck           | X.XX  | X.XX    |
| tricycle        | X.XX  | X.XX    |
| awning-tricycle | X.XX  | X.XX    |
| bus             | X.XX  | X.XX    |
| motor           | X.XX  | X.XX    |

## Observations
- [Ghi nhận điểm mạnh]
- [Ghi nhận điểm yếu]
- [Class nào performance thấp nhất và giả thuyết tại sao]
```

## Bước 6.4 — Tạo script sinh qualitative results

Tạo `scripts/generate_qualitative.py`

Script phải:
1. Chạy inference trên 20-30 val images
2. Vẽ predictions lên ảnh
3. Hiển thị confidence scores
4. Lưu vào `reports/baseline/qualitative_results/`
5. Chọn mix: ảnh detect tốt, ảnh detect trung bình, ảnh detect kém

## Bước 6.5 — Commit

```bash
git add reports/baseline_report.md scripts/generate_qualitative.py
git commit -m "feat: baseline evaluation report and qualitative analysis"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 7: ĐÁNH GIÁ SMALL-OBJECT
# ═══════════════════════════════════════════════

## Bước 7.1 — Tạo module đánh giá theo kích thước

Tạo `src/aeroeval/metrics/detection.py`

### Định nghĩa kích thước (COCO-based):
```python
SIZE_THRESHOLDS = {
    "small": (0, 32**2),       # area < 1024 px²
    "medium": (32**2, 96**2),  # 1024 ≤ area < 9216 px²
    "large": (96**2, float("inf")),  # area ≥ 9216 px²
}
```

### Logic:
1. Load ground truth annotations
2. Load predictions (từ YOLO val output)
3. Phân loại mỗi GT box vào size category dựa trên area
4. Match predictions với GT boxes (IoU ≥ 0.5)
5. Tính AP riêng cho mỗi size category
6. Tính precision/recall riêng cho mỗi size category

### Output metrics:
```
mAP_small:  X.XX
mAP_medium: X.XX
mAP_large:  X.XX

Precision_small:  X.XX
Precision_medium: X.XX
Precision_large:  X.XX

Recall_small:  X.XX
Recall_medium: X.XX
Recall_large:  X.XX
```

## Bước 7.2 — Tạo script đánh giá

Tạo `scripts/evaluate_by_size.py`

## Bước 7.3 — Chạy đánh giá

```bash
python scripts/evaluate_by_size.py \
    --model experiments/baseline_yolo11n/weights/best.pt \
    --data configs/visdrone.yaml
```

## Bước 7.4 — Tạo visualization

Output:
```
reports/small_object_analysis.png    — bar chart so sánh mAP theo size
reports/size_distribution_vs_ap.png  — scatter plot
```

## Bước 7.5 — Ghi nhận phân tích

Thêm vào report:
- Small objects chiếm bao nhiêu % dataset
- Performance drop giữa small vs large
- Giả thuyết nguyên nhân (resolution, feature map size, etc.)

## Bước 7.6 — Commit

```bash
git add src/aeroeval/metrics/detection.py scripts/evaluate_by_size.py
git commit -m "feat: small-object performance evaluation"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 8: ĐÁNH GIÁ ROBUSTNESS
# ═══════════════════════════════════════════════

## Bước 8.1 — Tạo các corruption modules

### 8.1.1 — `evaluation/robustness/blur.py`
```python
# Gaussian blur: kernel = [3, 5, 7, 9, 11]
# Motion blur: kernel = [5, 10, 15], angle = random
```

### 8.1.2 — `evaluation/robustness/brightness.py`
```python
# Brightness scaling: [0.3, 0.5, 0.7, 1.0, 1.3, 1.5]
# Gamma correction: [0.5, 0.7, 1.0, 1.5, 2.0]
```

### 8.1.3 — `evaluation/robustness/noise.py`
```python
# Gaussian noise: sigma = [5, 10, 15, 20, 30]
# Salt & pepper noise: ratio = [0.01, 0.02, 0.05]
```

### 8.1.4 — `evaluation/robustness/compression.py`
```python
# JPEG quality: [90, 70, 50, 30, 10]
```

### 8.1.5 — `evaluation/robustness/resize.py`
```python
# Resolution downscale: original → 640 → 480 → 320 → 160
# Then upscale back to original (simulate low-quality capture)
```

### 8.1.6 — `evaluation/robustness/occlusion.py`
```python
# Random black patches: cover [5%, 10%, 15%, 20%, 30%] of image
# Positioned randomly
```

### 8.1.7 — `evaluation/robustness/__init__.py`
```python
# Registry pattern: name → corruption function
CORRUPTIONS = {
    "gaussian_blur": ...,
    "motion_blur": ...,
    "brightness_dark": ...,
    "brightness_bright": ...,
    "gaussian_noise": ...,
    "jpeg_compression": ...,
    "resolution_degrade": ...,
    "occlusion": ...,
}
```

## Bước 8.2 — Tạo robustness evaluation runner

Tạo `scripts/evaluate_robustness.py`

Logic:
1. Load model
2. For each corruption type:
   a. For each severity level:
      - Apply corruption to val images
      - Run inference
      - Calculate mAP, precision, recall
      - Record results
3. Output summary table

## Bước 8.3 — Chạy robustness evaluation

```bash
python scripts/evaluate_robustness.py \
    --model experiments/baseline_yolo11n/weights/best.pt \
    --data configs/visdrone.yaml \
    --output reports/robustness/
```

## Bước 8.4 — Tạo robustness report

Output:
```
reports/robustness/
├── robustness_summary.csv
├── robustness_heatmap.png        — corruption type × severity → mAP
├── robustness_drop_chart.png     — bar chart of Δ mAP
├── per_corruption_curves.png     — line chart: severity → mAP
└── worst_case_examples/          — ảnh minh họa failure cases
```

Format bảng:
```
Condition           | mAP50  | Δ from Clean
--------------------|--------|-------------
Clean               | 42.3   | 0.0
Gaussian blur k=3   | 40.1   | -2.2
Gaussian blur k=7   | 35.6   | -6.7
Motion blur k=10    | 37.2   | -5.1
Dark (0.5x)         | 34.8   | -7.5
Noise σ=10          | 39.5   | -2.8
Noise σ=20          | 33.1   | -9.2
JPEG q=50           | 38.7   | -3.6
JPEG q=30           | 35.2   | -7.1
Occlusion 10%       | 37.9   | -4.4
Occlusion 20%       | 29.4   | -12.9
Resolution 320      | 28.3   | -14.0
```

## Bước 8.5 — Commit

```bash
git add evaluation/robustness/ scripts/evaluate_robustness.py
git commit -m "feat: robustness evaluation pipeline with 7 corruption types"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 9: SO SÁNH MODELS
# ═══════════════════════════════════════════════

## Bước 9.1 — Chọn model thứ 2

Tùy hardware, chọn 1-2 models bổ sung:
- **Option A**: YOLOv11s (small) — nhiều params hơn nano
- **Option B**: YOLOv11m (medium) — nếu GPU đủ mạnh
- **Option C**: RT-DETR-l — transformer-based detector

## Bước 9.2 — Train model thứ 2

Tạo config tương tự baseline, chỉ đổi model name.

```bash
yolo detect train \
    model=yolo11s.pt \
    data=configs/visdrone.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    project=experiments \
    name=comparison_yolo11s \
    seed=42
```

## Bước 9.3 — Đánh giá model thứ 2

Chạy lại toàn bộ evaluation pipeline:
- Validation metrics
- Small-object evaluation
- Robustness evaluation (ít nhất clean + 3-4 corruptions chính)

## Bước 9.4 — Tạo comparison table

Tạo `scripts/compare_models.py`

Output bảng:
```
| Model    | mAP50 | mAP50-95 | Latency(ms) | FPS   | Params(M) | Size(MB) |
|----------|-------|----------|-------------|-------|-----------|----------|
| YOLOv11n | X.XX  | X.XX     | X.X         | XXX   | X.X       | X.X      |
| YOLOv11s | X.XX  | X.XX     | X.X         | XXX   | X.X       | X.X      |
```

## Bước 9.5 — Tạo visualization

```
reports/model_comparison/
├── accuracy_comparison.png
├── latency_vs_map.png           — scatter plot: key chart
├── per_class_comparison.png
├── robustness_comparison.png
└── efficiency_radar.png         — radar chart: mAP, FPS, robustness, size
```

## Bước 9.6 — Commit

```bash
git add scripts/compare_models.py configs/
git commit -m "feat: multi-model comparison framework"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 10: CONFIDENCE & CALIBRATION ANALYSIS
# ═══════════════════════════════════════════════

## Bước 10.1 — Tạo calibration module

Tạo `src/aeroeval/metrics/calibration.py`

### Phân tích:
1. **Confidence distribution**: histogram of prediction scores
2. **Correct vs incorrect confidence**: box plot comparison
3. **Threshold sweep**: precision & recall tại threshold 0.1 → 0.9 (step 0.05)
4. **Optimal threshold**: F1-maximizing threshold
5. **Reliability diagram**: expected accuracy vs confidence

## Bước 10.2 — Tạo script phân tích

Tạo `scripts/analyze_calibration.py`

Output:
```
reports/calibration/
├── confidence_distribution.png
├── threshold_sweep.png           — precision, recall, F1 vs threshold
├── optimal_threshold.json
├── reliability_diagram.png
└── confidence_correct_vs_incorrect.png
```

## Bước 10.3 — Commit

```bash
git add src/aeroeval/metrics/calibration.py scripts/analyze_calibration.py
git commit -m "feat: confidence calibration analysis"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 11: ERROR TAXONOMY
# ═══════════════════════════════════════════════

## Bước 11.1 — Tạo error analysis module

Tạo `src/aeroeval/metrics/error_analysis.py`

### Error categories:
```python
ERROR_TYPES = [
    "false_positive",           # Detection where no GT exists
    "false_negative",           # GT object not detected
    "small_object_miss",        # FN where GT area < 32²
    "occlusion_miss",           # FN where GT has occlusion flag
    "crowded_scene_miss",       # FN in images with >50 objects
    "low_confidence_detection", # TP but confidence < 0.3
    "class_confusion",          # Detection with wrong class
    "localization_error",       # IoU between 0.1-0.5 (detected but poorly localized)
    "duplicate_detection",      # Multiple detections for same GT
]
```

### Logic:
1. Match predictions với GT boxes (IoU threshold = 0.5)
2. Phân loại mỗi error vào categories
3. Đếm số lượng mỗi loại
4. Chọn example images cho mỗi loại error
5. Tạo summary table

## Bước 11.2 — Tạo script

Tạo `scripts/analyze_errors.py`

## Bước 11.3 — Output

```
reports/error_analysis/
├── error_summary.csv
├── error_distribution.png
├── top_failure_modes.md
├── examples/
│   ├── false_positive_01.png
│   ├── false_negative_01.png
│   ├── small_object_miss_01.png
│   ├── class_confusion_01.png
│   └── ...
```

## Bước 11.4 — Commit

```bash
git add src/aeroeval/metrics/error_analysis.py scripts/analyze_errors.py
git commit -m "feat: automated error taxonomy and failure mode analysis"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 12: VIDEO OBJECT DETECTION
# ═══════════════════════════════════════════════

> **Prerequisite**: Giai đoạn 5-11 phải hoàn thành trước.

## Bước 12.1 — Tải VisDrone-VID dataset

Chỉ tải sau khi DET pipeline chạy ổn.

```bash
python scripts/download_dataset.py --subset VID
```

## Bước 12.2 — Tạo video inference script

Tạo `scripts/video_inference.py`

Logic:
1. Load trained model (best.pt)
2. Đọc video frame by frame (OpenCV VideoCapture)
3. Chạy detection trên mỗi frame
4. Vẽ bounding boxes
5. Ghi output video (OpenCV VideoWriter)
6. Đo FPS thực tế (không bao gồm visualization)

## Bước 12.3 — Benchmark per-frame latency

Ghi nhận:
- Frame read time
- Preprocessing time
- Inference time
- Postprocessing time
- Visualization time (nếu có)
- Total per-frame time
- Effective FPS

## Bước 12.4 — Commit

```bash
git add scripts/video_inference.py
git commit -m "feat: video object detection pipeline"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 13: MULTI-OBJECT TRACKING
# ═══════════════════════════════════════════════

## Bước 13.1 — Cài đặt tracker

Ultralytics đã tích hợp ByteTrack và BoT-SORT.

## Bước 13.2 — Tạo tracking script

Tạo `scripts/run_tracking.py`

```python
# ByteTrack
yolo track model=best.pt source=video.mp4 tracker=bytetrack.yaml

# BoT-SORT  
yolo track model=best.pt source=video.mp4 tracker=botsort.yaml
```

## Bước 13.3 — Tạo tracking config

Tạo `configs/bytetrack.yaml` và `configs/botsort.yaml`

## Bước 13.4 — Đánh giá tracking metrics

Tạo `src/aeroeval/metrics/tracking.py`

Metrics:
- MOTA (Multi-Object Tracking Accuracy)
- IDF1 (ID F1 Score)
- HOTA (Higher Order Tracking Accuracy)
- ID Switches
- MT (Mostly Tracked)
- ML (Mostly Lost)
- FP (False Positives)
- FN (False Negatives)

Sử dụng thư viện `trackeval` hoặc `motmetrics`:
```bash
pip install motmetrics
```

## Bước 13.5 — Tạo tracking visualization

Output video với:
- Bounding boxes
- Track IDs (số trên mỗi box)
- Trajectories (đường đi lịch sử)
- Màu unique cho mỗi track ID

## Bước 13.6 — Commit

```bash
git add scripts/run_tracking.py src/aeroeval/metrics/tracking.py configs/bytetrack.yaml configs/botsort.yaml
git commit -m "feat: multi-object tracking with ByteTrack/BoT-SORT"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 14: REAL-TIME BENCHMARKING
# ═══════════════════════════════════════════════

## Bước 14.1 — Tạo benchmarking module

Tạo `src/aeroeval/metrics/efficiency.py`

### Metrics đo:
```python
BENCHMARK_METRICS = {
    "model_latency_ms": ...,       # Pure inference time
    "preprocess_time_ms": ...,     # Image preprocessing
    "postprocess_time_ms": ...,    # NMS, filtering
    "tracking_time_ms": ...,       # Tracker update (if applicable)
    "e2e_latency_ms": ...,         # End-to-end per frame
    "fps_model": ...,              # 1000 / model_latency
    "fps_e2e": ...,                # 1000 / e2e_latency
    "cpu_usage_percent": ...,
    "gpu_usage_percent": ...,
    "vram_mb": ...,
    "ram_mb": ...,
    "model_size_mb": ...,
    "num_parameters": ...,
    "flops": ...,
}
```

### Methodology:
1. Warmup: 50 frames (không đo)
2. Benchmark: 200 frames
3. Report: mean, std, min, max, p50, p95, p99
4. Dùng `torch.cuda.synchronize()` trước mỗi timestamp (nếu GPU)

## Bước 14.2 — Tạo benchmark script

Tạo `scripts/benchmark.py`

```bash
python scripts/benchmark.py \
    --model experiments/baseline_yolo11n/weights/best.pt \
    --imgsz 640 \
    --device 0 \
    --warmup 50 \
    --iterations 200
```

## Bước 14.3 — Output

```
reports/benchmark/
├── latency_breakdown.png      — stacked bar: preprocess, inference, postprocess
├── fps_over_time.png          — line chart showing FPS stability
├── resource_usage.png         — CPU, GPU, memory over time
├── benchmark_summary.json
└── benchmark_summary.csv
```

## Bước 14.4 — Commit

```bash
git add src/aeroeval/metrics/efficiency.py scripts/benchmark.py
git commit -m "feat: real-time performance benchmarking"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 15: ONNX EXPORT & OPTIMIZATION
# ═══════════════════════════════════════════════

## Bước 15.1 — Export ONNX

```bash
yolo export model=experiments/baseline_yolo11n/weights/best.pt format=onnx
```

## Bước 15.2 — Validate ONNX model

Tạo `scripts/validate_onnx.py`

1. Load ONNX model với onnxruntime
2. Chạy inference trên val set
3. So sánh results với PyTorch model
4. Đo accuracy difference

## Bước 15.3 — Benchmark ONNX

So sánh:
```
| Engine   | Precision | mAP50 | mAP50-95 | Latency(ms) | FPS   | Size(MB) |
|----------|-----------|-------|----------|-------------|-------|----------|
| PyTorch  | FP32      | X.XX  | X.XX     | X.X         | XXX   | X.X      |
| ONNX     | FP32      | X.XX  | X.XX     | X.X         | XXX   | X.X      |
```

## Bước 15.4 — FP16 Export (nếu có NVIDIA GPU)

```bash
yolo export model=best.pt format=onnx half=True
```

## Bước 15.5 — TensorRT (nếu có NVIDIA GPU + TensorRT)

```bash
yolo export model=best.pt format=engine
```

Benchmark:
```
| Engine    | Precision | mAP50 | Latency(ms) | FPS   |
|-----------|-----------|-------|-------------|-------|
| PyTorch   | FP32      | X.XX  | X.X         | XXX   |
| ONNX      | FP32      | X.XX  | X.X         | XXX   |
| TensorRT  | FP16      | X.XX  | X.X         | XXX   |
| TensorRT  | INT8      | X.XX  | X.X         | XXX   |
```

## Bước 15.6 — Commit

```bash
git add scripts/validate_onnx.py
git commit -m "feat: ONNX/TensorRT export and optimization benchmark"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 16: XÂY DỰNG AI EVALUATION ENGINE
# ═══════════════════════════════════════════════

## Bước 16.1 — Refactor code thành framework

Di chuyển và tổ chức tất cả code thành package:

```
src/aeroeval/
├── __init__.py
├── metrics/
│   ├── __init__.py
│   ├── detection.py      — mAP, AP, precision, recall, per-class, per-size
│   ├── tracking.py       — MOTA, IDF1, HOTA, ID switches
│   ├── calibration.py    — confidence analysis, reliability diagram
│   ├── error_analysis.py — error taxonomy, failure modes
│   └── efficiency.py     — latency, FPS, memory, model size
│
├── robustness/
│   ├── __init__.py
│   ├── corruptions.py    — unified corruption registry
│   ├── blur.py
│   ├── noise.py
│   ├── brightness.py
│   ├── compression.py
│   ├── resize.py
│   └── occlusion.py
│
├── models/
│   ├── __init__.py
│   ├── registry.py       — model registration & management
│   └── runner.py         — unified inference interface
│
├── reporting/
│   ├── __init__.py
│   ├── report.py         — generate HTML/JSON report
│   └── recommendation.py — model recommendation engine
│
└── pipeline/
    ├── __init__.py
    └── evaluate.py       — orchestrate full evaluation
```

## Bước 16.2 — Tạo Model Registry

`src/aeroeval/models/registry.py`:

```python
class ModelRegistry:
    def register(name, path, format, metadata)
    def get(name) -> ModelInfo
    def list() -> List[ModelInfo]
    def compare(names) -> ComparisonTable
```

## Bước 16.3 — Tạo Evaluation Pipeline

`src/aeroeval/pipeline/evaluate.py`:

```python
class EvaluationPipeline:
    def __init__(config):
        ...
    
    def run():
        results = {}
        results["detection"] = self.evaluate_detection()
        results["small_object"] = self.evaluate_by_size()
        results["robustness"] = self.evaluate_robustness()
        results["calibration"] = self.analyze_calibration()
        results["errors"] = self.analyze_errors()
        results["efficiency"] = self.benchmark()
        
        self.generate_report(results)
        return results
```

## Bước 16.4 — Tạo CLI entry point

Tạo `scripts/aeroeval_cli.py` hoặc setup console_scripts:

```bash
python -m aeroeval evaluate \
    --model models/yolo11n.pt \
    --dataset configs/visdrone.yaml \
    --robustness full \
    --benchmark \
    --output reports/run_001/
```

## Bước 16.5 — Output structure

Mỗi evaluation run tạo:
```
reports/run_001/
├── summary.json           — all metrics in one file
├── metrics.csv            — detection metrics table
├── robustness.csv         — robustness results table
├── efficiency.csv         — benchmark results
├── errors.csv             — error taxonomy counts
├── figures/
│   ├── confusion_matrix.png
│   ├── pr_curve.png
│   ├── robustness_heatmap.png
│   ├── latency_breakdown.png
│   ├── error_distribution.png
│   └── ...
└── evaluation_report.html — standalone HTML report
```

## Bước 16.6 — Commit

```bash
git add src/aeroeval/
git commit -m "feat: unified AI evaluation engine framework"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 17: MODEL RECOMMENDATION
# ═══════════════════════════════════════════════

## Bước 17.1 — Tạo recommendation engine

Tạo `src/aeroeval/reporting/recommendation.py`

### Deployment profiles:
```python
PROFILES = {
    "real_time_uav": {
        "accuracy": 0.30,
        "latency": 0.30,
        "robustness": 0.25,
        "memory": 0.15,
    },
    "high_accuracy": {
        "accuracy": 0.50,
        "latency": 0.15,
        "robustness": 0.25,
        "memory": 0.10,
    },
    "edge_device": {
        "accuracy": 0.20,
        "latency": 0.25,
        "robustness": 0.15,
        "memory": 0.40,
    },
}
```

### Logic:
1. Normalize tất cả metrics về [0, 1]
2. Tính weighted score cho mỗi model
3. Rank models
4. Output recommendation với justification

## Bước 17.2 — Output

```json
{
    "profile": "real_time_uav",
    "recommendation": "yolo11n",
    "score": 0.82,
    "justification": "Best balance of accuracy (mAP 42.3) and latency (8.2ms) for real-time UAV deployment",
    "rankings": [
        {"model": "yolo11n", "score": 0.82},
        {"model": "yolo11s", "score": 0.76}
    ]
}
```

## Bước 17.3 — Commit

```bash
git add src/aeroeval/reporting/recommendation.py
git commit -m "feat: model recommendation engine with deployment profiles"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 18: FASTAPI EVALUATION API
# ═══════════════════════════════════════════════

## Bước 18.1 — Tạo API structure

```
src/aeroeval/api/
├── __init__.py
├── main.py          — FastAPI app
├── routes/
│   ├── __init__.py
│   ├── models.py    — model registration endpoints
│   ├── evaluate.py  — evaluation endpoints
│   └── results.py   — results retrieval endpoints
├── schemas.py       — Pydantic models
└── dependencies.py  — shared dependencies
```

## Bước 18.2 — Implement endpoints

### `POST /models/register`
Register a model for evaluation.

### `GET /models`
List all registered models.

### `POST /evaluate`
Run full evaluation pipeline.
Request:
```json
{
    "model": "yolo11n",
    "dataset": "visdrone",
    "profile": "real_time_uav",
    "robustness": true,
    "benchmark": true
}
```

### `POST /robustness`
Run robustness evaluation only.

### `POST /benchmark`
Run benchmark only.

### `GET /results/{run_id}`
Get results of a specific evaluation run.

### `GET /results/{run_id}/report`
Get HTML report.

## Bước 18.3 — Chạy API

```bash
uvicorn src.aeroeval.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Kiểm tra: http://localhost:8000/docs (Swagger UI)

## Bước 18.4 — Commit

```bash
git add src/aeroeval/api/
git commit -m "feat: FastAPI evaluation API"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 19: STREAMLIT DASHBOARD
# ═══════════════════════════════════════════════

## Bước 19.1 — Tạo dashboard structure

```
src/aeroeval/dashboard/
├── app.py               — main Streamlit app
├── pages/
│   ├── 1_Overview.py
│   ├── 2_Model_Comparison.py
│   ├── 3_Robustness.py
│   ├── 4_Error_Analysis.py
│   └── 5_Deployment.py
└── utils.py             — shared dashboard utilities
```

## Bước 19.2 — Page 1: Overview
- Best model summary card
- Key metrics: mAP, FPS, latency, robustness score, model size
- Latest evaluation run info

## Bước 19.3 — Page 2: Model Comparison
- Side-by-side metrics table
- Interactive charts: mAP vs latency scatter
- Per-class AP comparison bar chart
- Radar chart: accuracy, speed, robustness, efficiency

## Bước 19.4 — Page 3: Robustness
- Corruption type selector
- Severity slider
- Before/after image comparison
- mAP drop visualization
- Heatmap: corruption × severity

## Bước 19.5 — Page 4: Error Analysis
- Error distribution pie chart
- Example failure images (interactive gallery)
- Top failure modes table
- Filter by error type

## Bước 19.6 — Page 5: Deployment
- PyTorch vs ONNX vs TensorRT comparison
- FP32 vs FP16 vs INT8 comparison
- Latency breakdown chart
- Deployment recommendation

## Bước 19.7 — Chạy dashboard

```bash
streamlit run src/aeroeval/dashboard/app.py
```

## Bước 19.8 — Commit

```bash
git add src/aeroeval/dashboard/
git commit -m "feat: Streamlit evaluation dashboard with 5 pages"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 20: TESTING
# ═══════════════════════════════════════════════

## Bước 20.1 — Unit tests

```
tests/
├── conftest.py            — shared fixtures
├── test_metrics.py        — test IoU, mAP calculation, AP computation
├── test_robustness.py     — test each corruption function
├── test_dataset.py        — test annotation parsing, conversion
├── test_api.py            — test FastAPI endpoints
├── test_model_registry.py — test model registration
├── test_recommendation.py — test recommendation engine
├── test_calibration.py    — test calibration analysis
└── test_error_analysis.py — test error categorization
```

### test_metrics.py examples:
```python
def test_iou_perfect_overlap():
    ...

def test_iou_no_overlap():
    ...

def test_iou_partial_overlap():
    ...

def test_map_calculation():
    ...

def test_per_class_ap():
    ...

def test_size_categorization():
    ...
```

### test_robustness.py examples:
```python
def test_gaussian_blur_output_shape():
    ...

def test_brightness_range():
    ...

def test_noise_is_applied():
    ...

def test_jpeg_compression():
    ...
```

### test_api.py examples:
```python
def test_health_check():
    ...

def test_register_model():
    ...

def test_evaluate_endpoint():
    ...

def test_get_results():
    ...
```

## Bước 20.2 — Chạy tests

```bash
pytest -q
pytest --cov=src/aeroeval --cov-report=html
```

## Bước 20.3 — Commit

```bash
git add tests/
git commit -m "feat: comprehensive test suite"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 21: DOCKER
# ═══════════════════════════════════════════════

## Bước 21.1 — Tạo Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/

EXPOSE 8000 8501

CMD ["uvicorn", "src.aeroeval.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Bước 21.2 — Tạo docker-compose.yml

```yaml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./reports:/app/reports
      - ./experiments:/app/experiments
    command: uvicorn src.aeroeval.api.main:app --host 0.0.0.0 --port 8000

  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./reports:/app/reports
    command: streamlit run src/aeroeval/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    depends_on:
      - api
```

## Bước 21.3 — Tạo .dockerignore

```
.venv/
__pycache__/
*.pyc
.git/
data/VisDrone*/
data/visdrone_yolo/
experiments/*/
*.pt
*.pth
*.onnx
*.engine
.env
```

## Bước 21.4 — Test Docker build

```bash
docker compose up --build
```

Kiểm tra:
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

## Bước 21.5 — Commit

```bash
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "feat: Docker containerization for API and dashboard"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 22: CI/CD (GitHub Actions)
# ═══════════════════════════════════════════════

## Bước 22.1 — Tạo workflow

Tạo `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check src/ tests/ scripts/

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest -q --tb=short

  docker:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - run: docker compose build
```

## Bước 22.2 — Commit

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions for lint, test, and Docker build"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 23: EXPERIMENT REPRODUCIBILITY
# ═══════════════════════════════════════════════

## Bước 23.1 — Tạo experiment config files

```
configs/
├── visdrone.yaml        — dataset config
├── baseline.yaml        — baseline training config
├── robustness.yaml      — robustness test config
├── benchmark.yaml       — benchmark config
└── deployment.yaml      — deployment profile config
```

## Bước 23.2 — Tạo experiment logger

Tạo `src/aeroeval/pipeline/experiment_logger.py`

Mỗi experiment ghi:
```json
{
    "experiment_id": "exp_001",
    "timestamp": "2024-XX-XX",
    "model": "yolo11n",
    "dataset": "visdrone_det",
    "git_commit": "abc123",
    "seed": 42,
    "hardware": "NVIDIA RTX 3060",
    "parameters": {...},
    "metrics": {...},
    "duration_seconds": 3600
}
```

## Bước 23.3 — Tạo dataset preparation script

Tạo `scripts/prepare_dataset.py`:
- Tải dataset (nếu chưa có)
- Giải nén
- Chuyển đổi annotations
- Verify checksums
- Report statistics

Đây là one-command setup cho người mới:
```bash
python scripts/prepare_dataset.py
```

## Bước 23.4 — Commit

```bash
git add configs/ src/aeroeval/pipeline/experiment_logger.py scripts/prepare_dataset.py
git commit -m "feat: experiment reproducibility with config management"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 24: FINAL REPORT
# ═══════════════════════════════════════════════

## Bước 24.1 — Viết báo cáo cuối cùng

Tạo `reports/final_report.md` (15-25 trang)

### Cấu trúc:

```
1. Problem Definition
   - UAV vision challenges
   - Why evaluation matters
   - Project objectives

2. Dataset
   - VisDrone description
   - Statistics (from Phase 2)
   - Class imbalance analysis
   - Small object prevalence

3. Baseline
   - Model selection rationale
   - Training setup
   - Baseline results

4. Detection Performance
   - Overall metrics
   - Per-class analysis
   - Qualitative results

5. Small-object Analysis
   - Size distribution
   - Performance by object size
   - Key findings

6. Robustness Evaluation
   - Corruption types tested
   - Results table
   - Performance degradation analysis
   - Worst-case scenarios

7. Tracking (nếu hoàn thành)
   - Tracker selection
   - Tracking metrics
   - Temporal analysis

8. Model Comparison
   - Models compared
   - Accuracy vs efficiency trade-off
   - Radar chart analysis

9. Runtime Benchmark
   - Latency breakdown
   - FPS analysis
   - Resource usage

10. ONNX/TensorRT Optimization
    - Export results
    - Accuracy vs speed trade-off
    - Precision comparison (FP32/FP16/INT8)

11. Evaluation Framework
    - Architecture design
    - Reusability
    - CLI usage

12. API & Dashboard
    - API endpoints
    - Dashboard features
    - Screenshots

13. Error Analysis
    - Error taxonomy
    - Top failure modes
    - Example failures

14. Limitations
    - Dataset limitations
    - Model limitations
    - Evaluation limitations
    - What would improve with more time/resources

15. Conclusion
    - Key findings
    - Practical recommendations
    - Future work
```

## Bước 24.2 — Tạo required figures

Tối thiểu 10 figures:
1. System architecture diagram
2. Dataset examples (ảnh mẫu với annotations)
3. Ground truth vs prediction comparison
4. Model comparison chart
5. Robustness degradation curves
6. Latency vs mAP scatter plot
7. FP32 vs FP16 vs INT8 comparison
8. Tracking visualization frames
9. Dashboard screenshot
10. Automated evaluation report screenshot

## Bước 24.3 — Commit

```bash
git add reports/final_report.md
git commit -m "docs: comprehensive final evaluation report"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 25: README & DEMO
# ═══════════════════════════════════════════════

## Bước 25.1 — Viết README.md hoàn chỉnh

Cấu trúc:

```markdown
# AeroEval
Real-Time UAV Vision & AI Evaluation Platform

[Badges: CI status, Python version, License]

## Overview
A computer vision evaluation framework for drone imagery that
benchmarks detection, tracking, robustness and deployment efficiency.

## Features
- Object Detection (YOLOv11, RT-DETR)
- Multi-object Tracking (ByteTrack, BoT-SORT)
- Robustness Testing (7 corruption types)
- Small-object Evaluation
- Runtime Benchmarking
- ONNX/TensorRT Optimization
- FastAPI Evaluation API
- Streamlit Dashboard
- Automated Evaluation Reports

## Architecture
[Diagram]

## Quick Start
### Prerequisites
### Installation
### Dataset Setup
### Training
### Evaluation

## Results
[Key metrics table]
[Key figures]

## API
[Endpoint documentation]

## Dashboard
[Screenshots]

## Docker
[Docker commands]

## Project Structure
[Directory tree]

## Limitations

## License
```

## Bước 25.2 — Tạo demo video (60-90 giây)

Cấu trúc video:
```
0-15s:   Drone video input
15-30s:  Detection + tracking overlay
30-45s:  Robustness evaluation visualization
45-60s:  Model comparison charts
60-75s:  Dashboard walkthrough
75-90s:  Deployment benchmark results
```

Tools: OBS Studio, ffmpeg, hoặc screen recording.

Lưu GIF/video vào repo (hoặc link YouTube).

## Bước 25.3 — Commit

```bash
git add README.md
git commit -m "docs: comprehensive README with architecture and results"
git push origin main
```

---

# ═══════════════════════════════════════════════
# GIAI ĐOẠN 26: POLISH & FINALIZE
# ═══════════════════════════════════════════════

## Bước 26.1 — Code quality check

```bash
ruff check src/ tests/ scripts/ --fix
ruff format src/ tests/ scripts/
pytest -q --tb=short
```

## Bước 26.2 — Kiểm tra .gitignore

Đảm bảo KHÔNG có trong repo:
- [ ] Dataset files
- [ ] Model weights (.pt, .onnx, .engine)
- [ ] Large image files
- [ ] .env files
- [ ] __pycache__/
- [ ] .venv/

## Bước 26.3 — Kiểm tra reproducibility

Một người mới clone repo phải có thể:
1. `git clone` → OK
2. `python -m venv .venv && source .venv/bin/activate` → OK
3. `pip install -r requirements.txt` → OK
4. `python scripts/prepare_dataset.py` → download & setup data
5. `python scripts/train_baseline.py` → train model
6. `python -m aeroeval evaluate --model ... --dataset ...` → run eval
7. `docker compose up --build` → API + Dashboard

## Bước 26.4 — Final commit

```bash
git add -A
git commit -m "chore: final polish and cleanup"
git push origin main
```

---

# ═══════════════════════════════════════════════
# TIMELINE TỔNG HỢP (8-10 Tuần)
# ═══════════════════════════════════════════════

| Tuần | Giai đoạn                              | Output chính                          |
|------|----------------------------------------|---------------------------------------|
| 1    | 0-4: Setup, Data, Inspect, Convert     | Repo + dataset pipeline + statistics  |
| 2    | 5-6: Baseline train + evaluation       | Trained model + baseline report       |
| 3    | 7-8: Small-object + Robustness         | Size analysis + robustness report     |
| 4    | 9-10: Model comparison + Calibration   | Comparison table + calibration report |
| 5    | 11-13: Errors + Video + Tracking       | Error taxonomy + tracking pipeline    |
| 6    | 14-15: Benchmarking + ONNX/TensorRT   | Benchmark report + optimized models   |
| 7    | 16-17: Eval Engine + Recommendation    | Unified framework + recommendation    |
| 8    | 18-19: API + Dashboard                 | FastAPI + Streamlit                   |
| 9    | 20-22: Testing + Docker + CI           | Tests + Docker + GitHub Actions       |
| 10   | 23-26: Report + README + Demo + Polish | Final deliverables                    |

---

# ═══════════════════════════════════════════════
# MILESTONE CHECKPOINTS
# ═══════════════════════════════════════════════

## Milestone A — MVP (Tuần 1-4)
Đủ để demo cơ bản:
- [x] Dataset pipeline hoạt động
- [x] Baseline model trained
- [x] Basic evaluation metrics
- [x] Robustness testing
- [x] Error analysis

## Milestone B — Advanced (Tuần 5-6)
Thêm video + tracking + optimization:
- [x] Video inference
- [x] Multi-object tracking
- [x] ONNX export
- [x] Comprehensive benchmarking

## Milestone C — Professional (Tuần 7-10)
Đầy đủ engineering:
- [x] Unified evaluation engine
- [x] FastAPI
- [x] Streamlit dashboard
- [x] Docker
- [x] CI/CD
- [x] Tests
- [x] Final report
- [x] README + Demo

## Minimum CV-Ready (sau Milestone A + API + Dashboard)
- Detection pipeline ✓
- Robustness evaluation ✓
- Error analysis ✓
- Model comparison ✓
- FastAPI ✓
- Dashboard ✓
- Docker ✓
- Tests ✓

---

# NGUYÊN TẮC XUYÊN SUỐT

1. **Data trước, model sau** — Hiểu data trước khi train
2. **Measure everything** — Không claim nếu chưa đo
3. **Reproducibility** — Mọi thứ phải tái tạo được
4. **Không fake metrics** — Chỉ report kết quả thực
5. **Progressive complexity** — Từ đơn giản đến phức tạp
6. **Commit thường xuyên** — Mỗi feature = 1 commit
7. **Không commit data/weights** — Chỉ code và config
8. **Document limitations** — Thành thật về giới hạn
