# Step 6F — Deployment Trade-off & Model Selection Analysis

### 1. Raw Engineering Benchmark Matrix

| Experiment   | model   |   size |   mAP50 |   mAP50_95 |   robustness_retention |   latency_ms |   fps |   params_m |   size_mb |
|:-------------|:--------|-------:|--------:|-----------:|-----------------------:|-------------:|------:|-----------:|----------:|
| A            | YOLO11n |    640 |   0.291 |      0.164 |                   87.7 |         9.82 | 101.9 |       2.58 |       5.4 |
| B1           | YOLO11s |    960 |   0.473 |      0.285 |                   83.5 |        19.19 |  52.1 |       9.42 |      18.4 |
| B2           | YOLO11s |   1280 |   0.533 |      0.329 |                   80.7 |        34.05 |  29.4 |       9.42 |      18.4 |
| B3           | YOLO11m |    960 |   0.531 |      0.326 |                   82.7 |        45.78 |  21.8 |      20.04 |      39.5 |

### 2. Composite Decision Scores across Deployment Profiles

| Deployment_Profile                 |   A |   B1 |   B2 |   B3 |
|:-----------------------------------|----:|-----:|-----:|-----:|
| Balanced General UAV               |  70 | 53.2 | 42.1 | 36.9 |
| Edge AI / Low-Power Compute        |  80 | 55.4 | 46.6 | 25.5 |
| High-Altitude Precision Inspection |  55 | 57.7 | 52.6 | 53.2 |
| Real-Time Embedded Drone           |  75 | 52.4 | 40.7 | 30.5 |

### 3. Detailed Profile Rankings

| Deployment_Profile                 | Experiment   | Model_Config   |   Composite_Score |   mAP50 |   FPS |   Latency_ms |   Size_MB |   Rank |
|:-----------------------------------|:-------------|:---------------|------------------:|--------:|------:|-------------:|----------:|-------:|
| Balanced General UAV               | A            | YOLO11n@640    |              70   |   0.291 | 101.9 |         9.82 |       5.4 |      1 |
| Balanced General UAV               | B1           | YOLO11s@960    |              53.2 |   0.473 |  52.1 |        19.19 |      18.4 |      2 |
| Balanced General UAV               | B2           | YOLO11s@1280   |              42.1 |   0.533 |  29.4 |        34.05 |      18.4 |      3 |
| Balanced General UAV               | B3           | YOLO11m@960    |              36.9 |   0.531 |  21.8 |        45.78 |      39.5 |      4 |
| Edge AI / Low-Power Compute        | A            | YOLO11n@640    |              80   |   0.291 | 101.9 |         9.82 |       5.4 |      1 |
| Edge AI / Low-Power Compute        | B1           | YOLO11s@960    |              55.4 |   0.473 |  52.1 |        19.19 |      18.4 |      2 |
| Edge AI / Low-Power Compute        | B2           | YOLO11s@1280   |              46.6 |   0.533 |  29.4 |        34.05 |      18.4 |      3 |
| Edge AI / Low-Power Compute        | B3           | YOLO11m@960    |              25.5 |   0.531 |  21.8 |        45.78 |      39.5 |      4 |
| High-Altitude Precision Inspection | B1           | YOLO11s@960    |              57.7 |   0.473 |  52.1 |        19.19 |      18.4 |      1 |
| High-Altitude Precision Inspection | A            | YOLO11n@640    |              55   |   0.291 | 101.9 |         9.82 |       5.4 |      2 |
| High-Altitude Precision Inspection | B3           | YOLO11m@960    |              53.2 |   0.531 |  21.8 |        45.78 |      39.5 |      3 |
| High-Altitude Precision Inspection | B2           | YOLO11s@1280   |              52.6 |   0.533 |  29.4 |        34.05 |      18.4 |      4 |
| Real-Time Embedded Drone           | A            | YOLO11n@640    |              75   |   0.291 | 101.9 |         9.82 |       5.4 |      1 |
| Real-Time Embedded Drone           | B1           | YOLO11s@960    |              52.4 |   0.473 |  52.1 |        19.19 |      18.4 |      2 |
| Real-Time Embedded Drone           | B2           | YOLO11s@1280   |              40.7 |   0.533 |  29.4 |        34.05 |      18.4 |      3 |
| Real-Time Embedded Drone           | B3           | YOLO11m@960    |              30.5 |   0.531 |  21.8 |        45.78 |      39.5 |      4 |

