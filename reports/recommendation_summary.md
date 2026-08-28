# UAV Model Deployment Recommendations

## Profile: `real_time_uav`
> **Description**: Balanced high-speed UAV tracking with robustness to camera motion and blur

- **Recommended Choice**: **`Exp B2 (YOLO11s-1280)`** (Score: `0.6554`)
- **Justification**: Model 'Exp B2 (YOLO11s-1280)' ranks #1 for profile 'real_time_uav' (Composite Score: 0.655). It delivers 46.8% accuracy at 38.5 ms latency (26.0 FPS), maintaining 88.6% robustness under environmental corruptions with a compact footprint of 18.4 MB.

| Rank | Model | Score | Accuracy | Latency (ms) | FPS | Robustness (%) | Memory (MB) |
|---|---|---|---|---|---|---|---|
| 1 | **Exp B2 (YOLO11s-1280)** | 0.655 | 46.8 | 38.5 ms | 26.0 | 88.6% | 18.4 MB |
| 2 | **Exp B1 (YOLO11s-960)** | 0.592 | 43.1 | 23.8 ms | 42.0 | 84.2% | 18.4 MB |
| 3 | **Exp B3 (YOLO11m-960)** | 0.550 | 47.2 | 42.1 ms | 23.8 | 89.1% | 39.2 MB |
| 4 | **Exp A (YOLO11n-640)** | 0.450 | 37.4 | 13.4 ms | 74.6 | 78.5% | 5.2 MB |

---

## Profile: `high_accuracy`
> **Description**: High-altitude reconnaissance prioritizing small-object precision and recall

- **Recommended Choice**: **`Exp B2 (YOLO11s-1280)`** (Score: `0.7978`)
- **Justification**: Model 'Exp B2 (YOLO11s-1280)' ranks #1 for profile 'high_accuracy' (Composite Score: 0.798). It delivers 46.8% accuracy at 38.5 ms latency (26.0 FPS), maintaining 88.6% robustness under environmental corruptions with a compact footprint of 18.4 MB.

| Rank | Model | Score | Accuracy | Latency (ms) | FPS | Robustness (%) | Memory (MB) |
|---|---|---|---|---|---|---|---|
| 1 | **Exp B2 (YOLO11s-1280)** | 0.798 | 46.8 | 38.5 ms | 26.0 | 88.6% | 18.4 MB |
| 2 | **Exp B3 (YOLO11m-960)** | 0.750 | 47.2 | 42.1 ms | 23.8 | 89.1% | 39.2 MB |
| 3 | **Exp B1 (YOLO11s-960)** | 0.582 | 43.1 | 23.8 ms | 42.0 | 84.2% | 18.4 MB |
| 4 | **Exp A (YOLO11n-640)** | 0.250 | 37.4 | 13.4 ms | 74.6 | 78.5% | 5.2 MB |

---

## Profile: `edge_device`
> **Description**: Low-power companion microcontrollers / Jetson Nano with strict memory limits

- **Recommended Choice**: **`Exp A (YOLO11n-640)`** (Score: `0.65`)
- **Justification**: Model 'Exp A (YOLO11n-640)' ranks #1 for profile 'edge_device' (Composite Score: 0.650). It delivers 37.4% accuracy at 13.4 ms latency (74.6 FPS), maintaining 78.5% robustness under environmental corruptions with a compact footprint of 5.2 MB.

| Rank | Model | Score | Accuracy | Latency (ms) | FPS | Robustness (%) | Memory (MB) |
|---|---|---|---|---|---|---|---|
| 1 | **Exp A (YOLO11n-640)** | 0.650 | 37.4 | 13.4 ms | 74.6 | 78.5% | 5.2 MB |
| 2 | **Exp B2 (YOLO11s-1280)** | 0.611 | 46.8 | 38.5 ms | 26.0 | 88.6% | 18.4 MB |
| 3 | **Exp B1 (YOLO11s-960)** | 0.601 | 43.1 | 23.8 ms | 42.0 | 84.2% | 18.4 MB |
| 4 | **Exp B3 (YOLO11m-960)** | 0.350 | 47.2 | 42.1 ms | 23.8 | 89.1% | 39.2 MB |

---
