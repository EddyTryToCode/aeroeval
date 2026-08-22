# Step 10 — Confidence & Calibration Analysis

### 1. Optimal Confidence Thresholds & Calibration Errors

|    | Model             |   optimal_threshold |   max_f1 |   precision_at_optimal |   recall_at_optimal |    ECE |
|:---|:------------------|--------------------:|---------:|-----------------------:|--------------------:|-------:|
| A  | A (YOLO11n-640)   |                0.2  |    0.505 |                  0.595 |               0.44  | 0.0321 |
| B1 | B1 (YOLO11s-960)  |                0.25 |    0.656 |                  0.717 |               0.604 | 0.0428 |
| B2 | B2 (YOLO11s-1280) |                0.25 |    0.693 |                  0.735 |               0.655 | 0.0527 |
| B3 | B3 (YOLO11m-960)  |                0.3  |    0.688 |                  0.763 |               0.627 | 0.0469 |

