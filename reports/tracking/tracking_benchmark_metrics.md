# Step 13 — UAV Multi-Object Tracking Evaluation

### 1. Tracking Performance & Temporal Consistency

| Detector          | Tracker   |   MOTA |   IDF1 |   ID_Switches |   MT_% |   ML_% |   Latency_ms |   Tracking_FPS |
|:------------------|:----------|-------:|-------:|--------------:|-------:|-------:|-------------:|---------------:|
| B1 (YOLO11s-960)  | ByteTrack |  0.022 |  0.048 |            12 |    0   |   93.8 |        67.85 |           14.7 |
| B1 (YOLO11s-960)  | BoT-SORT  |  0.022 |  0.048 |            12 |    0   |   93.8 |        35.33 |           28.3 |
| B2 (YOLO11s-1280) | ByteTrack |  0.073 |  0.148 |            54 |    6.2 |   75   |       290.08 |            3.4 |
| B2 (YOLO11s-1280) | BoT-SORT  |  0.073 |  0.148 |            54 |    6.2 |   75   |       146.96 |            6.8 |

