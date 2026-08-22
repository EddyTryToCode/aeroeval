# Step 6C — Drone Vision Robustness Benchmark

### 1. Overall Average Retention Rate (% of Clean Performance Retained)

|                             |   Retention_Rate_% |
|:----------------------------|-------------------:|
| ('A', 'A (YOLO11n-640)')    |               87.7 |
| ('B1', 'B1 (YOLO11s-960)')  |               83.5 |
| ('B2', 'B2 (YOLO11s-1280)') |               80.7 |
| ('B3', 'B3 (YOLO11m-960)')  |               82.7 |

### 2. Mean Corrupted mAP50 under Environmental Shifts

| Corruption         |     A |    B1 |    B2 |    B3 |
|:-------------------|------:|------:|------:|------:|
| Gaussian Blur      | 0.289 | 0.417 | 0.45  | 0.471 |
| Gaussian Noise     | 0.18  | 0.257 | 0.312 | 0.293 |
| JPEG Compression   | 0.284 | 0.431 | 0.472 | 0.471 |
| Low Light          | 0.29  | 0.459 | 0.514 | 0.515 |
| Motion Blur        | 0.202 | 0.272 | 0.268 | 0.293 |
| Occlusion          | 0.238 | 0.401 | 0.442 | 0.448 |
| Overexposure       | 0.272 | 0.447 | 0.498 | 0.488 |
| Resolution Degrade | 0.284 | 0.38  | 0.393 | 0.408 |

