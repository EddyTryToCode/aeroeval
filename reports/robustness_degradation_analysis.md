# Step 6D — Robustness Degradation Analysis

### 1. Absolute mAP50 Drop by Perturbation

| Corruption         |     A |    B1 |    B2 |     B3 |
|:-------------------|------:|------:|------:|-------:|
| Gaussian Blur      | 0.002 | 0.041 | 0.069 |  0.041 |
| Gaussian Noise     | 0.111 | 0.202 | 0.207 |  0.219 |
| JPEG Compression   | 0.007 | 0.028 | 0.046 |  0.04  |
| Low Light          | 0.001 | 0     | 0.005 | -0.003 |
| Motion Blur        | 0.088 | 0.187 | 0.251 |  0.219 |
| Occlusion          | 0.053 | 0.058 | 0.077 |  0.064 |
| Overexposure       | 0.019 | 0.012 | 0.02  |  0.024 |
| Resolution Degrade | 0.006 | 0.079 | 0.126 |  0.104 |

### 2. Relative Performance Degradation (% Loss from Clean)

| Corruption         |    A |   B1 |   B2 |   B3 |
|:-------------------|-----:|-----:|-----:|-----:|
| Gaussian Blur      |  0.5 |  9   | 13.3 |  8   |
| Gaussian Noise     | 38.1 | 44   | 40   | 42.8 |
| JPEG Compression   |  2.4 |  6.2 |  8.9 |  8   |
| Low Light          |  0.4 |  0   |  1.1 | -0.6 |
| Motion Blur        | 30.4 | 40.7 | 48.4 | 42.7 |
| Occlusion          | 18.2 | 12.6 | 14.8 | 12.4 |
| Overexposure       |  6.4 |  2.6 |  3.9 |  4.6 |
| Resolution Degrade |  2.1 | 17.2 | 24.3 | 20.3 |

