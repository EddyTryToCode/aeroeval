"""
Corruption operators for Aerial & Drone Computer Vision Robustness Benchmark.

Supports:
1. Gaussian Blur (kernel = 3, 5, 7, 9)
2. Motion Blur (kernel = 5, 9, 15, 21)
3. Brightness Dark / Low Light (scale = 0.7, 0.5, 0.3)
4. Brightness Glare / Overexposure (scale = 1.3, 1.6, 2.0)
5. Gaussian Noise (sigma = 10, 20, 35, 50)
6. JPEG Compression Artifacts (quality = 70, 50, 30, 10)
7. Occlusion / Sensor Drop (patch coverage = 5%, 10%, 20%, 30%)
8. Resolution Downscaling (320px, 480px, 640px)
"""

import random
import cv2
import numpy as np


def apply_gaussian_blur(img: np.ndarray, severity: int) -> np.ndarray:
    kernels = [3, 5, 7, 9, 13]
    k = kernels[min(severity, len(kernels) - 1)]
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    sizes = [5, 9, 15, 21, 31]
    size = sizes[min(severity, len(sizes) - 1)]
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(img, -1, kernel)


def apply_low_light(img: np.ndarray, severity: int) -> np.ndarray:
    scales = [0.8, 0.6, 0.45, 0.3, 0.15]
    factor = scales[min(severity, len(scales) - 1)]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_overexposure(img: np.ndarray, severity: int) -> np.ndarray:
    scales = [1.2, 1.4, 1.7, 2.0, 2.5]
    factor = scales[min(severity, len(scales) - 1)]
    table = np.array([min(255, int(i * factor)) for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    sigmas = [10, 20, 35, 50, 75]
    sigma = sigmas[min(severity, len(sigmas) - 1)]
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img


def apply_jpeg_compression(img: np.ndarray, severity: int) -> np.ndarray:
    qualities = [75, 50, 30, 15, 5]
    q = qualities[min(severity, len(qualities) - 1)]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)


def apply_occlusion(img: np.ndarray, severity: int) -> np.ndarray:
    coverage_ratios = [0.05, 0.10, 0.18, 0.28, 0.40]
    ratio = coverage_ratios[min(severity, len(coverage_ratios) - 1)]
    out = img.copy()
    h, w, _ = out.shape
    total_area = h * w
    target_patch_area = total_area * ratio
    
    num_patches = 4
    patch_w = int(np.sqrt(target_patch_area / num_patches))
    patch_h = patch_w

    for _ in range(num_patches):
        x = random.randint(0, max(0, w - patch_w))
        y = random.randint(0, max(0, h - patch_h))
        out[y : y + patch_h, x : x + patch_w] = 0
    return out


def apply_resolution_degradation(img: np.ndarray, severity: int) -> np.ndarray:
    downscale_factors = [0.75, 0.50, 0.35, 0.25, 0.15]
    factor = downscale_factors[min(severity, len(downscale_factors) - 1)]
    h, w, _ = img.shape
    small_w = max(16, int(w * factor))
    small_h = max(16, int(h * factor))
    
    down = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)


CORRUPTIONS = {
    "Gaussian Blur": apply_gaussian_blur,
    "Motion Blur": apply_motion_blur,
    "Low Light": apply_low_light,
    "Overexposure": apply_overexposure,
    "Gaussian Noise": apply_gaussian_noise,
    "JPEG Compression": apply_jpeg_compression,
    "Occlusion": apply_occlusion,
    "Resolution Degrade": apply_resolution_degradation
}
