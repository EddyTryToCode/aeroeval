"""
AeroEval Metrics Subpackage.

Provides unified interfaces for:
- Standard object detection and scale-stratified evaluation (`detection.py`)
- Real-time latency, throughput FPS, and resource profiling (`efficiency.py`)
- Multi-object tracking evaluation (`tracking.py`)
- Confidence calibration and ECE calculation (`calibration.py`)
- Failure taxonomy and error root cause analysis (`error_analysis.py`)
"""

from aeroeval.metrics.calibration import evaluate_calibration
from aeroeval.metrics.detection import (
    SIZE_THRESHOLDS,
    VISDRONE_CLASSES,
    evaluate_by_object_size,
    evaluate_detection_model,
)
from aeroeval.metrics.efficiency import benchmark_model_efficiency
from aeroeval.metrics.error_analysis import analyze_failure_taxonomy
from aeroeval.metrics.tracking import evaluate_tracking_sequence

__all__ = [
    "evaluate_detection_model",
    "evaluate_by_object_size",
    "benchmark_model_efficiency",
    "evaluate_tracking_sequence",
    "evaluate_calibration",
    "analyze_failure_taxonomy",
    "VISDRONE_CLASSES",
    "SIZE_THRESHOLDS",
]
