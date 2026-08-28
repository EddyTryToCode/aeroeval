"""
AeroEval: Real-Time UAV Vision & AI Evaluation Platform.

A comprehensive modular evaluation framework for UAV computer vision models:
- Detection, Small-Object & Multi-Object Tracking Metrics
- Optical & Environmental Robustness Benchmarking
- Real-Time Latency, Throughput FPS & Resource Profiling
- ONNX & Edge Optimization Verification
- Deployment Recommendation Engine
"""

__version__ = "0.1.0"

from aeroeval.models.registry import ModelInfo, ModelRegistry
from aeroeval.models.runner import ModelRunner
from aeroeval.pipeline.evaluate import EvaluationPipeline
from aeroeval.reporting.recommendation import ModelRecommendationEngine
from aeroeval.reporting.report import EvaluationReport

__all__ = [
    "EvaluationPipeline",
    "ModelRegistry",
    "ModelInfo",
    "ModelRunner",
    "ModelRecommendationEngine",
    "EvaluationReport",
]
