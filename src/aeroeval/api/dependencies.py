"""
FastAPI Shared Dependencies & Registry Singleton.
"""

from pathlib import Path

from aeroeval.models.registry import ModelRegistry

# Global singleton ModelRegistry
_registry = ModelRegistry()

# Populate with default repository models if they exist on disk
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_MODELS = [
    {
        "name": "baseline_yolo11n",
        "path": ROOT_DIR / "experiments" / "baseline_yolo11n" / "weights" / "best.pt",
        "format": "PyTorch",
        "imgsz": 640,
        "description": "Baseline YOLOv11 nano model on VisDrone"
    },
    {
        "name": "baseline_yolo11n_onnx",
        "path": ROOT_DIR / "experiments" / "baseline_yolo11n" / "weights" / "best.onnx",
        "format": "ONNX",
        "imgsz": 640,
        "description": "ONNX Export of baseline YOLOv11 nano"
    },
    {
        "name": "exp_b1_yolo11s_960",
        "path": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b1_yolo11s_960" / "weights" / "best.pt",
        "format": "PyTorch",
        "imgsz": 960,
        "description": "Exp B1 - YOLO11 small @ 960px"
    },
    {
        "name": "exp_b2_yolo11s_1280",
        "path": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b2_yolo11s_1280" / "weights" / "best.pt",
        "format": "PyTorch",
        "imgsz": 1280,
        "description": "Exp B2 - YOLO11 small @ 1280px High-Res"
    },
    {
        "name": "exp_b3_yolo11m_960",
        "path": ROOT_DIR / "runs" / "detect" / "experiments" / "exp_b3_yolo11m_960" / "weights" / "best.pt",
        "format": "PyTorch",
        "imgsz": 960,
        "description": "Exp B3 - YOLO11 medium @ 960px Capacity"
    }
]

for m in DEFAULT_MODELS:
    if m["path"].exists():
        _registry.register(
            name=m["name"],
            path=m["path"],
            format=m["format"],
            imgsz=m["imgsz"],
            description=m["description"]
        )


def get_model_registry() -> ModelRegistry:
    """Dependency injector for ModelRegistry."""
    return _registry
