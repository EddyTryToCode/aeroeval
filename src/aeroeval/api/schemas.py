"""
Pydantic Schemas for AeroEval API.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelRegisterRequest(BaseModel):
    name: str = Field(..., description="Unique identifier for the model", json_schema_extra={"example": "baseline_yolo11n"})
    path: str = Field(..., description="Local path to model weights (.pt or .onnx)", json_schema_extra={"example": "experiments/baseline_yolo11n/weights/best.pt"})
    format: Optional[str] = Field(None, description="Model format ('PyTorch' or 'ONNX')", json_schema_extra={"example": "PyTorch"})
    imgsz: int = Field(640, description="Input inference resolution", json_schema_extra={"example": 640})
    description: Optional[str] = Field("", description="Model description or notes")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom metadata tags")


class ModelInfoResponse(BaseModel):
    name: str
    path: str
    format: str
    imgsz: int
    size_mb: float
    description: str
    metadata: Dict[str, Any]


class ModelListResponse(BaseModel):
    total_models: int
    models: List[ModelInfoResponse]


class EvaluateRequest(BaseModel):
    model: str = Field(..., description="Model name or file path", json_schema_extra={"example": "experiments/baseline_yolo11n/weights/best.pt"})
    dataset: str = Field("configs/visdrone.yaml", description="Path to dataset config YAML", json_schema_extra={"example": "configs/visdrone.yaml"})
    imgsz: int = Field(640, description="Image resolution", json_schema_extra={"example": 640})
    device: str = Field("0", description="Device index ('0' or 'cpu')", json_schema_extra={"example": "0"})
    profile: str = Field("real_time_uav", description="Deployment profile target", json_schema_extra={"example": "real_time_uav"})
    output_dir: Optional[str] = Field(None, description="Custom output directory")


class BenchmarkRequest(BaseModel):
    model: str = Field(..., description="Model name or file path", json_schema_extra={"example": "experiments/baseline_yolo11n/weights/best.pt"})
    imgsz: int = Field(640, description="Inference resolution", json_schema_extra={"example": 640})
    device: str = Field("0", description="Device index", json_schema_extra={"example": "0"})
    warmup: int = Field(25, description="Warmup frames", json_schema_extra={"example": 25})
    iterations: int = Field(100, description="Benchmark measurement iterations", json_schema_extra={"example": 100})


class RobustnessRequest(BaseModel):
    model: str = Field(..., description="Model path", json_schema_extra={"example": "experiments/baseline_yolo11n/weights/best.pt"})
    corruptions: Optional[List[str]] = Field(None, description="List of corruption names to evaluate")
    severity: int = Field(2, description="Corruption severity level [0-4]", ge=0, le=4)
    samples: int = Field(20, description="Number of validation images to test", json_schema_extra={"example": 20})


class EvaluationRunSummary(BaseModel):
    run_id: str
    timestamp: str
    model_name: str
    mAP50: Optional[float] = None
    mAP50_95: Optional[float] = None
    e2e_fps: Optional[float] = None
    latency_ms: Optional[float] = None
    recommendation_score: Optional[float] = None
    status: str = "completed"
