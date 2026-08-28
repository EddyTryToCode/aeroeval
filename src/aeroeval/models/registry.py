"""
Model Registry Module for AeroEval.

Provides centralized management, tracking, and comparison of UAV computer vision models.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


@dataclass
class ModelInfo:
    name: str
    path: str
    format: str  # "PyTorch" (.pt) or "ONNX" (.onnx)
    imgsz: int = 640
    description: str = ""
    parameters_m: float = 0.0
    size_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """
    Central repository for registering and querying evaluation models.
    """

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}

    def register(
        self,
        name: str,
        path: Union[str, Path],
        format: Optional[str] = None,
        imgsz: int = 640,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Registers a new model into the registry.
        """
        model_path = Path(path)
        if format is None:
            format = "ONNX" if model_path.suffix.lower() == ".onnx" else "PyTorch"

        size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0.0

        info = ModelInfo(
            name=name,
            path=str(model_path.resolve() if model_path.exists() else model_path),
            format=format,
            imgsz=imgsz,
            description=description,
            size_mb=round(size_mb, 2),
            metadata=metadata or {}
        )
        self._models[name] = info
        return info

    def get(self, name: str) -> Optional[ModelInfo]:
        """Retrieves a model by its registration name."""
        return self._models.get(name)

    def list(self) -> List[ModelInfo]:
        """Returns all registered models."""
        return list(self._models.values())

    def remove(self, name: str) -> bool:
        """Removes a model from the registry."""
        if name in self._models:
            del self._models[name]
            return True
        return False

    def compare(self, names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Produces a pandas DataFrame comparison table across registered models.
        """
        target_models = (
            [self._models[n] for n in names if n in self._models]
            if names
            else list(self._models.values())
        )

        rows = []
        for m in target_models:
            row = {
                "Model Name": m.name,
                "Format": m.format,
                "Input Size": m.imgsz,
                "Size (MB)": m.size_mb,
                "Path": m.path,
                "Description": m.description,
            }
            if m.metadata:
                for k, v in m.metadata.items():
                    if isinstance(v, (int, float, str, bool)):
                        row[k] = v
            rows.append(row)

        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self._models)

    def __repr__(self) -> str:
        return f"<ModelRegistry with {len(self._models)} models: {list(self._models.keys())}>"
