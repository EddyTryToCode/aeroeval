"""
Unified Model Runner Module for PyTorch and ONNX Models.

Provides a unified interface for inference across different backends.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class ModelRunner:
    """
    Unified inference abstraction supporting PyTorch (.pt) and ONNX (.onnx) backends.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        imgsz: int = 640,
        device: str = "0",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45
    ):
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.device = str(device)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.is_onnx = self.model_path.suffix.lower() == ".onnx"
        self.is_cuda = (self.device not in ["cpu", "-1"]) and torch.cuda.is_available()

        if self.is_onnx:
            self.model = YOLO(str(self.model_path), task="detect")
        else:
            self.model = YOLO(str(self.model_path))

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs inference on a single image and returns a list of normalized bounding box dictionaries:
        [
            {
                "box": [x1, y1, x2, y2],  # pixel coordinates
                "score": 0.85,
                "cls": 0,
                "class_name": "pedestrian"
            }
        ]
        """
        conf_val = conf if conf is not None else self.conf_thresh
        iou_val = iou if iou is not None else self.iou_thresh

        res = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            device=self.device,
            conf=conf_val,
            iou=iou_val,
            verbose=False
        )[0]

        detections = []
        if len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            names = res.names

            for b, s, c in zip(boxes, scores, classes):
                detections.append({
                    "box": b.tolist(),
                    "score": float(s),
                    "cls": int(c),
                    "class_name": names.get(int(c), str(c))
                })

        return detections

    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        conf: Optional[float] = None
    ) -> List[List[Dict[str, Any]]]:
        """Runs batch inference."""
        return [self.predict(img, conf=conf) for img in images]
