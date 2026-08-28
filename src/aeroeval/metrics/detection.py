"""
Detection Metrics and Scale-Stratified (Small/Medium/Large) Evaluation Module.

Computes:
1. Standard Object Detection Metrics: mAP50, mAP50-95, Precision, Recall, F1
2. Per-Class Average Precision (AP50 and AP50-95) across all classes
3. Scale-Stratified Object Detection Metrics:
   - Small objects:  area < 32^2 px (1024 px^2)
   - Medium objects: 32^2 <= area < 96^2 px (1024 to 9216 px^2)
   - Large objects:  area >= 96^2 px (9216 px^2)
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from ultralytics import YOLO

# Standard VisDrone class mapping
VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

SIZE_THRESHOLDS = {
    "small": (0, 32**2),            # area < 1024 px²
    "medium": (32**2, 96**2),       # 1024 ≤ area < 9216 px²
    "large": (96**2, float("inf")), # area ≥ 9216 px²
}


def box_area(box: np.ndarray) -> float:
    """Calculates box pixel area for [x1, y1, x2, y2]."""
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Vectorized pairwise IoU calculation between N boxes and M boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    ix1 = np.maximum(b1_x1[:, None], b2_x1)
    iy1 = np.maximum(b1_y1[:, None], b2_y1)
    ix2 = np.minimum(b1_x2[:, None], b2_x2)
    iy2 = np.minimum(b1_y2[:, None], b2_y2)

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter_area = iw * ih

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1[:, None] + area2 - inter_area
    return np.where(union_area > 0, inter_area / union_area, 0.0)


def evaluate_detection_model(
    model: Union[str, Path, YOLO],
    data_yaml: Union[str, Path] = "configs/visdrone.yaml",
    imgsz: int = 640,
    device: str = "0",
    split: str = "val"
) -> Dict[str, Any]:
    """
    Runs standard Ultralytics validation and returns structured detection metrics.
    """
    if isinstance(model, (str, Path)):
        model_obj = YOLO(str(model))
    else:
        model_obj = model

    val_res = model_obj.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        split=split,
        verbose=False
    )

    box_metrics = val_res.box
    map50 = float(box_metrics.map50)
    map50_95 = float(box_metrics.map)
    mp = float(box_metrics.mp)
    mr = float(box_metrics.mr)
    f1 = 2 * (mp * mr) / (mp + mr + 1e-6)

    # Per-class metrics
    per_class = {}
    class_names = val_res.names if hasattr(val_res, "names") else {i: name for i, name in enumerate(VISDRONE_CLASSES)}
    maps_per_class = box_metrics.maps if hasattr(box_metrics, "maps") and box_metrics.maps is not None else []

    for cls_id, cls_name in class_names.items():
        if cls_id < len(maps_per_class):
            per_class[cls_name] = {
                "class_id": int(cls_id),
                "ap50_95": round(float(maps_per_class[cls_id]), 4),
            }

    return {
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision": round(mp, 4),
        "recall": round(mr, 4),
        "f1_score": round(f1, 4),
        "per_class_metrics": per_class
    }


def evaluate_by_object_size(
    gt_annotations: Dict[str, List[Dict]],
    pred_detections: Dict[str, List[Dict]],
    iou_thresholds: List[float] = [0.5, 0.75]
) -> Dict[str, Any]:
    """
    Evaluates Precision, Recall, and mAP partitioned strictly by object size categories (Small, Medium, Large).
    """
    size_results = {}

    for size_cat, (min_area, max_area) in SIZE_THRESHOLDS.items():
        total_gt_count = 0
        matched_tp_50 = 0
        total_pred_count = 0

        for img_id, gts in gt_annotations.items():
            preds = pred_detections.get(img_id, [])

            filtered_gts = [g for g in gts if min_area <= box_area(np.array(g["box"])) < max_area]
            filtered_preds = [p for p in preds if min_area <= box_area(np.array(p["box"])) < max_area]

            total_gt_count += len(filtered_gts)
            total_pred_count += len(filtered_preds)

            if len(filtered_gts) > 0 and len(filtered_preds) > 0:
                gt_boxes = np.array([g["box"] for g in filtered_gts])
                pred_boxes = np.array([p["box"] for p in filtered_preds])
                ious = compute_iou_matrix(pred_boxes, gt_boxes)

                gt_matched = np.zeros(len(filtered_gts), dtype=bool)
                pred_order = np.argsort(-np.array([p["score"] for p in filtered_preds]))

                for p_idx in pred_order:
                    p = filtered_preds[p_idx]
                    best_iou = 0.0
                    best_g_idx = -1
                    for g_idx in range(len(filtered_gts)):
                        if not gt_matched[g_idx] and filtered_gts[g_idx]["cls"] == p["cls"]:
                            if ious[p_idx, g_idx] > best_iou:
                                best_iou = ious[p_idx, g_idx]
                                best_g_idx = g_idx
                    if best_iou >= 0.5 and best_g_idx >= 0:
                        gt_matched[best_g_idx] = True
                        matched_tp_50 += 1

        rec = (matched_tp_50 / max(1, total_gt_count)) if total_gt_count > 0 else 0.0
        prec = (matched_tp_50 / max(1, total_pred_count)) if total_pred_count > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec + 1e-6)

        size_results[size_cat] = {
            "gt_objects": total_gt_count,
            "pred_objects": total_pred_count,
            "true_positives_50": matched_tp_50,
            "precision_50": round(prec, 4),
            "recall_50": round(rec, 4),
            "f1_score_50": round(f1, 4),
        }

    return size_results
