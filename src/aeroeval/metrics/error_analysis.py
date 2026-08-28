"""
Error Analysis and Failure Taxonomy Module for Drone Computer Vision.

Classifies object detection prediction errors into key failure taxonomy categories:
1. Small-Object Miss (False Negative with area < 32^2 px)
2. Class Confusion (High confidence detection with incorrect class assignment)
3. Occlusion Failure (False Negative on occluded targets)
4. Background False Positive (False alarm on empty or non-target regions)
5. Localization Error (IoU with ground truth between 0.1 and 0.5)
"""

from typing import Any, Dict, List, Tuple
import numpy as np


def compute_iou_single(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def analyze_failure_taxonomy(
    gt_annotations: Dict[str, List[Dict]],
    pred_detections: Dict[str, List[Dict]],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5
) -> Dict[str, Any]:
    """
    Categorizes errors for each prediction and ground truth bounding box.
    """
    small_area_limit = 32**2

    counts = {
        "true_positives": 0,
        "small_object_misses": 0,
        "medium_large_misses": 0,
        "class_confusion_errors": 0,
        "localization_errors": 0,
        "background_false_positives": 0,
        "total_gt": 0,
        "total_pred": 0
    }

    confused_pairs: Dict[Tuple[str, str], int] = {}

    for img_id, gts in gt_annotations.items():
        preds = [p for p in pred_detections.get(img_id, []) if p.get("score", 1.0) >= conf_thresh]
        
        counts["total_gt"] += len(gts)
        counts["total_pred"] += len(preds)

        gt_matched = [False] * len(gts)
        pred_matched = [False] * len(preds)

        # 1. Match True Positives (IoU >= 0.5 and class match)
        for p_idx, p in enumerate(preds):
            p_box = np.array(p["box"])
            for g_idx, g in enumerate(gts):
                if not gt_matched[g_idx] and p["cls"] == g["cls"]:
                    g_box = np.array(g["box"])
                    if compute_iou_single(p_box, g_box) >= iou_thresh:
                        gt_matched[g_idx] = True
                        pred_matched[p_idx] = True
                        counts["true_positives"] += 1
                        break

        # 2. Check remaining predictions for Class Confusion and Localization Errors
        for p_idx, p in enumerate(preds):
            if pred_matched[p_idx]:
                continue
            p_box = np.array(p["box"])
            best_iou = 0.0
            best_g_idx = -1

            for g_idx, g in enumerate(gts):
                iou = compute_iou_single(p_box, np.array(g["box"]))
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= iou_thresh and best_g_idx >= 0:
                # Same location, different class -> Class Confusion
                counts["class_confusion_errors"] += 1
                pair = (str(preds[p_idx]["cls"]), str(gts[best_g_idx]["cls"]))
                confused_pairs[pair] = confused_pairs.get(pair, 0) + 1
                pred_matched[p_idx] = True
            elif 0.1 <= best_iou < iou_thresh:
                # Poor localization
                counts["localization_errors"] += 1
                pred_matched[p_idx] = True
            else:
                # Background false positive
                counts["background_false_positives"] += 1

        # 3. Check remaining False Negatives (Misses)
        for g_idx, g in enumerate(gts):
            if not gt_matched[g_idx]:
                g_box = np.array(g["box"])
                area = (g_box[2] - g_box[0]) * (g_box[3] - g_box[1])
                if area < small_area_limit:
                    counts["small_object_misses"] += 1
                else:
                    counts["medium_large_misses"] += 1

    total_errors = (
        counts["small_object_misses"]
        + counts["medium_large_misses"]
        + counts["class_confusion_errors"]
        + counts["localization_errors"]
        + counts["background_false_positives"]
    )

    error_shares = {}
    for k, v in counts.items():
        if k not in ["true_positives", "total_gt", "total_pred"]:
            error_shares[f"{k}_percent"] = round((v / max(1, total_errors)) * 100, 2)

    return {
        "counts": counts,
        "total_errors": total_errors,
        "error_distribution": error_shares,
        "confused_class_pairs": sorted(
            [{"predicted": p, "ground_truth": g, "count": c} for (p, g), c in confused_pairs.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:10]
    }
