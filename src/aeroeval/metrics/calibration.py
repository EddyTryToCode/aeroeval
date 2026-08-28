"""
Confidence & Calibration Analysis Module for Aerial Object Detection.

Computes:
1. Confidence Distribution (Histogram of raw confidence scores)
2. Confidence of Correct (TP) vs Incorrect (FP) Predictions
3. Threshold Sweep: Precision, Recall, F1-Score across thresholds [0.05 : 0.95]
4. Optimal Confidence Threshold (F1-maximizing Operating Point)
5. Expected Calibration Error (ECE) & Reliability Diagram
"""

import numpy as np
import pandas as pd


def compute_iou_batch(boxes1, boxes2):
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
    iarea = iw * ih

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1[:, None] + area2 - iarea
    return np.where(union > 0, iarea / union, 0.0)


def evaluate_calibration(gt_by_img: dict, preds_by_img: dict, iou_thresh: float = 0.5):
    """
    Matches predictions to ground truth to tag every prediction as True Positive (1) or False Positive (0).
    """
    total_gt = sum(len(gts) for gts in gt_by_img.values())

    matched_preds = []

    for img_name, gts in gt_by_img.items():
        preds = preds_by_img.get(img_name, [])
        if not preds:
            continue

        gt_matched = np.zeros(len(gts), dtype=bool)

        if len(gts) > 0:
            gt_boxes = np.array([g["box"] for g in gts])
            gt_classes = np.array([g["cls"] for g in gts])
            pred_boxes = np.array([p["box"] for p in preds])
            ious = compute_iou_batch(pred_boxes, gt_boxes)

            # Sort predictions descending by score
            pred_order = np.argsort(-np.array([p["score"] for p in preds]))

            for p_idx in pred_order:
                p = preds[p_idx]
                best_iou = 0.0
                best_g_idx = -1
                for g_idx in range(len(gts)):
                    if not gt_matched[g_idx] and gt_classes[g_idx] == p["cls"]:
                        if ious[p_idx, g_idx] >= iou_thresh and ious[p_idx, g_idx] > best_iou:
                            best_iou = ious[p_idx, g_idx]
                            best_g_idx = g_idx

                if best_g_idx >= 0:
                    gt_matched[best_g_idx] = True
                    matched_preds.append({
                        "score": p["score"],
                        "is_correct": 1,
                        "cls": p["cls"]
                    })
                else:
                    matched_preds.append({
                        "score": p["score"],
                        "is_correct": 0,
                        "cls": p["cls"]
                    })
        else:
            for p in preds:
                matched_preds.append({
                    "score": p["score"],
                    "is_correct": 0,
                    "cls": p["cls"]
                })

    df_preds = pd.DataFrame(matched_preds)

    # 1. Threshold Sweep
    thresholds = np.arange(0.05, 0.96, 0.05)
    sweep_results = []

    for th in thresholds:
        sub = df_preds[df_preds["score"] >= th]
        tp = (sub["is_correct"] == 1).sum()
        fp = (sub["is_correct"] == 0).sum()
        fn = total_gt - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / total_gt if total_gt > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        sweep_results.append({
            "Threshold": round(th, 2),
            "Precision": round(p, 3),
            "Recall": round(r, 3),
            "F1_Score": round(f1, 3),
            "True_Positives": int(tp),
            "False_Positives": int(fp),
            "False_Negatives": int(fn)
        })

    df_sweep = pd.DataFrame(sweep_results)
    best_row = df_sweep.loc[df_sweep["F1_Score"].idxmax()]

    # 2. Expected Calibration Error (ECE) calculation with 10 bins
    bins = np.linspace(0.0, 1.0, 11)
    df_preds["bin"] = pd.cut(df_preds["score"], bins, include_lowest=True)

    ece = 0.0
    reliability_data = []
    total_n = len(df_preds)

    for bin_interval, group in df_preds.groupby("bin", observed=False):
        n_b = len(group)
        if n_b > 0:
            avg_conf = group["score"].mean()
            acc = group["is_correct"].mean()
            ece += (n_b / total_n) * abs(acc - avg_conf)
            reliability_data.append({
                "Bin": str(bin_interval),
                "Avg_Confidence": round(float(avg_conf), 3),
                "Accuracy": round(float(acc), 3),
                "Count": int(n_b)
            })
        else:
            mid = (bin_interval.left + bin_interval.right) / 2.0
            reliability_data.append({
                "Bin": str(bin_interval),
                "Avg_Confidence": round(float(mid), 3),
                "Accuracy": 0.0,
                "Count": 0
            })

    return {
        "df_predictions": df_preds,
        "df_sweep": df_sweep,
        "best_operating_point": {
            "optimal_threshold": float(best_row["Threshold"]),
            "max_f1": float(best_row["F1_Score"]),
            "precision_at_optimal": float(best_row["Precision"]),
            "recall_at_optimal": float(best_row["Recall"])
        },
        "ece": round(float(ece), 4),
        "reliability_table": pd.DataFrame(reliability_data)
    }
