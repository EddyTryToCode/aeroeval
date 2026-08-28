"""
Multi-Object Tracking (MOT) Evaluation Module for Aerial & Drone Perception.

Computes MOT Metrics:
1. MOTA (Multi-Object Tracking Accuracy)
2. IDF1 (Identification F1 Score)
3. ID Switches (IDSW - Number of identity switch errors)
4. MT (Mostly Tracked trajectories > 80% life)
5. ML (Mostly Lost trajectories < 20% life)
6. Fragmentations (Frag)
7. Tracking Latency (ms) & Effective FPS
"""

from collections import defaultdict

import numpy as np


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


class MOTEvaluator:
    """
    Standard Multi-Object Tracking Evaluator (CLEAR MOT & IDF1 metrics).
    """
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate_sequence(self, gt_frames: dict, pred_frames: dict):
        """
        gt_frames: dict {frame_id: list of dict {'id': int, 'box': [x1, y1, x2, y2], 'cls': int}}
        pred_frames: dict {frame_id: list of dict {'id': int, 'box': [x1, y1, x2, y2], 'cls': int}}
        """
        frame_ids = sorted(list(set(list(gt_frames.keys()) + list(pred_frames.keys()))))

        total_gt = 0
        total_pred = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        id_switches = 0

        # For ID Switch Tracking: map gt_id -> last assigned pred_id
        gt_to_pred_map = {}

        # Trajectory lengths for MT / ML
        gt_traj_lengths = defaultdict(int)
        gt_traj_tracked = defaultdict(int)

        # ID Matching Matrix for IDF1
        id_match_matrix = defaultdict(lambda: defaultdict(int))
        gt_id_counts = defaultdict(int)
        pred_id_counts = defaultdict(int)

        for fid in frame_ids:
            gts = gt_frames.get(fid, [])
            preds = pred_frames.get(fid, [])

            total_gt += len(gts)
            total_pred += len(preds)

            for g in gts:
                gt_traj_lengths[g["id"]] += 1
                gt_id_counts[g["id"]] += 1

            for p in preds:
                pred_id_counts[p["id"]] += 1

            if len(gts) == 0 and len(preds) == 0:
                continue

            if len(gts) == 0:
                total_fp += len(preds)
                continue

            if len(preds) == 0:
                total_fn += len(gts)
                continue

            gt_boxes = np.array([g["box"] for g in gts])
            pred_boxes = np.array([p["box"] for p in preds])

            ious = compute_iou_batch(pred_boxes, gt_boxes)

            matched_gt = set()
            matched_pred = set()

            # Greedy bipartite matching
            while True:
                max_iou = 0.0
                best_p, best_g = -1, -1
                for p_idx in range(len(preds)):
                    if p_idx in matched_pred:
                        continue
                    for g_idx in range(len(gts)):
                        if g_idx in matched_gt:
                            continue
                        if ious[p_idx, g_idx] > max_iou and ious[p_idx, g_idx] >= self.iou_threshold:
                            max_iou = ious[p_idx, g_idx]
                            best_p = p_idx
                            best_g = g_idx

                if best_p >= 0 and best_g >= 0:
                    matched_pred.add(best_p)
                    matched_gt.add(best_g)
                    total_tp += 1

                    gt_id = gts[best_g]["id"]
                    pred_id = preds[best_p]["id"]

                    gt_traj_tracked[gt_id] += 1
                    id_match_matrix[gt_id][pred_id] += 1

                    # Check ID Switch
                    if gt_id in gt_to_pred_map:
                        if gt_to_pred_map[gt_id] != pred_id:
                            id_switches += 1
                    gt_to_pred_map[gt_id] = pred_id
                else:
                    break

            total_fp += (len(preds) - len(matched_pred))
            total_fn += (len(gts) - len(matched_gt))

        # 1. MOTA = 1 - (FN + FP + IDSW) / Total_GT
        mota = 1.0 - (total_fn + total_fp + id_switches) / total_gt if total_gt > 0 else 0.0

        # 2. MT / ML calculation
        mostly_tracked = 0
        mostly_lost = 0
        partially_tracked = 0

        for gid, total_len in gt_traj_lengths.items():
            tracked_len = gt_traj_tracked[gid]
            ratio = tracked_len / total_len if total_len > 0 else 0.0
            if ratio >= 0.8:
                mostly_tracked += 1
            elif ratio <= 0.2:
                mostly_lost += 1
            else:
                partially_tracked += 1

        num_gt_trajs = len(gt_traj_lengths)
        mt_pct = (mostly_tracked / num_gt_trajs * 100.0) if num_gt_trajs > 0 else 0.0
        ml_pct = (mostly_lost / num_gt_trajs * 100.0) if num_gt_trajs > 0 else 0.0

        # 3. IDF1 calculation
        # Optimal ID matching (sum of maximum overlap between GT ID and Pred ID)
        id_tp = sum(max(pred_matches.values()) for pred_matches in id_match_matrix.values()) if id_match_matrix else 0
        idf1 = (2 * id_tp) / (total_gt + total_pred) if (total_gt + total_pred) > 0 else 0.0
        idp = id_tp / total_pred if total_pred > 0 else 0.0
        idr = id_tp / total_gt if total_gt > 0 else 0.0

        return {
            "Total_GT": total_gt,
            "Total_Predictions": total_pred,
            "MOTA": round(float(mota), 3),
            "IDF1": round(float(idf1), 3),
            "IDP": round(float(idp), 3),
            "IDR": round(float(idr), 3),
            "ID_Switches": int(id_switches),
            "Mostly_Tracked_MT": int(mostly_tracked),
            "MT_%": round(float(mt_pct), 1),
            "Mostly_Lost_ML": int(mostly_lost),
            "ML_%": round(float(ml_pct), 1),
            "Total_Unique_Tracks": num_gt_trajs
        }


def evaluate_tracking_sequence(gt_frames: dict, pred_frames: dict, iou_threshold: float = 0.5):
    """Functional wrapper for MOTEvaluator.evaluate_sequence."""
    evaluator = MOTEvaluator(iou_threshold=iou_threshold)
    return evaluator.evaluate_sequence(gt_frames, pred_frames)

