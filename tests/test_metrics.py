"""
Unit tests for AeroEval Metrics.
"""

import numpy as np

from aeroeval.metrics.detection import (
    SIZE_THRESHOLDS,
    box_area,
    compute_iou_matrix,
    evaluate_by_object_size,
)
from aeroeval.metrics.error_analysis import analyze_failure_taxonomy
from aeroeval.metrics.tracking import evaluate_tracking_sequence


def test_compute_iou_matrix_perfect_overlap():
    b1 = np.array([[10, 10, 50, 50]])
    b2 = np.array([[10, 10, 50, 50]])
    iou = compute_iou_matrix(b1, b2)
    assert np.isclose(iou[0, 0], 1.0)


def test_compute_iou_matrix_no_overlap():
    b1 = np.array([[0, 0, 10, 10]])
    b2 = np.array([[20, 20, 30, 30]])
    iou = compute_iou_matrix(b1, b2)
    assert np.isclose(iou[0, 0], 0.0)


def test_compute_iou_matrix_partial_overlap():
    b1 = np.array([[0, 0, 20, 20]])  # area 400
    b2 = np.array([[10, 0, 30, 20]]) # area 400, intersection 10x20 = 200, union = 600
    iou = compute_iou_matrix(b1, b2)
    assert np.isclose(iou[0, 0], 200.0 / 600.0)


def test_box_area():
    box = np.array([10, 20, 30, 60]) # w=20, h=40 -> 800
    assert box_area(box) == 800.0


def test_size_thresholds():
    assert SIZE_THRESHOLDS["small"] == (0, 1024)
    assert SIZE_THRESHOLDS["medium"] == (1024, 9216)
    assert SIZE_THRESHOLDS["large"] == (9216, float("inf"))


def test_evaluate_by_object_size_synthetic():
    gt = {
        "img1": [
            {"box": [0, 0, 20, 20], "cls": 0},   # small: area 400
            {"box": [0, 0, 50, 50], "cls": 1},   # medium: area 2500
            {"box": [0, 0, 150, 150], "cls": 2}, # large: area 22500
        ]
    }
    pred = {
        "img1": [
            {"box": [0, 0, 20, 20], "score": 0.9, "cls": 0},
            {"box": [0, 0, 50, 50], "score": 0.85, "cls": 1},
        ]
    }
    res = evaluate_by_object_size(gt, pred)
    assert res["small"]["true_positives_50"] == 1
    assert res["medium"]["true_positives_50"] == 1
    assert res["large"]["true_positives_50"] == 0
    assert res["small"]["recall_50"] == 1.0
    assert res["large"]["recall_50"] == 0.0


def test_failure_taxonomy_synthetic():
    gt = {
        "img1": [
            {"box": [0, 0, 20, 20], "cls": 0}, # small target
            {"box": [50, 50, 100, 100], "cls": 1}, # medium target
        ]
    }
    pred = {
        "img1": [
            {"box": [50, 50, 100, 100], "score": 0.9, "cls": 2}, # class confusion (predicted 2 instead of 1)
        ]
    }
    res = analyze_failure_taxonomy(gt, pred)
    assert res["counts"]["small_object_misses"] == 1
    assert res["counts"]["class_confusion_errors"] == 1
    assert res["counts"]["true_positives"] == 0


def test_tracking_evaluator_synthetic():
    gt = {
        1: [{"id": 1, "box": [10, 10, 50, 50], "cls": 0}],
        2: [{"id": 1, "box": [12, 12, 52, 52], "cls": 0}],
    }
    pred = {
        1: [{"id": 1, "box": [10, 10, 50, 50], "cls": 0}],
        2: [{"id": 1, "box": [12, 12, 52, 52], "cls": 0}],
    }
    res = evaluate_tracking_sequence(gt, pred)
    assert res["MOTA"] == 1.0
    assert res["IDF1"] == 1.0
    assert res["ID_Switches"] == 0
