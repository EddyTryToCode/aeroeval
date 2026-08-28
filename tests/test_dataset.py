"""
Unit tests for VisDrone dataset parsing and YOLO coordinate conversion.
"""

import numpy as np


def convert_visdrone_bbox_to_yolo(bbox_left, bbox_top, bbox_width, bbox_height, img_w, img_h):
    """Normalized YOLO bbox [x_center, y_center, w, h] from VisDrone format."""
    x_center = (bbox_left + bbox_width / 2.0) / img_w
    y_center = (bbox_top + bbox_height / 2.0) / img_h
    w = bbox_width / img_w
    h = bbox_height / img_h
    return (
        np.clip(x_center, 0.0, 1.0),
        np.clip(y_center, 0.0, 1.0),
        np.clip(w, 0.0, 1.0),
        np.clip(h, 0.0, 1.0)
    )


def test_visdrone_to_yolo_conversion_center():
    # Box in center of 1000x1000 image
    # left=400, top=400, w=200, h=200 -> center=(500, 500) -> norm=(0.5, 0.5, 0.2, 0.2)
    xc, yc, w, h = convert_visdrone_bbox_to_yolo(400, 400, 200, 200, 1000, 1000)
    assert np.isclose(xc, 0.5)
    assert np.isclose(yc, 0.5)
    assert np.isclose(w, 0.2)
    assert np.isclose(h, 0.2)


def test_visdrone_to_yolo_conversion_bounds():
    # Box at corner
    xc, yc, w, h = convert_visdrone_bbox_to_yolo(0, 0, 100, 100, 1000, 1000)
    assert np.isclose(xc, 0.05)
    assert np.isclose(yc, 0.05)
    assert np.isclose(w, 0.1)
    assert np.isclose(h, 0.1)
