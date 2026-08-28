"""
Pytest configuration and shared fixtures for AeroEval.
"""

import pytest


@pytest.fixture
def mock_detection_boxes():
    return {
        "img_001": [
            {"box": [10, 10, 50, 50], "score": 0.88, "cls": 0, "class_name": "pedestrian"},
            {"box": [100, 100, 200, 200], "score": 0.92, "cls": 3, "class_name": "car"},
        ]
    }


@pytest.fixture
def mock_ground_truth_boxes():
    return {
        "img_001": [
            {"box": [10, 10, 50, 50], "cls": 0, "class_name": "pedestrian"},
            {"box": [100, 100, 200, 200], "cls": 3, "class_name": "car"},
        ]
    }
