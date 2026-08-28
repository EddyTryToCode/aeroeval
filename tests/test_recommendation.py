"""
Unit tests for ModelRecommendationEngine.
"""

import pytest

from aeroeval.reporting.recommendation import DEPLOYMENT_PROFILES, ModelRecommendationEngine


@pytest.fixture
def candidates():
    return [
        {"name": "Model_Fast", "accuracy": 35.0, "latency_ms": 10.0, "robustness": 75.0, "memory_mb": 5.0},
        {"name": "Model_Accurate", "accuracy": 50.0, "latency_ms": 40.0, "robustness": 90.0, "memory_mb": 40.0},
        {"name": "Model_Balanced", "accuracy": 45.0, "latency_ms": 20.0, "robustness": 85.0, "memory_mb": 15.0},
    ]


def test_recommendation_profiles_exist():
    assert "real_time_uav" in DEPLOYMENT_PROFILES
    assert "high_accuracy" in DEPLOYMENT_PROFILES
    assert "edge_device" in DEPLOYMENT_PROFILES


def test_recommendation_engine_runs(candidates):
    engine = ModelRecommendationEngine()
    res = engine.recommend(candidates, profile_name="real_time_uav")
    assert "recommended_model" in res
    assert "rankings" in res
    assert len(res["rankings"]) == 3
    assert res["best_score"] >= 0.0


def test_recommendation_all_profiles(candidates):
    engine = ModelRecommendationEngine()
    res = engine.recommend_all_profiles(candidates)
    assert "real_time_uav" in res
    assert "high_accuracy" in res
    assert "edge_device" in res
