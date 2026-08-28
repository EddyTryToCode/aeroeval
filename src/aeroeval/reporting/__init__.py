"""
AeroEval Reporting Subpackage.

Provides HTML/JSON report generation and Model Recommendation Engine.
"""

from aeroeval.reporting.recommendation import DEPLOYMENT_PROFILES, ModelRecommendationEngine
from aeroeval.reporting.report import EvaluationReport

__all__ = [
    "EvaluationReport",
    "ModelRecommendationEngine",
    "DEPLOYMENT_PROFILES",
]
