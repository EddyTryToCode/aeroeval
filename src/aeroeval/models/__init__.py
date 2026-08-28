"""
AeroEval Models Subpackage.

Provides model registration and unified inference execution.
"""

from aeroeval.models.registry import ModelInfo, ModelRegistry
from aeroeval.models.runner import ModelRunner

__all__ = [
    "ModelInfo",
    "ModelRegistry",
    "ModelRunner",
]
