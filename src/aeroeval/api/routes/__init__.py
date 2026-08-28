"""
AeroEval API Routes Subpackage.
"""

from aeroeval.api.routes.evaluate import router as evaluate_router
from aeroeval.api.routes.models import router as models_router
from aeroeval.api.routes.results import router as results_router

__all__ = ["models_router", "evaluate_router", "results_router"]
