"""
Evaluation and Benchmarking Routes for AeroEval API.
"""

import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from aeroeval.api.dependencies import get_model_registry
from aeroeval.api.schemas import BenchmarkRequest, EvaluateRequest
from aeroeval.metrics.efficiency import benchmark_model_efficiency
from aeroeval.models.registry import ModelRegistry
from aeroeval.pipeline.evaluate import EvaluationPipeline

router = APIRouter(tags=["Evaluation"])


@router.post("/evaluate")
def run_evaluation(
    req: EvaluateRequest,
    registry: ModelRegistry = Depends(get_model_registry)
) -> Dict[str, Any]:
    """
    Triggers the full multi-modal AI evaluation pipeline on a model.
    """
    model_entry = registry.get(req.model)
    model_path = model_entry.path if model_entry else req.model

    if not Path(model_path).exists():
        raise HTTPException(status_code=400, detail=f"Model weight file '{model_path}' does not exist.")

    run_id = f"run_{int(time.time())}"
    out_dir = Path(req.output_dir) if req.output_dir else Path(f"reports/{run_id}")

    try:
        pipeline = EvaluationPipeline(
            model_path=model_path,
            data_yaml=req.dataset,
            imgsz=req.imgsz,
            device=req.device,
            output_dir=out_dir,
            profile=req.profile
        )
        results = pipeline.run()
        results["run_id"] = run_id
        results["report_url"] = f"/results/{run_id}/report"
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/benchmark")
def run_benchmark(
    req: BenchmarkRequest,
    registry: ModelRegistry = Depends(get_model_registry)
) -> Dict[str, Any]:
    """
    Runs real-time latency and efficiency benchmark on the specified model.
    """
    model_entry = registry.get(req.model)
    model_path = model_entry.path if model_entry else req.model

    if not Path(model_path).exists():
        raise HTTPException(status_code=400, detail=f"Model weight file '{model_path}' not found.")

    try:
        res = benchmark_model_efficiency(
            model_path=model_path,
            imgsz=req.imgsz,
            device=req.device,
            warmup=req.warmup,
            iterations=req.iterations
        )
        _ = res.pop("time_series", None)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark profiling failed: {str(e)}")
