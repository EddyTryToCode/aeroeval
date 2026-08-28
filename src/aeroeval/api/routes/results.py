"""
Results and Report Retrieval Routes for AeroEval API.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/results", tags=["Results"])

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
REPORTS_DIR = ROOT_DIR / "reports"


@router.get("/{run_id}")
def get_run_results(run_id: str):
    """Retrieves JSON summary metrics for a specific evaluation run."""
    run_dir = REPORTS_DIR / run_id
    if not run_dir.exists():
        # Check direct fallback
        if run_id == "latest":
            run_dir = REPORTS_DIR / "run_latest"
        elif run_id == "baseline":
            run_dir = REPORTS_DIR / "run_baseline"
        else:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in reports.")

    json_file = run_dir / "summary.json"
    if not json_file.exists():
        raise HTTPException(status_code=404, detail=f"No summary.json found in run '{run_id}'.")

    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/{run_id}/report", response_class=HTMLResponse)
def get_run_html_report(run_id: str):
    """Serves the standalone interactive HTML evaluation report."""
    run_dir = REPORTS_DIR / run_id
    if not run_dir.exists():
        if run_id == "latest":
            run_dir = REPORTS_DIR / "run_latest"
        elif run_id == "baseline":
            run_dir = REPORTS_DIR / "run_baseline"
        else:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    html_file = run_dir / "evaluation_report.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail=f"HTML report not found for run '{run_id}'.")

    with open(html_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)
