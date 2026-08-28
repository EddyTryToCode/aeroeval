"""
Experiment Logger & Reproducibility Tracker for AeroEval.
"""

import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import torch


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return commit.decode("ascii").strip()
    except Exception:
        return "unknown"


def get_hardware_info() -> str:
    """Identifies primary compute device (GPU model / CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


class ExperimentLogger:
    """
    Structured logger recording full experimental metadata for 100% reproducibility.
    """

    def __init__(self, experiment_id: str, output_dir: Path = Path("experiments")):
        self.experiment_id = experiment_id
        self.output_dir = output_dir / experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.datetime.now()

    def log(
        self,
        model: str,
        dataset: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        seed: int = 42
    ) -> Path:
        """
        Saves experiment artifact JSON.
        """
        end_time = datetime.datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        record = {
            "experiment_id": self.experiment_id,
            "timestamp": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "dataset": dataset,
            "git_commit": get_git_commit_hash(),
            "seed": seed,
            "hardware": get_hardware_info(),
            "parameters": parameters,
            "metrics": metrics,
            "duration_seconds": round(duration, 2)
        }

        out_file = self.output_dir / "experiment_record.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return out_file
