"""
Stage 17: Model Recommendation Engine with Deployment Profiles.

Evaluates candidate UAV vision models across multiple operational profiles:
- real_time_uav: Optimized for high FPS, low latency, and robust field operation.
- high_accuracy: Optimized for maximum detection precision and small object recall.
- edge_device:   Optimized for low VRAM, small memory footprint, and low power envelope.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


DEPLOYMENT_PROFILES = {
    "real_time_uav": {
        "description": "Balanced high-speed UAV tracking with robustness to camera motion and blur",
        "weights": {
            "accuracy": 0.30,
            "latency": 0.30,
            "robustness": 0.25,
            "memory": 0.15,
        }
    },
    "high_accuracy": {
        "description": "High-altitude reconnaissance prioritizing small-object precision and recall",
        "weights": {
            "accuracy": 0.50,
            "latency": 0.15,
            "robustness": 0.25,
            "memory": 0.10,
        }
    },
    "edge_device": {
        "description": "Low-power companion microcontrollers / Jetson Nano with strict memory limits",
        "weights": {
            "accuracy": 0.20,
            "latency": 0.25,
            "robustness": 0.15,
            "memory": 0.40,
        }
    },
}


class ModelRecommendationEngine:
    """
    Multi-Criteria Decision Analysis (MCDA) Recommendation Engine for Drone AI.
    """

    def __init__(self, profiles: Optional[Dict[str, Any]] = None):
        self.profiles = profiles or DEPLOYMENT_PROFILES

    def recommend(
        self,
        candidate_models: List[Dict[str, Any]],
        profile_name: str = "real_time_uav",
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Ranks candidate models and selects the optimal recommendation for a deployment profile.

        candidate_models format expected:
        [
            {
                "name": "yolo11n-640",
                "accuracy": 42.3,   # mAP50 or mAP50-95
                "latency_ms": 11.1, # E2E latency in ms
                "robustness": 78.5, # Robustness score / retention %
                "memory_mb": 5.2,   # Model size or peak memory MB
            }, ...
        ]
        """
        if not candidate_models:
            return {"error": "No candidate models provided"}

        if profile_name not in self.profiles and not custom_weights:
            profile_name = "real_time_uav"

        weights = custom_weights or self.profiles[profile_name]["weights"]

        # Extract metric arrays
        accuracies = np.array([m.get("accuracy", 0.0) for m in candidate_models], dtype=float)
        latencies = np.array([m.get("latency_ms", 1.0) for m in candidate_models], dtype=float)
        robustnesses = np.array([m.get("robustness", 50.0) for m in candidate_models], dtype=float)
        memories = np.array([m.get("memory_mb", 10.0) for m in candidate_models], dtype=float)

        # Min-Max Normalization to [0, 1]
        def norm_higher_better(arr):
            min_v, max_v = np.min(arr), np.max(arr)
            if max_v == min_v:
                return np.ones_like(arr)
            return (arr - min_v) / (max_v - min_v)

        def norm_lower_better(arr):
            min_v, max_v = np.min(arr), np.max(arr)
            if max_v == min_v:
                return np.ones_like(arr)
            return (max_v - arr) / (max_v - min_v)

        norm_acc = norm_higher_better(accuracies)
        norm_lat = norm_lower_better(latencies)
        norm_rob = norm_higher_better(robustnesses)
        norm_mem = norm_lower_better(memories)

        # Compute Composite Weighted Scores
        rankings = []
        for i, m in enumerate(candidate_models):
            score = (
                weights["accuracy"] * norm_acc[i]
                + weights["latency"] * norm_lat[i]
                + weights["robustness"] * norm_rob[i]
                + weights["memory"] * norm_mem[i]
            )
            rankings.append({
                "model": m["name"],
                "score": round(float(score), 4),
                "accuracy": round(float(m.get("accuracy", 0.0)), 2),
                "latency_ms": round(float(m.get("latency_ms", 0.0)), 2),
                "fps": round(1000.0 / max(0.1, float(m.get("latency_ms", 1.0))), 1),
                "robustness_retention": round(float(m.get("robustness", 0.0)), 1),
                "memory_mb": round(float(m.get("memory_mb", 0.0)), 2),
                "norm_accuracy": round(float(norm_acc[i]), 3),
                "norm_latency": round(float(norm_lat[i]), 3),
                "norm_robustness": round(float(norm_rob[i]), 3),
                "norm_memory": round(float(norm_mem[i]), 3)
            })

        # Sort descending by score
        rankings.sort(key=lambda x: x["score"], reverse=True)
        best = rankings[0]

        # Generate intelligent natural language justification
        justification = (
            f"Model '{best['model']}' ranks #1 for profile '{profile_name}' (Composite Score: {best['score']:.3f}). "
            f"It delivers {best['accuracy']}% accuracy at {best['latency_ms']} ms latency ({best['fps']} FPS), "
            f"maintaining {best['robustness_retention']}% robustness under environmental corruptions "
            f"with a compact footprint of {best['memory_mb']} MB."
        )

        return {
            "profile": profile_name,
            "profile_description": self.profiles.get(profile_name, {}).get("description", ""),
            "weights": weights,
            "recommended_model": best["model"],
            "best_score": best["score"],
            "justification": justification,
            "rankings": rankings
        }

    def recommend_all_profiles(
        self,
        candidate_models: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Runs recommendation analysis across all predefined deployment profiles."""
        results = {}
        for p in self.profiles.keys():
            results[p] = self.recommend(candidate_models, profile_name=p)
        return results

    def export_summary(
        self,
        recommendation_results: Dict[str, Any],
        output_dir: Union[str, Path] = "reports"
    ) -> Dict[str, Path]:
        """Exports recommendations into JSON, Markdown, and CSV formats."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON Export
        json_path = out_dir / "recommendation_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(recommendation_results, f, indent=2)

        # Markdown Export
        md_lines = ["# UAV Model Deployment Recommendations\n"]
        for prof, res in recommendation_results.items():
            md_lines.append(f"## Profile: `{prof}`")
            md_lines.append(f"> **Description**: {res.get('profile_description', '')}\n")
            md_lines.append(f"- **Recommended Choice**: **`{res['recommended_model']}`** (Score: `{res['best_score']}`)")
            md_lines.append(f"- **Justification**: {res['justification']}\n")
            md_lines.append("| Rank | Model | Score | Accuracy | Latency (ms) | FPS | Robustness (%) | Memory (MB) |")
            md_lines.append("|---|---|---|---|---|---|---|---|")
            for r_idx, r in enumerate(res["rankings"], start=1):
                md_lines.append(
                    f"| {r_idx} | **{r['model']}** | {r['score']:.3f} | {r['accuracy']} | {r['latency_ms']} ms | {r['fps']} | {r['robustness_retention']}% | {r['memory_mb']} MB |"
                )
            md_lines.append("\n---\n")

        md_path = out_dir / "recommendation_summary.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        return {"json": json_path, "md": md_path}
