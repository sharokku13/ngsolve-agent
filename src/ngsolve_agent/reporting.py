"""Write stable machine-readable records for each experiment run."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import ngsolve

from .schemas import MetricsSpec, TaskSpec


def write_report(
    task: TaskSpec,
    metrics: MetricsSpec,
    failures: list[str],
    output_dir: Path,
    image_path: Path,
    run_id: str,
    architecture: str = "B0",
) -> Path:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{run_id}.json"
    report = {
        "experiment_id": run_id,
        "architecture": architecture,
        "task": task.model_dump(mode="json"),
        "metrics": metrics.model_dump(mode="json"),
        "checks": {"passed": not failures, "failures": failures},
        "artifacts": {"solution_plot": str(image_path.relative_to(output_dir))},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ngsolve": getattr(ngsolve, "__version__", "unknown"),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path
