"""Narrow API exposed to statically validated model-generated drivers."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .reporting import write_report
from .schemas import parse_task
from .task_registry import evaluate_task, solve_task


def run(task_id: str, **parameters: Any) -> dict[str, Any]:
    """Run one allowlisted task and print a machine-readable result as the last line."""

    task_data = {"task_id": task_id, **parameters}
    task = parse_task(task_data)
    run_id = os.environ.get(
        "NGSOLVE_D3_RUN_ID",
        datetime.now(UTC).strftime("D3-%Y%m%dT%H%M%S%fZ"),
    )
    output_dir = Path(os.environ.get("NGSOLVE_D3_OUTPUT_DIR", "results/d3"))
    solver_dir = output_dir / "solver"
    metrics, image_path = solve_task(task, solver_dir)
    failures = evaluate_task(task, metrics.model_dump(mode="json"))
    report_path = write_report(
        task,
        metrics,
        failures,
        output_dir,
        image_path,
        run_id,
        architecture="D3",
    )
    result: dict[str, Any] = {
        "status": "passed" if not failures else "verification_failed",
        "task": task.model_dump(mode="json"),
        "metrics": metrics.model_dump(mode="json"),
        "image_path": str(image_path),
        "report_path": str(report_path),
        "run_id": run_id,
        "verification_failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result
