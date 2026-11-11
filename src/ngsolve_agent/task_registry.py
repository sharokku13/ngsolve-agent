"""Allowlisted task, solver, and independent evaluator registry."""

from __future__ import annotations

from pathlib import Path

from .fem_solver import solve_poisson
from .schemas import MetricsSpec, PoissonTask, StokesTask, TaskSpec, parse_metrics, parse_task
from .stokes_solver import solve_stokes
from .validation import validate_poisson_result, validate_stokes_result


def solve_task(task: TaskSpec, output_dir: Path) -> tuple[MetricsSpec, Path]:
    """Dispatch validated task data to its fixed NGSolve implementation."""

    if isinstance(task, StokesTask):
        return solve_stokes(task, output_dir)
    if isinstance(task, PoissonTask):
        return solve_poisson(task, output_dir)
    raise TypeError(f"no solver registered for {type(task).__name__}")


def evaluate_task(task: TaskSpec, metrics_data: dict[str, object]) -> list[str]:
    """Parse matching metrics and run the task-specific independent checks."""

    metrics = parse_metrics(task, metrics_data)
    if isinstance(task, StokesTask):
        return validate_stokes_result(task, metrics)
    return validate_poisson_result(task, metrics)


__all__ = ["evaluate_task", "parse_task", "solve_task"]
