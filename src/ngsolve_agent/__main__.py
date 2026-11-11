"""Command-line entry point for the deterministic B0 baseline."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from .fem_solver import solve_poisson
from .reporting import write_report
from .schemas import PoissonTask
from .validation import validate_poisson_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the NGSolve B0 Poisson baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--maxh", type=float, default=0.2)
    parser.add_argument("--order", type=int, default=2)
    args = parser.parse_args()

    task = PoissonTask(maxh=args.maxh, order=args.order)
    run_id = datetime.now(UTC).strftime("B0-%Y%m%dT%H%M%S%fZ")
    figure_dir = args.output_dir / "figures" / run_id
    metrics, image_path = solve_poisson(task, figure_dir)
    failures = validate_poisson_result(task, metrics)
    report_path = write_report(task, metrics, failures, args.output_dir, image_path, run_id)

    print(f"task: {task.task_id}")
    print(f"L2 error: {metrics.l2_error:.6e}")
    print(f"H1 seminorm error: {metrics.h1_seminorm_error:.6e}")
    print(f"free-DOF residual: {metrics.free_dof_residual:.6e}")
    print(f"degrees of freedom: {metrics.ndof}")
    print(f"checks: {'PASS' if not failures else 'FAIL'}")
    print(f"report: {report_path}")
    print(f"plot: {image_path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
