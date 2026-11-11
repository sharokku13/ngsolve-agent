"""Run the four deterministic NGSolve benchmark cases and summarize their checks."""

from __future__ import annotations

import json
import math
import platform
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ngsolve
from maxwell_eigenvalue_fichera import run as run_maxwell
from netgen.libngpy._meshing import NgException
from nonlinear_elasticity import run as run_elasticity
from stokes_cylinder import run as run_stokes

from ngsolve_agent.fem_solver import solve_poisson
from ngsolve_agent.schemas import PoissonTask
from ngsolve_agent.validation import validate_poisson_result


def run_suite(output_root: Path) -> dict[str, Any]:
    """Execute Poisson, nonlinear elasticity, Stokes, and Maxwell cases in order."""

    run_id = datetime.now(UTC).strftime("suite-%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / "benchmark_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    # B0 manufactured Poisson: exact solution and error norms provide its oracle.
    case_dir = run_dir / "poisson"
    task = PoissonTask()
    metrics, image_path = solve_poisson(task, case_dir / "figures")
    failures = validate_poisson_result(task, metrics)
    records.append(
        {
            "task_id": task.task_id,
            "benchmark": "B0",
            "passed": not failures,
            "failures": failures,
            "metrics": metrics.model_dump(mode="json"),
            "artifacts": {"solution_plot": str(image_path.relative_to(run_dir))},
        }
    )

    # For models without a closed-form solution, use solver-specific invariants.
    cases: list[tuple[str, Callable[[Path], dict[str, Any]], Callable[[dict[str, Any]], list[str]]]] = [
        (
            "neo_hookean_cantilever",
            run_elasticity,
            lambda report: _check_elasticity(report),
        ),
        ("stokes_cylinder", run_stokes, lambda report: _check_stokes(report)),
        ("maxwell_fichera", run_maxwell, lambda report: _check_maxwell(report)),
    ]
    for case_name, runner, checker in cases:
        case_dir = run_dir / case_name
        try:
            report = runner(case_dir)
            failures = checker(report)
            report["checks"] = {"passed": not failures, "failures": failures}
            report_path = case_dir / {
                "neo_hookean_cantilever": "nonlinear_elasticity.json",
                "stokes_cylinder": "stokes_cylinder.json",
                "maxwell_fichera": "maxwell_eigenvalue.json",
            }[case_name]
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            records.append(
                {
                    "task_id": report["task_id"],
                    "benchmark": case_name,
                    "passed": not failures,
                    "failures": failures,
                    "results": report["results"],
                    "report": str(report_path.relative_to(run_dir)),
                }
            )
        except (NgException, OSError, ValueError, ArithmeticError, RuntimeError) as exc:
            records.append(
                {
                    "task_id": case_name,
                    "benchmark": case_name,
                    "passed": False,
                    "failures": [f"{type(exc).__name__}: {exc}"],
                }
            )

    suite_report = {
        "suite_run_id": run_id,
        "architecture": "B0 deterministic solvers; no LLM connected",
        "all_passed": all(record["passed"] for record in records),
        "cases": records,
        "environment": {
            "python": platform.python_version(),
            "ngsolve": getattr(ngsolve, "__version__", "unknown"),
        },
    }
    suite_path = run_dir / "suite.json"
    suite_path.write_text(json.dumps(suite_report, indent=2), encoding="utf-8")
    print(json.dumps(suite_report, indent=2))
    print(f"suite report: {suite_path}")
    return suite_report


def _check_elasticity(report: dict[str, Any]) -> list[str]:
    results = report["results"]
    failures: list[str] = []
    if not results["converged"]:
        failures.append("Newton continuation did not converge")
    if not math.isfinite(results["final_energy"]):
        failures.append("final energy is not finite")
    if results["free_dof_residual"] > 1.0e-8:
        failures.append("free-DOF equilibrium residual exceeds 1e-8")
    return failures


def _check_stokes(report: dict[str, Any]) -> list[str]:
    results = report["results"]
    failures: list[str] = []
    if results["relative_flux_imbalance"] > 1.0e-6:
        failures.append("relative inlet/outlet flux imbalance exceeds 1e-6")
    if results["free_dof_residual"] > 1.0e-8:
        failures.append("free-DOF residual exceeds 1e-8")
    if not math.isfinite(results["divergence_l2"]):
        failures.append("divergence norm is not finite")
    return failures


def _check_maxwell(report: dict[str, Any]) -> list[str]:
    results = report["results"]
    values = results["eigenvalues"]
    failures: list[str] = []
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        failures.append("eigenvalues must be finite and positive")
    if values != sorted(values):
        failures.append("eigenvalues are not sorted")
    if results["first_eigenvalue_relative_difference_from_tutorial"] > 5.0e-3:
        failures.append("first eigenvalue differs from tutorial reference by more than 0.5%")
    return failures


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    summary = run_suite(project_root / "results")
    raise SystemExit(0 if summary["all_passed"] else 1)
