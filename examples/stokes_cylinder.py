"""Run the NGSolve Taylor-Hood Stokes tutorial around a cylinder."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import ngsolve

from ngsolve_agent.schemas import StokesTask
from ngsolve_agent.stokes_solver import solve_stokes


def run(output_dir: Path, maxh: float = 0.05) -> dict[str, object]:
    task = StokesTask(maxh=maxh)
    metrics, image_path = solve_stokes(task, output_dir)
    report: dict[str, object] = {
        "task_id": task.task_id,
        "source": "https://docu.ngsolve.org/latest/i-tutorials/unit-2.6-stokes/stokes.html",
        "formulation": "steady 2D Stokes, Taylor-Hood P2/P1, parabolic inlet, no-slip walls/cylinder",
        "parameters": {
            "channel": [2.0, 0.41],
            "cylinder_center": [0.2, 0.2],
            "cylinder_radius": 0.05,
            "mesh_maxh": maxh,
            "mesh_curve_order": 3,
            "velocity_order": 2,
            "pressure_order": 1,
        },
        "results": {
            "divergence_l2": metrics.divergence_l2,
            "inlet_flux": metrics.inlet_flux,
            "outlet_flux": metrics.outlet_flux,
            "relative_flux_imbalance": metrics.relative_flux_imbalance,
            "free_dof_residual": metrics.free_dof_residual,
            "ndof": metrics.ndof,
            "elements": metrics.nelements,
            "runtime_seconds": metrics.runtime_seconds,
            "converged": metrics.converged,
        },
        "environment": {
            "python": platform.python_version(),
            "ngsolve": getattr(ngsolve, "__version__", "unknown"),
        },
        "artifact": image_path.name,
    }
    (output_dir / "stokes_cylinder.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxh", type=float, default=0.05)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/tutorials/stokes_cylinder")
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, maxh=args.maxh), indent=2))
