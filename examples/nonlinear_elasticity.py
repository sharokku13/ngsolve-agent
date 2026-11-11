"""Reproducible nonlinear Neo-Hookean cantilever benchmark from NGSolve tutorials.

The model is a 2D beam fixed on its left edge and loaded by gravity. The load is
introduced incrementally; each increment is solved by Newton iteration. The
script records a JSON report and a displacement/deformed-shape figure.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import ngsolve
import numpy as np
from netgen.occ import OCCGeometry, Rectangle, X
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    Det,
    GridFunction,
    Id,
    InnerProduct,
    Mesh,
    Parameter,
    Trace,
    Variation,
    VectorH1,
    dx,
    exp,
    grad,
    log,
)


def run(output_dir: Path, mesh_maxh: float = 0.05) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    # NGSolve tutorial geometry and material parameters.
    shape = Rectangle(1.0, 0.1).Face()
    shape.edges.Min(X).name = "left"
    shape.edges.Min(X).maxh = 0.01
    geometry = OCCGeometry(shape, dim=2)
    mesh = Mesh(geometry.GenerateMesh(maxh=mesh_maxh))

    young_modulus, poisson_ratio = 210.0, 0.2
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lam = young_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    fes = VectorH1(mesh, order=2, dirichlet="left")
    displacement = fes.TrialFunction()

    # Compressible Neo-Hookean energy, matching the NGSolve tutorial formulation.
    identity = Id(mesh.dim)
    deformation_gradient = identity + grad(displacement)
    right_cauchy_green = deformation_gradient.trans * deformation_gradient
    jacobian_squared = Det(right_cauchy_green)
    neo_hookean = 0.5 * mu * (
        Trace(right_cauchy_green - identity)
        + (2.0 * mu / lam) * exp((-lam / (2.0 * mu)) * log(jacobian_squared))
        - 1.0
    )
    load = Parameter(0.0)
    body_force = CoefficientFunction((0.0, -1.0))
    energy_form = BilinearForm(fes, symmetric=True)
    energy_form += Variation(
        neo_hookean * dx - load * InnerProduct(body_force, displacement) * dx
    )

    solution = GridFunction(fes, name="displacement")
    residual = solution.vec.CreateVector()
    increment = solution.vec.CreateVector()
    newton_counts: list[int] = []
    final_criterion = math.inf
    final_energy = math.nan

    # Tutorial load continuation: gamma = 0.1, 0.2, ..., 5.0.
    for step in range(1, 51):
        load.Set(step / 10.0)
        converged = False
        for iteration in range(20):
            energy_form.Apply(solution.vec, residual)
            energy_form.AssembleLinearization(solution.vec)
            increment.data = energy_form.mat.Inverse(fes.FreeDofs()) * residual
            solution.vec.data -= increment
            final_criterion = math.sqrt(
                abs(float(InnerProduct(increment, residual)))
            )
            if final_criterion < 1.0e-9:
                newton_counts.append(iteration + 1)
                converged = True
                break
        if not converged:
            raise RuntimeError(
                f"Newton failed at load step {step}/10; criterion={final_criterion:.3e}"
            )
        final_energy = float(energy_form.Energy(solution.vec))

    l2_displacement_norm = float(
        ngsolve.Integrate(ngsolve.Norm(solution) ** 2, mesh)
    ) ** 0.5
    # Evaluate the final equilibrium residual on free DOFs.
    energy_form.Apply(solution.vec, residual)
    free_dofs = fes.FreeDofs()
    free_residual = math.sqrt(
        sum(float(residual[i]) ** 2 for i in range(fes.ndof) if free_dofs[i])
    )

    # Sample displacement on the physical domain for a lightweight artifact.
    nx, ny = 161, 31
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 0.1, ny)
    ux = np.zeros((ny, nx))
    uy = np.zeros_like(ux)
    for row, yy in enumerate(ys):
        for col, xx in enumerate(xs):
            value = solution(mesh(float(xx), float(yy)))
            ux[row, col], uy[row, col] = float(value[0]), float(value[1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    magnitude = np.sqrt(ux**2 + uy**2)
    image = axes[0].imshow(
        uy,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 0.1),
        aspect="auto",
        cmap="coolwarm",
    )
    axes[0].set_title("Vertical displacement")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(image, ax=axes[0], label="displacement")
    axes[1].imshow(
        magnitude,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 0.1),
        aspect="auto",
        cmap="viridis",
    )
    # Exaggerated deformation overlay, useful for inspecting the bending direction.
    step_x, step_y = 10, 3
    axes[1].quiver(
        xs[::step_x], ys[::step_y], ux[::step_y, ::step_x], uy[::step_y, ::step_x],
        angles="xy", scale_units="xy", scale=0.08, color="white", width=0.003,
    )
    axes[1].set_title("Displacement magnitude and vectors")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plot_path = output_dir / "nonlinear_elasticity.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "task_id": "ngsolve_tutorial_neo_hookean_cantilever_2d",
        "source": "https://docu.ngsolve.org/latest/i-tutorials/unit-3.4-nonlmin/nonlmin.html",
        "formulation": "2D compressible Neo-Hookean beam, left edge clamped, gravity load",
        "parameters": {
            "domain": [1.0, 0.1],
            "mesh_maxh": mesh_maxh,
            "polynomial_order": 2,
            "young_modulus": young_modulus,
            "poisson_ratio": poisson_ratio,
            "load_steps": 50,
            "final_load_factor": 5.0,
        },
        "results": {
            "converged": True,
            "newton_iterations_per_load_step": newton_counts,
            "max_newton_criterion": final_criterion,
            "final_energy": final_energy,
            "free_dof_residual": free_residual,
            "l2_displacement_norm": l2_displacement_norm,
            "ndof": fes.ndof,
            "elements": mesh.ne,
            "runtime_seconds": elapsed,
        },
        "environment": {
            "python": platform.python_version(),
            "ngsolve": getattr(ngsolve, "__version__", "unknown"),
        },
        "artifact": plot_path.name,
    }
    (output_dir / "nonlinear_elasticity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxh", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tutorials/neo_hookean_cantilever"),
    )
    args = parser.parse_args()
    result = run(args.output_dir, mesh_maxh=args.maxh)
    print(json.dumps(result, indent=2))
