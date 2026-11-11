"""Deterministic NGSolve B0 implementation for a manufactured Poisson problem."""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from netgen.geom2d import unit_square  # type: ignore[import-not-found]
from ngsolve import (  # type: ignore[import-not-found]
    H1,
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    Integrate,
    LinearForm,
    Mesh,
    cos,
    dx,
    grad,
    sin,
    x,
    y,
)

from .schemas import PoissonMetrics, PoissonTask


def _exact(x_value: float | np.ndarray, y_value: float | np.ndarray) -> float | np.ndarray:
    return np.sin(np.pi * x_value) * np.sin(np.pi * y_value)


def solve_poisson(task: PoissonTask, output_dir: Path) -> tuple[PoissonMetrics, Path]:
    """Assemble and solve the B0 problem, then independently calculate error norms."""

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    mesh = Mesh(unit_square.GenerateMesh(maxh=task.maxh))
    fes = H1(mesh, order=task.order, dirichlet="left|right|top|bottom")
    trial, test = fes.TnT()

    exact = sin(np.pi * x) * sin(np.pi * y)
    forcing = 2.0 * np.pi**2 * exact

    bilinear = BilinearForm(fes, symmetric=True)
    bilinear += grad(trial) * grad(test) * dx
    bilinear.Assemble()

    linear = LinearForm(fes)
    linear += forcing * test * dx
    linear.Assemble()

    solution = GridFunction(fes, name="u")
    solution.vec.data = bilinear.mat.Inverse(fes.FreeDofs()) * linear.vec

    l2_error = math.sqrt(max(0.0, float(Integrate((solution - exact) ** 2, mesh))))
    exact_gradient = CoefficientFunction(
        (
            np.pi * cos(np.pi * x) * sin(np.pi * y),
            np.pi * sin(np.pi * x) * cos(np.pi * y),
        )
    )
    gradient_error = grad(solution) - exact_gradient
    h1_error = math.sqrt(
        max(0.0, float(Integrate(gradient_error * gradient_error, mesh)))
    )
    residual = linear.vec.CreateVector()
    residual.data = bilinear.mat * solution.vec - linear.vec
    free_dofs = fes.FreeDofs()
    free_residual = math.sqrt(
        sum(float(residual[i]) ** 2 for i in range(fes.ndof) if free_dofs[i])
    )

    elapsed = time.perf_counter() - started
    free_rhs_norm = math.sqrt(
        sum(float(linear.vec[i]) ** 2 for i in range(fes.ndof) if free_dofs[i])
    )
    residual_tolerance = max(1e-12, free_rhs_norm * 1e-8)
    metrics = PoissonMetrics(
        l2_error=l2_error,
        h1_seminorm_error=h1_error,
        free_dof_residual=free_residual,
        ndof=fes.ndof,
        nelements=mesh.ne,
        maxh=task.maxh,
        order=task.order,
        runtime_seconds=elapsed,
        converged=(
            math.isfinite(l2_error)
            and math.isfinite(h1_error)
            and free_residual <= residual_tolerance
        ),
    )

    image_path = output_dir / "solution.png"
    _save_solution_plot(solution, mesh, task, image_path)
    return metrics, image_path


def _save_solution_plot(solution: GridFunction, mesh: Mesh, task: PoissonTask, path: Path) -> None:
    values = np.linspace(0.0, 1.0, task.grid_samples)
    xx, yy = np.meshgrid(values, values)
    numerical = np.empty_like(xx)
    for row in range(task.grid_samples):
        for col in range(task.grid_samples):
            numerical[row, col] = float(solution(mesh(xx[row, col], yy[row, col])))
    exact = _exact(xx, yy)
    error = np.abs(numerical - exact)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    plots = (
        (numerical, "NGSolve solution"),
        (exact, "Exact solution"),
        (error, "Absolute error"),
    )
    for axis, (field, title) in zip(axes, plots, strict=True):
        artist = axis.imshow(
            field,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(artist, ax=axis, shrink=0.82)
    figure.suptitle(f"{task.task_id} (order {task.order}, maxh {task.maxh:g})")
    figure.savefig(path, dpi=160)
    plt.close(figure)

