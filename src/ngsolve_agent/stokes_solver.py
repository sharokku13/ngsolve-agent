"""Deterministic NGSolve Stokes solver for the tutorial channel/cylinder task."""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (
    H1,
    BilinearForm,
    CoefficientFunction,
    FESpace,
    GridFunction,
    InnerProduct,
    Integrate,
    Mesh,
    VectorH1,
    div,
    dx,
    grad,
    y,
)

from .schemas import (
    STOKES_FLUX_IMBALANCE_LIMIT,
    STOKES_FREE_RESIDUAL_LIMIT,
    StokesMetrics,
    StokesTask,
)


def solve_stokes(task: StokesTask, output_dir: Path) -> tuple[StokesMetrics, Path]:
    """Solve the steady cylinder-flow tutorial and return its checked metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    geometry = SplineGeometry()
    geometry.AddRectangle(
        (0.0, 0.0), (2.0, 0.41), bcs=("wall", "outlet", "wall", "inlet")
    )
    geometry.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
    mesh = Mesh(geometry.GenerateMesh(maxh=task.maxh))
    mesh.Curve(3)

    velocity_space = VectorH1(mesh, order=2, dirichlet="wall|inlet|cyl")
    pressure_space = H1(mesh, order=1)
    mixed_space = FESpace([velocity_space, pressure_space])
    (velocity, pressure), (test_velocity, test_pressure) = mixed_space.TnT()

    form = BilinearForm(mixed_space, symmetric=True)
    form += (
        InnerProduct(grad(velocity), grad(test_velocity))
        + div(velocity) * test_pressure
        + div(test_velocity) * pressure
    ) * dx
    form.Assemble()

    solution = GridFunction(mixed_space, name="stokes_solution")
    velocity_field, pressure_field = solution.components
    inlet_profile = 1.5 * 4.0 * y * (0.41 - y) / (0.41 * 0.41)
    velocity_field.Set(
        CoefficientFunction((inlet_profile, 0.0)),
        definedon=mesh.Boundaries("inlet"),
    )
    rhs = solution.vec.CreateVector()
    rhs.data = -form.mat * solution.vec
    inverse = form.mat.Inverse(freedofs=mixed_space.FreeDofs(), inverse="umfpack")
    solution.vec.data += inverse * rhs

    divergence_l2 = math.sqrt(max(0.0, float(Integrate(div(velocity_field) ** 2, mesh))))
    inlet_flux = float(
        Integrate(velocity_field[0], mesh, definedon=mesh.Boundaries("inlet"))
    )
    outlet_flux = float(
        Integrate(velocity_field[0], mesh, definedon=mesh.Boundaries("outlet"))
    )
    flux_imbalance = abs(inlet_flux - outlet_flux) / max(abs(inlet_flux), 1.0e-14)
    residual = solution.vec.CreateVector()
    residual.data = form.mat * solution.vec
    free_dofs = mixed_space.FreeDofs()
    free_residual = math.sqrt(
        sum(
            float(residual[index]) ** 2
            for index in range(mixed_space.ndof)
            if free_dofs[index]
        )
    )

    xs = np.linspace(0.0, 2.0, 321)
    ys = np.linspace(0.0, 0.41, 101)
    speed = np.full((len(ys), len(xs)), np.nan)
    pressure_values = np.full_like(speed, np.nan)
    for row, yy in enumerate(ys):
        for col, xx in enumerate(xs):
            if (xx - 0.2) ** 2 + (yy - 0.2) ** 2 < 0.05**2:
                continue
            try:
                point = mesh(float(xx), float(yy))
                value = velocity_field(point)
                speed[row, col] = math.hypot(float(value[0]), float(value[1]))
                pressure_values[row, col] = float(pressure_field(point))
            except (ValueError, RuntimeError):
                continue

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for axis, values, title, cmap in (
        (axes[0], speed, "Velocity magnitude", "viridis"),
        (axes[1], pressure_values, "Pressure", "coolwarm"),
    ):
        artist = axis.imshow(
            values,
            origin="lower",
            extent=(0.0, 2.0, 0.0, 0.41),
            aspect="auto",
            cmap=cmap,
        )
        axis.add_patch(plt.Circle((0.2, 0.2), 0.05, color="white", zorder=3))
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(artist, ax=axis)
    image_path = output_dir / "stokes_cylinder.png"
    figure.savefig(image_path, dpi=160)
    plt.close(figure)

    elapsed = time.perf_counter() - started
    converged = (
        math.isfinite(divergence_l2)
        and math.isfinite(flux_imbalance)
        and math.isfinite(free_residual)
        and flux_imbalance <= STOKES_FLUX_IMBALANCE_LIMIT
        and free_residual <= STOKES_FREE_RESIDUAL_LIMIT
    )
    metrics = StokesMetrics(
        divergence_l2=divergence_l2,
        inlet_flux=inlet_flux,
        outlet_flux=outlet_flux,
        relative_flux_imbalance=flux_imbalance,
        free_dof_residual=free_residual,
        ndof=mixed_space.ndof,
        nelements=mesh.ne,
        maxh=task.maxh,
        runtime_seconds=elapsed,
        converged=converged,
    )
    return metrics, image_path
