"""Independent evaluator checks for baseline outputs."""

from __future__ import annotations

import math

from .schemas import (
    POISSON_L2_ERROR_LIMIT,
    STOKES_FLUX_IMBALANCE_LIMIT,
    STOKES_FREE_RESIDUAL_LIMIT,
    PoissonMetrics,
    PoissonTask,
    StokesMetrics,
    StokesTask,
)


def validate_poisson_result(task: PoissonTask, metrics: PoissonMetrics) -> list[str]:
    """Return check failures; an empty list means the run passed these checks."""

    failures: list[str] = []
    for name in (
        "l2_error",
        "h1_seminorm_error",
        "free_dof_residual",
        "runtime_seconds",
    ):
        value = getattr(metrics, name)
        if not math.isfinite(value):
            failures.append(f"{name} is not finite")

    if metrics.ndof <= 0:
        failures.append("ndof must be positive")
    if metrics.nelements <= 0:
        failures.append("nelements must be positive")
    if metrics.order != task.order:
        failures.append("reported finite-element order does not match task")
    if not math.isclose(metrics.maxh, task.maxh, rel_tol=0.0, abs_tol=1e-12):
        failures.append("reported maxh does not match task")
    if not metrics.converged:
        failures.append("solver did not report convergence")
    if metrics.l2_error > POISSON_L2_ERROR_LIMIT:
        failures.append(
            f"L2 error {metrics.l2_error:.3e} exceeds tolerance "
            f"{POISSON_L2_ERROR_LIMIT:.3e}"
        )
    return failures


def validate_stokes_result(task: StokesTask, metrics: StokesMetrics) -> list[str]:
    """Check finite solver output and mass balance; divergence is diagnostic."""

    failures: list[str] = []
    for name in (
        "divergence_l2",
        "inlet_flux",
        "outlet_flux",
        "relative_flux_imbalance",
        "free_dof_residual",
        "runtime_seconds",
    ):
        if not math.isfinite(getattr(metrics, name)):
            failures.append(f"{name} is not finite")
    if metrics.ndof <= 0:
        failures.append("ndof must be positive")
    if metrics.nelements <= 0:
        failures.append("nelements must be positive")
    if metrics.maxh != task.maxh:
        failures.append("reported maxh does not match task")
    if not metrics.converged:
        failures.append("solver did not report convergence")
    if metrics.relative_flux_imbalance > STOKES_FLUX_IMBALANCE_LIMIT:
        failures.append(
            "relative inlet/outlet flux imbalance "
            f"{metrics.relative_flux_imbalance:.3e} exceeds tolerance "
            f"{STOKES_FLUX_IMBALANCE_LIMIT:.3e}"
        )
    if metrics.free_dof_residual > STOKES_FREE_RESIDUAL_LIMIT:
        failures.append(
            f"free-DOF residual {metrics.free_dof_residual:.3e} exceeds tolerance "
            f"{STOKES_FREE_RESIDUAL_LIMIT:.3e}"
        )
    return failures
