"""Typed task and result contracts shared by the baseline and future agent arms."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# Fixed evaluator policy. These are deliberately not task parameters, so an
# LLM request cannot relax the checks used to judge its own result.
POISSON_L2_ERROR_LIMIT = 2.0e-2
STOKES_FLUX_IMBALANCE_LIMIT = 1.0e-8
STOKES_FREE_RESIDUAL_LIMIT = 1.0e-10


class PoissonTask(BaseModel):
    """Manufactured Poisson task on the unit square."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: Literal["poisson_unit_square_smooth"] = "poisson_unit_square_smooth"
    maxh: float = Field(default=0.2, ge=0.025, le=1.0)
    order: int = Field(default=2, ge=1, le=6)
    grid_samples: int = Field(default=64, ge=32, le=512)


class PoissonMetrics(BaseModel):
    """Numerical checks for a single baseline run."""

    l2_error: float
    h1_seminorm_error: float
    free_dof_residual: float
    ndof: int
    nelements: int
    maxh: float
    order: int
    runtime_seconds: float
    converged: bool


class StokesTask(BaseModel):
    """NGSolve channel-flow tutorial with a cylinder and Taylor-Hood elements."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: Literal["ngsolve_tutorial_stokes_flow_around_cylinder"] = (
        "ngsolve_tutorial_stokes_flow_around_cylinder"
    )
    maxh: float = Field(default=0.05, ge=0.025, le=0.2)


class StokesMetrics(BaseModel):
    """Numerical and physical checks for the cylinder-flow benchmark."""

    divergence_l2: float
    inlet_flux: float
    outlet_flux: float
    relative_flux_imbalance: float
    free_dof_residual: float
    ndof: int
    nelements: int
    maxh: float
    runtime_seconds: float
    converged: bool


type TaskSpec = PoissonTask | StokesTask
type MetricsSpec = PoissonMetrics | StokesMetrics


def parse_task(data: dict[str, object]) -> TaskSpec:
    """Parse one supported task using its explicit registry ID."""

    task_id = data.get("task_id", "poisson_unit_square_smooth")
    if task_id == "poisson_unit_square_smooth":
        return PoissonTask.model_validate(data)
    if task_id == "ngsolve_tutorial_stokes_flow_around_cylinder":
        return StokesTask.model_validate(data)
    raise ValueError(f"unsupported task_id: {task_id}")


def parse_metrics(task: TaskSpec, data: dict[str, object]) -> MetricsSpec:
    """Validate the metric shape corresponding to a task type."""

    if isinstance(task, StokesTask):
        return StokesMetrics.model_validate(data)
    return PoissonMetrics.model_validate(data)
