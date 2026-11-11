from pydantic import ValidationError

from ngsolve_agent.schemas import PoissonMetrics, PoissonTask, StokesMetrics, StokesTask
from ngsolve_agent.validation import validate_poisson_result, validate_stokes_result


def test_valid_poisson_result_passes() -> None:
    task = PoissonTask()
    metrics = PoissonMetrics(
        l2_error=0.01,
        h1_seminorm_error=0.1,
        free_dof_residual=1e-12,
        ndof=100,
        nelements=128,
        maxh=task.maxh,
        order=task.order,
        runtime_seconds=0.2,
        converged=True,
    )

    assert validate_poisson_result(task, metrics) == []


def test_inaccurate_result_fails_accuracy_gate() -> None:
    task = PoissonTask()
    metrics = PoissonMetrics(
        l2_error=0.03,
        h1_seminorm_error=0.1,
        free_dof_residual=1e-12,
        ndof=100,
        nelements=128,
        maxh=task.maxh,
        order=task.order,
        runtime_seconds=0.2,
        converged=True,
    )

    assert any("exceeds tolerance" in failure for failure in validate_poisson_result(task, metrics))


def test_valid_stokes_result_passes_flux_and_divergence_checks() -> None:
    task = StokesTask()
    metrics = StokesMetrics(
        divergence_l2=0.2,
        inlet_flux=0.41,
        outlet_flux=0.41,
        relative_flux_imbalance=0.0,
        free_dof_residual=1e-12,
        ndof=100,
        nelements=128,
        maxh=task.maxh,
        runtime_seconds=0.2,
        converged=True,
    )

    assert validate_stokes_result(task, metrics) == []


def test_stokes_divergence_is_reported_but_not_an_arbitrary_pass_gate() -> None:
    task = StokesTask()
    metrics = StokesMetrics(
        divergence_l2=0.8,
        inlet_flux=0.41,
        outlet_flux=0.41,
        relative_flux_imbalance=0.0,
        free_dof_residual=1e-12,
        ndof=100,
        nelements=128,
        maxh=task.maxh,
        runtime_seconds=0.2,
        converged=True,
    )

    assert validate_stokes_result(task, metrics) == []


def test_task_schemas_reject_model_controlled_evaluator_tolerances() -> None:
    for task_type, field in (
        (PoissonTask, "l2_tolerance"),
        (StokesTask, "divergence_tolerance"),
        (StokesTask, "flux_imbalance_tolerance"),
        (StokesTask, "residual_tolerance"),
    ):
        try:
            task_type(**{field: 0.99})
        except ValidationError:
            continue
        raise AssertionError(f"{task_type.__name__} accepted evaluator setting {field}")


def test_stokes_flux_imbalance_fails_independent_evaluator() -> None:
    task = StokesTask()
    metrics = StokesMetrics(
        divergence_l2=0.2,
        inlet_flux=0.41,
        outlet_flux=0.39,
        relative_flux_imbalance=0.05,
        free_dof_residual=1e-12,
        ndof=100,
        nelements=128,
        maxh=task.maxh,
        runtime_seconds=0.2,
        converged=True,
    )

    failures = validate_stokes_result(task, metrics)
    assert any("flux imbalance" in failure for failure in failures)


def test_task_schemas_reject_pathological_mesh_sizes() -> None:
    for task_type in (PoissonTask, StokesTask):
        try:
            task_type(maxh=1.0e-20)
        except ValidationError:
            continue
        raise AssertionError(f"{task_type.__name__} accepted an unsafe mesh size")
