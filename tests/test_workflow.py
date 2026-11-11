from pathlib import Path

from ngsolve_agent.schemas import PoissonMetrics, StokesMetrics
from ngsolve_agent.workflow import build_d4_workflow


def _working_solver(task, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "solution.png"
    image_path.write_bytes(b"test plot")
    return (
        PoissonMetrics(
            l2_error=0.01,
            h1_seminorm_error=0.1,
            free_dof_residual=1e-12,
            ndof=100,
            nelements=128,
            maxh=task.maxh,
            order=task.order,
            runtime_seconds=0.2,
            converged=True,
        ),
        image_path,
    )


def test_d4_runs_valid_task_through_solver_and_evaluator(tmp_path: Path) -> None:
    def interpreter(_request: str, _errors: list[str]) -> dict[str, object]:
        return {"maxh": 0.2, "order": 2}

    workflow = build_d4_workflow(interpreter, tmp_path, solver=_working_solver)
    result = workflow.invoke({"request": "Solve the unit-square Poisson benchmark"})

    assert result["status"] == "passed"
    assert result["metrics"]["l2_error"] == 0.01
    assert Path(result["report_path"]).is_file()
    assert Path(result["image_path"]).is_file()


def test_d4_recovers_once_from_invalid_typed_parameters(tmp_path: Path) -> None:
    responses = iter(
        [
            {"maxh": 2.0, "order": 2},
            {"maxh": 0.2, "order": 2},
        ]
    )
    calls: list[list[str]] = []

    def interpreter(_request: str, previous_errors: list[str]) -> dict[str, object]:
        calls.append(previous_errors)
        return next(responses)

    workflow = build_d4_workflow(
        interpreter, tmp_path, max_attempts=2, solver=_working_solver
    )
    result = workflow.invoke({"request": "Solve the unit-square Poisson benchmark"})

    assert result["status"] == "passed"
    assert len(calls) == 2
    assert calls[1]
    assert len(list((tmp_path / "reports").glob("D4-*.json"))) == 1


def test_d4_routes_stokes_task_to_typed_evaluator(tmp_path: Path) -> None:
    def interpreter(_request: str, _errors: list[str]) -> dict[str, object]:
        return {
            "task_id": "ngsolve_tutorial_stokes_flow_around_cylinder",
            "maxh": 0.05,
        }

    def stokes_solver(task, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "stokes.png"
        image_path.write_bytes(b"test plot")
        return (
            StokesMetrics(
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
            ),
            image_path,
        )

    workflow = build_d4_workflow(interpreter, tmp_path, solver=stokes_solver)
    result = workflow.invoke(
        {
            "request": "Solve Stokes flow around a cylinder",
            "expected_task_id": "ngsolve_tutorial_stokes_flow_around_cylinder",
        }
    )

    assert result["status"] == "passed"
    assert result["task"]["task_id"] == "ngsolve_tutorial_stokes_flow_around_cylinder"
    assert result["metrics"]["relative_flux_imbalance"] == 0.0
