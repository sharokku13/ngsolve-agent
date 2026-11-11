from pathlib import Path

from ngsolve_agent.d1_agent import run_d1_agent
from ngsolve_agent.schemas import PoissonMetrics, StokesMetrics


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


def test_d1_model_selects_typed_tools_to_complete_run(tmp_path: Path) -> None:
    decisions = iter(
        [
            {"tool": "set_task", "arguments": {"task": {"maxh": 0.2, "order": 2}}},
            {"tool": "run_solver", "arguments": {}},
            {"tool": "evaluate", "arguments": {}},
        ]
    )
    observed_states: list[dict[str, object]] = []

    def select_tool(_request, state, _available_tools):
        observed_states.append(state)
        return next(decisions)

    result = run_d1_agent(
        "Solve the unit-square Poisson benchmark",
        select_tool,
        tmp_path,
        solver=_working_solver,
    )

    assert result["status"] == "passed"
    assert [entry["tool"] for entry in result["trace"]] == [
        "set_task",
        "run_solver",
        "evaluate",
    ]
    assert observed_states[1]["status"] == "task_ready"
    assert Path(result["report_path"]).is_file()


def test_d1_dispatches_supported_stokes_task(tmp_path: Path) -> None:
    decisions = iter(
        [
            {
                "tool": "set_task",
                "arguments": {
                    "task": {
                        "task_id": "ngsolve_tutorial_stokes_flow_around_cylinder",
                        "maxh": 0.05,
                    }
                },
            },
            {"tool": "run_solver", "arguments": {}},
            {"tool": "evaluate", "arguments": {}},
        ]
    )

    def select_tool(_request, _state, _available_tools):
        return next(decisions)

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

    result = run_d1_agent(
        "Solve Stokes flow around a cylinder in a channel",
        select_tool,
        tmp_path,
        solver=stokes_solver,
        expected_task_id="ngsolve_tutorial_stokes_flow_around_cylinder",
    )

    assert result["status"] == "passed"
    assert result["task"]["task_id"] == "ngsolve_tutorial_stokes_flow_around_cylinder"
