from pathlib import Path

import pytest

from ngsolve_agent.agent_cli import _classify_request, run_agent
from ngsolve_agent.llm_adapter import LLMSettings


@pytest.mark.parametrize("architecture", ["d1", "d4"])
@pytest.mark.parametrize(
    "prompt",
    [
        "Solve three-dimensional incompressible Navier-Stokes flow around a cylinder.",
        "Use Poisson for comparison, but solve the Navier-Stokes flow around a cylinder.",
    ],
)
def test_agent_rejects_out_of_scope_request_before_model_call(
    architecture: str, prompt: str, tmp_path: Path
) -> None:
    settings = LLMSettings(model="unused")

    report = run_agent(
        architecture,
        prompt,
        tmp_path,
        settings,
    )

    assert report["status"] == "unsupported_request"
    assert report["llm_calls"] == []
    assert report["agent_result"]["trace"][0]["action"] == "scope_check"
    report_path = next(tmp_path.glob("agent_runs/*/agent_run.json"))
    assert report_path.is_file()


@pytest.mark.parametrize(
    ("prompt", "expected_task_id"),
    [
        (
            "Solve the manufactured Poisson problem on the unit square with mesh size 0.1.",
            "poisson_unit_square_smooth",
        ),
        (
            "Solve the manufactured Poisson equation on [0,1]^2 with h=0.1.",
            "poisson_unit_square_smooth",
        ),
        (
            "Solve steady Stokes flow around a cylinder in a channel with mesh size 0.05.",
            "ngsolve_tutorial_stokes_flow_around_cylinder",
        ),
        ("Solve Poisson flow on a complex three-dimensional geometry.", None),
        ("Solve Maxwell in the unit square with Poisson mentioned for context.", None),
    ],
)
def test_request_classifier_routes_only_supported_problem_forms(
    prompt: str, expected_task_id: str | None
) -> None:
    assert _classify_request(prompt) == expected_task_id
