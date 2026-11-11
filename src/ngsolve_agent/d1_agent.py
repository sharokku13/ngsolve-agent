"""D1: model-directed calls to a small set of typed FEM tools."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from netgen.libngpy._meshing import NgException
from pydantic import BaseModel, ConfigDict, ValidationError

from .reporting import write_report
from .schemas import MetricsSpec, TaskSpec, parse_metrics, parse_task
from .task_registry import evaluate_task, solve_task

TOOL_CONTRACTS: list[dict[str, Any]] = [
    {
        "name": "set_task",
        "description": (
            "Validate and store one allowlisted Poisson or tutorial Stokes task. "
            "Set the matching task_id and include every explicitly requested setting."
        ),
        "arguments": {
            "task": {
                "task_id": "poisson_unit_square_smooth or ngsolve_tutorial_stokes_flow_around_cylinder",
                "maxh": "optional number; Poisson [0.025,1] default 0.2; Stokes [0.025,0.2] default 0.05",
                "order": "optional integer 1..6, Poisson only, default 2",
                "grid_samples": "optional integer 32..512, Poisson only, default 64",
            }
        },
    },
    {
        "name": "run_solver",
        "description": "Run NGSolve for the stored task and save its solution plot.",
        "arguments": {},
    },
    {
        "name": "evaluate",
        "description": "Check the numerical result and write a JSON report.",
        "arguments": {},
    },
    {
        "name": "finish",
        "description": "End the interaction after reviewing the current tool results.",
        "arguments": {},
    },
]


class ToolDecision(BaseModel):
    """One model-selected action from the allowlisted D1 tools."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = {}


class ToolSelector(Protocol):
    """Provider adapter that turns conversation state into one typed tool decision."""

    def __call__(
        self,
        request: str,
        state: dict[str, Any],
        available_tools: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class TaskSolver(Protocol):
    def __call__(
        self, task: TaskSpec, output_dir: Path
    ) -> tuple[MetricsSpec, Path]: ...


def run_d1_agent(
    request: str,
    select_tool: ToolSelector,
    output_dir: Path,
    *,
    max_steps: int = 6,
    solver: TaskSolver = solve_task,
    expected_task_id: str | None = None,
) -> dict[str, Any]:
    """Run a bounded D1 loop in which the model chooses each typed tool call."""

    if max_steps < 1 or max_steps > 12:
        raise ValueError("max_steps must be between 1 and 12")

    state: dict[str, Any] = {
        "status": "started",
        "expected_task_id": expected_task_id,
        "trace": [],
    }
    handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "set_task": lambda arguments: _set_task(arguments, expected_task_id),
        "run_solver": lambda _arguments: _run_solver(state, output_dir, solver),
        "evaluate": lambda _arguments: _evaluate(state, output_dir),
        "finish": lambda _arguments: _finish(state),
    }

    for step in range(1, max_steps + 1):
        try:
            decision = ToolDecision.model_validate(
                select_tool(request, _public_state(state), TOOL_CONTRACTS)
            )
        except ValidationError as exc:
            state["status"] = "invalid_model_tool_call"
            state["last_error"] = str(exc)
            state["trace"].append(
                {"step": step, "tool": None, "error": str(exc)}
            )
            continue

        handler = handlers.get(decision.tool)
        if handler is None:
            result = {"error": f"tool is not allowed: {decision.tool}"}
        else:
            try:
                result = handler(decision.arguments)
            except (NgException, OSError, ValueError, ArithmeticError, RuntimeError) as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}

        state.update({key: value for key, value in result.items() if key != "error"})
        if "error" in result:
            state["last_error"] = result["error"]
        else:
            state.pop("last_error", None)
        state["trace"].append(
            {
                "step": step,
                "tool": decision.tool,
                "arguments": decision.arguments,
                "result": result,
            }
        )
        if state.get("status") in {"passed", "finished_incomplete"}:
            break
    else:
        state["status"] = "max_steps_reached"

    return _public_state(state)


def _set_task(
    arguments: dict[str, Any], expected_task_id: str | None = None
) -> dict[str, Any]:
    task = parse_task(arguments.get("task", arguments))
    if expected_task_id and task.task_id != expected_task_id:
        raise ValueError(
            f"request requires task_id={expected_task_id}, received {task.task_id}"
        )
    return {"task": task.model_dump(mode="json"), "status": "task_ready"}


def _run_solver(
    state: dict[str, Any], output_dir: Path, solver: TaskSolver
) -> dict[str, Any]:
    if "task" not in state:
        return {"error": "set_task must succeed before run_solver"}
    task = parse_task(state["task"])
    run_id = datetime.now(UTC).strftime("D1-%Y%m%dT%H%M%S%fZ")
    figure_dir = output_dir / "figures" / run_id
    metrics, image_path = solver(task, figure_dir)
    return {
        "metrics": metrics.model_dump(mode="json"),
        "image_path": str(image_path),
        "run_id": run_id,
        "status": "solved",
    }


def _evaluate(state: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if "task" not in state or "metrics" not in state or "image_path" not in state:
        return {"error": "run_solver must succeed before evaluate"}
    task = parse_task(state["task"])
    metrics = parse_metrics(task, state["metrics"])
    failures = evaluate_task(task, metrics.model_dump(mode="json"))
    report_path = write_report(
        task,
        metrics,
        failures,
        output_dir,
        Path(state["image_path"]),
        state["run_id"],
        architecture="D1",
    )
    return {
        "report_path": str(report_path),
        "status": "passed" if not failures else "verification_failed",
        "verification_failures": failures,
    }


def _finish(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("status") == "passed":
        return {"status": "passed"}
    return {"status": "finished_incomplete"}


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    """Remove internal objects before passing state back to a model adapter."""

    return {
        key: value
        for key, value in state.items()
        if key not in {"private_handles"}
    }
