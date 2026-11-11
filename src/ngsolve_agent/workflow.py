"""D4: deterministic orchestration around a bounded task-interpreter call."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from netgen.libngpy._meshing import NgException
from pydantic import ValidationError

from .reporting import write_report
from .schemas import MetricsSpec, TaskSpec, parse_metrics, parse_task
from .task_registry import evaluate_task, solve_task


class TaskInterpreter(Protocol):
    """Adapter boundary for any LLM provider; credentials stay outside the graph."""

    def __call__(self, request: str, previous_errors: list[str]) -> dict[str, Any]: ...


class TaskSolver(Protocol):
    """Callable solver boundary, injectable for fast workflow tests."""

    def __call__(
        self, task: TaskSpec, output_dir: Path
    ) -> tuple[MetricsSpec, Path]: ...


class D4State(TypedDict, total=False):
    request: str
    expected_task_id: str
    candidate: dict[str, Any]
    task: dict[str, Any]
    attempt: int
    max_attempts: int
    errors: list[str]
    status: str
    metrics: dict[str, Any]
    image_path: str
    report_path: str
    run_id: str
    trace: list[dict[str, Any]]


def build_d4_workflow(
    interpret_task: TaskInterpreter,
    output_dir: Path,
    *,
    max_attempts: int = 2,
    solver: TaskSolver = solve_task,
):
    """Build a deterministic graph with bounded interpretation and recovery.

    The graph owns the order: interpret -> validate -> solve -> evaluate. The LLM only
    proposes typed task data at a single, bounded node. All FEM work and evaluation are
    deterministic Python functions.
    """

    if max_attempts < 1 or max_attempts > 3:
        raise ValueError("max_attempts must be between 1 and 3")

    def interpret_node(state: D4State) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        candidate = interpret_task(state["request"], errors)
        attempt = state.get("attempt", 0) + 1
        return {
            "candidate": candidate,
            "attempt": attempt,
            "trace": [
                *state.get("trace", []),
                {"step": attempt, "action": "interpret_task", "candidate": candidate},
            ],
        }

    def validate_node(state: D4State) -> dict[str, Any]:
        if state["candidate"].get("unsupported") is True:
            return {
                "status": "unsupported_request",
                "errors": [
                    *state.get("errors", []),
                    "the model classified the requested problem as unsupported",
                ],
                "trace": [
                    *state.get("trace", []),
                    {"step": state.get("attempt", 0), "action": "validate", "passed": False},
                ],
            }
        try:
            task = parse_task(state["candidate"])
        except (ValidationError, ValueError) as exc:
            return {
                "status": "invalid_task",
                "errors": [*state.get("errors", []), str(exc)],
                "trace": [
                    *state.get("trace", []),
                    {"step": state.get("attempt", 0), "action": "validate", "passed": False},
                ],
            }
        expected_task_id = state.get("expected_task_id")
        if expected_task_id and task.task_id != expected_task_id:
            message = (
                f"request was classified as {expected_task_id!r}, but model selected "
                f"{task.task_id!r}"
            )
            return {
                "status": "invalid_task",
                "errors": [*state.get("errors", []), message],
                "trace": [
                    *state.get("trace", []),
                    {"step": state.get("attempt", 0), "action": "validate", "passed": False},
                ],
            }
        return {
            "task": task.model_dump(mode="json"),
            "status": "validated",
            "trace": [
                *state.get("trace", []),
                {"step": state.get("attempt", 0), "action": "validate", "passed": True},
            ],
        }

    def solve_node(state: D4State) -> dict[str, Any]:
        task = parse_task(state["task"])
        run_id = datetime.now(UTC).strftime("D4-%Y%m%dT%H%M%S%fZ")
        figure_dir = output_dir / "figures" / run_id
        try:
            metrics, image_path = solver(task, figure_dir)
        except (NgException, OSError, ValueError, ArithmeticError, RuntimeError) as exc:
            return {
                "status": "solver_error",
                "errors": [*state.get("errors", []), f"solver error: {exc}"],
                "run_id": run_id,
                "trace": [
                    *state.get("trace", []),
                    {"step": state.get("attempt", 0), "action": "solve", "passed": False},
                ],
            }
        return {
            "metrics": metrics.model_dump(mode="json"),
            "image_path": str(image_path),
            "run_id": run_id,
            "status": "solved",
            "trace": [
                *state.get("trace", []),
                {
                    "step": state.get("attempt", 0),
                    "action": "solve",
                    "passed": True,
                    "task_id": task.task_id,
                    "ndof": metrics.ndof,
                    "metrics": metrics.model_dump(mode="json"),
                },
            ],
        }

    def evaluate_node(state: D4State) -> dict[str, Any]:
        task = parse_task(state["task"])
        metrics = parse_metrics(task, state["metrics"])
        failures = evaluate_task(task, metrics.model_dump(mode="json"))
        image_path = Path(state["image_path"])
        report_path = write_report(
            task,
            metrics,
            failures,
            output_dir,
            image_path,
            state["run_id"],
            architecture="D4",
        )
        if failures:
            return {
                "status": "verification_failed",
                "errors": [*state.get("errors", []), *failures],
                "report_path": str(report_path),
                "trace": [
                    *state.get("trace", []),
                    {"step": state.get("attempt", 0), "action": "evaluate", "passed": False},
                ],
            }
        return {
            "status": "passed",
            "report_path": str(report_path),
            "trace": [
                *state.get("trace", []),
                {"step": state.get("attempt", 0), "action": "evaluate", "passed": True},
            ],
        }

    def after_validation(state: D4State) -> str:
        if state.get("status") == "validated":
            return "solve"
        if state.get("status") == "unsupported_request":
            return END
        if state.get("attempt", 0) < max_attempts:
            return "interpret"
        return END

    def after_solve(state: D4State) -> str:
        if state.get("status") == "solved":
            return "evaluate"
        if state.get("attempt", 0) < max_attempts:
            return "interpret"
        return END

    def after_evaluation(state: D4State) -> str:
        if state.get("status") == "passed":
            return END
        if state.get("attempt", 0) < max_attempts:
            return "interpret"
        return END

    graph = StateGraph(D4State)
    graph.add_node("interpret", interpret_node)
    graph.add_node("validate", validate_node)
    graph.add_node("solve", solve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_edge(START, "interpret")
    graph.add_edge("interpret", "validate")
    graph.add_conditional_edges(
        "validate",
        after_validation,
        {"solve": "solve", "interpret": "interpret", END: END},
    )
    graph.add_conditional_edges(
        "solve",
        after_solve,
        {"evaluate": "evaluate", "interpret": "interpret", END: END},
    )
    graph.add_conditional_edges(
        "evaluate",
        after_evaluation,
        {"interpret": "interpret", END: END},
    )
    return graph.compile()
