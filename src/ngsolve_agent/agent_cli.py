"""Run the real D1 or D4 Poisson agent through a configured JSON chat model."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .d1_agent import run_d1_agent
from .d3_agent import run_d3_agent
from .llm_adapter import JsonChatClient, LLMSettings, make_task_interpreter, make_tool_selector
from .workflow import build_d4_workflow

_UNSUPPORTED_PROBLEM_MARKERS = (
    "navier-stokes",
    "navier stokes",
    "maxwell",
    "elasticity",
    "elastic",
    "helmholtz",
    "heat equation",
    "wave equation",
)


def _classify_request(request: str) -> str | None:
    normalized = request.casefold().replace("-", " ")
    if any(marker in normalized for marker in _UNSUPPORTED_PROBLEM_MARKERS):
        return None
    unit_square_domain = "unit square" in normalized or bool(
        re.search(
            r"\[\s*0(?:\.0+)?\s*,\s*1(?:\.0+)?\s*\]\s*(?:\^\s*2|²)",
            normalized,
        )
    )
    mentions_poisson = "poisson" in normalized and unit_square_domain
    mentions_stokes = "stokes" in normalized and "cylinder" in normalized
    if mentions_poisson == mentions_stokes:
        return None
    if mentions_poisson:
        return "poisson_unit_square_smooth"
    if mentions_stokes:
        return "ngsolve_tutorial_stokes_flow_around_cylinder"
    return None


def run_agent(
    architecture: str,
    request: str,
    output_dir: Path,
    settings: LLMSettings,
) -> dict[str, Any]:
    """Invoke one architecture and persist its model metadata and execution trace."""

    started = time.perf_counter()
    client = JsonChatClient(settings)
    run_id = datetime.now(UTC).strftime(f"{architecture.upper()}-%Y%m%dT%H%M%S%fZ")
    agent_dir = output_dir / "agent_runs" / run_id
    agent_dir.mkdir(parents=True, exist_ok=True)

    expected_task_id = _classify_request(request)
    if expected_task_id is None:
        status = "unsupported_request"
        result = {
            "status": status,
            "reason": (
                "This agent supports only explicitly requested unit-square Poisson "
                "or tutorial Stokes-cylinder problems. No model call or numerical "
                "solve was made."
            ),
            "trace": [
                {
                    "step": 0,
                    "action": "scope_check",
                    "passed": False,
                    "supported_task_ids": [
                        "poisson_unit_square_smooth",
                        "ngsolve_tutorial_stokes_flow_around_cylinder",
                    ],
                }
            ],
        }
    else:
        try:
            if architecture == "d1":
                result = run_d1_agent(
                    request,
                    make_tool_selector(client),
                    agent_dir,
                    expected_task_id=expected_task_id,
                )
            elif architecture == "d3":
                project_root = Path(__file__).resolve().parents[2]
                result = run_d3_agent(
                    request,
                    client,
                    agent_dir / "d3",
                    project_root,
                    expected_task_id,
                )
            else:
                graph = build_d4_workflow(
                    make_task_interpreter(client),
                    agent_dir,
                )
                result = graph.invoke(
                    {
                        "request": request,
                        "expected_task_id": expected_task_id,
                        "errors": [],
                        "attempt": 0,
                        "trace": [],
                    }
                )
            status = result.get("status", "unknown")
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            result = {"status": "agent_error", "error": f"{type(exc).__name__}: {exc}"}
            status = "agent_error"

    report = {
        "run_id": run_id,
        "architecture": architecture.upper(),
        "request": request,
        "provider": settings.provider,
        "model": settings.model,
        "duration_seconds": time.perf_counter() - started,
        "status": status,
        "agent_result": result,
        "llm_calls": client.calls,
    }
    report_path = agent_dir / "agent_run.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"agent report: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("d1", "d3", "d4"), required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    settings = LLMSettings.from_environment()
    report = run_agent(args.architecture, args.request, args.output_dir, settings)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
