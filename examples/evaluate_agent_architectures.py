"""Compare D1 and D4 on the same fixed natural-language Poisson cases."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

from ngsolve_agent.agent_cli import run_agent
from ngsolve_agent.llm_adapter import LLMSettings


def evaluate_case(report: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    result = report.get("agent_result", {})
    actual = result.get("task", {})
    mismatches: list[dict[str, Any]] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float) and isinstance(actual_value, (float, int)):
            matches = math.isclose(float(actual_value), expected_value, rel_tol=0, abs_tol=1e-10)
        else:
            matches = actual_value == expected_value
        if not matches:
            mismatches.append(
                {"field": key, "expected": expected_value, "actual": actual_value}
            )
    calls = report.get("llm_calls", [])
    return {
        "status": report.get("status"),
        "task_parameters_match": not mismatches,
        "parameter_mismatches": mismatches,
        "llm_call_count": len(calls),
        "llm_duration_seconds": sum(call.get("duration_seconds", 0.0) for call in calls),
        "agent_duration_seconds": report.get("duration_seconds", 0.0),
        "numerical_metrics": result.get("metrics"),
        "report_path": result.get("report_path"),
        "passed": report.get("status") == "passed" and not mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="case ID; repeat to select multiple")
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("d1", "d3", "d4"),
        default=["d1", "d4"],
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="independent runs per case/architecture (1..10)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be between 1 and 10")

    settings = LLMSettings.from_environment()
    if not settings.model:
        parser.error("set NGSOLVE_LLM_MODEL before running the model comparison")

    project_root = Path(__file__).resolve().parents[1]
    cases_path = project_root / "data" / "agent_task_cases.json"
    all_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    selected = [case for case in all_cases if not args.case or case["case_id"] in args.case]
    if args.case and {case["case_id"] for case in selected} != set(args.case):
        parser.error("one or more requested case IDs were not found")

    suite_id = datetime.now(UTC).strftime("agent-compare-%Y%m%dT%H%M%S%fZ")
    suite_dir = args.output_dir / "agent_benchmarks" / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    for case in selected:
        for repeat in range(1, args.repeats + 1):
            for architecture in args.architectures:
                case_output = (
                    suite_dir
                    / case["case_id"]
                    / f"repeat-{repeat:02d}"
                    / architecture
                )
                report = run_agent(
                    architecture,
                    case["request"],
                    case_output,
                    settings,
                )
                outcome = evaluate_case(report, case["expected_task"])
                outcomes.append(
                    {
                        "case_id": case["case_id"],
                        "repeat": repeat,
                        "architecture": architecture.upper(),
                        **outcome,
                    }
                )

    aggregates: list[dict[str, Any]] = []
    for case in selected:
        for architecture in args.architectures:
            samples = [
                item
                for item in outcomes
                if item["case_id"] == case["case_id"]
                and item["architecture"] == architecture.upper()
            ]
            aggregates.append(
                {
                    "case_id": case["case_id"],
                    "architecture": architecture.upper(),
                    "runs": len(samples),
                    "passed_runs": sum(item["passed"] for item in samples),
                    "pass_rate": sum(item["passed"] for item in samples) / len(samples),
                    "parameter_match_rate": sum(
                        item["task_parameters_match"] for item in samples
                    )
                    / len(samples),
                    "mean_llm_call_count": fmean(
                        item["llm_call_count"] for item in samples
                    ),
                    "mean_llm_duration_seconds": fmean(
                        item["llm_duration_seconds"] for item in samples
                    ),
                    "median_llm_duration_seconds": median(
                        item["llm_duration_seconds"] for item in samples
                    ),
                    "mean_agent_duration_seconds": fmean(
                        item["agent_duration_seconds"] for item in samples
                    ),
                    "median_agent_duration_seconds": median(
                        item["agent_duration_seconds"] for item in samples
                    ),
                }
            )

    summary = {
        "suite_id": suite_id,
        "dataset": str(cases_path.relative_to(project_root)),
        "provider": settings.provider,
        "model": settings.model,
        "repeats": args.repeats,
        "outcomes": outcomes,
        "aggregates": aggregates,
        "all_passed": bool(outcomes) and all(item["passed"] for item in outcomes),
    }
    summary_path = suite_dir / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"comparison report: {summary_path}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
