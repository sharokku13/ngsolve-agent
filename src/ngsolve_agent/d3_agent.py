"""D3: generate a bounded Python driver, validate it, then run it in a child process."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .llm_adapter import JsonChatClient
from .schemas import TaskSpec, parse_metrics, parse_task
from .task_registry import evaluate_task

_DRIVER_API_MODULE = "ngsolve_agent.generated_driver_api"
_MAX_CHILD_STDOUT_BYTES = 64_000
_MAX_CHILD_STDERR_BYTES = 16_000


def _run_child_bounded(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
) -> tuple[int, str, str, bool, bool]:
    """Run a child while draining its pipes with strict in-memory output caps."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()

    def drain(pipe, buffer: bytearray, limit: int) -> None:
        while chunk := pipe.read(4096):
            remaining = limit - len(buffer)
            if len(chunk) > remaining:
                buffer.extend(chunk[:remaining])
                if not overflow.is_set():
                    overflow.set()
                    try:
                        process.kill()
                    except OSError:
                        pass
            elif remaining > 0:
                buffer.extend(chunk)

    readers = (
        threading.Thread(
            target=drain, args=(process.stdout, stdout, _MAX_CHILD_STDOUT_BYTES), daemon=True
        ),
        threading.Thread(
            target=drain, args=(process.stderr, stderr, _MAX_CHILD_STDERR_BYTES), daemon=True
        ),
    )
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            process.kill()
        except OSError:
            pass
        process.wait()
    for reader in readers:
        reader.join()
    process.stdout.close()
    process.stderr.close()
    return (
        process.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        timed_out,
        overflow.is_set(),
    )
_ALLOWED_PARAMETERS = {
    "task_id",
    "maxh",
    "order",
    "grid_samples",
}


def validate_and_canonicalize_driver(
    source: str, expected_task_id: str
) -> tuple[str, TaskSpec]:
    """Accept one tiny driver program and rebuild it from validated literal values."""

    if len(source) > 16_000:
        raise ValueError("generated driver exceeds 16,000 characters")
    try:
        module = ast.parse(source, mode="exec")
    except (SyntaxError, RecursionError, MemoryError, ValueError) as exc:
        reason = getattr(exc, "msg", None) or str(exc) or type(exc).__name__
        raise ValueError(f"generated driver is not valid Python: {reason}") from exc
    if sum(1 for _ in ast.walk(module)) > 128:
        raise ValueError("generated driver exceeds the 128-node AST limit")
    if len(module.body) != 2:
        raise ValueError("driver must contain exactly one approved import and one run call")

    import_node, call_statement = module.body
    if not (
        isinstance(import_node, ast.ImportFrom)
        and import_node.module == _DRIVER_API_MODULE
        and import_node.level == 0
        and len(import_node.names) == 1
        and import_node.names[0].name == "run"
        and import_node.names[0].asname is None
    ):
        raise ValueError("driver may import only run from the approved driver API")

    if not isinstance(call_statement, ast.Expr) or not isinstance(
        call_statement.value, ast.Call
    ):
        raise ValueError(  # noqa: TRY004
            "driver's second statement must call the approved run function"
        )
    call = call_statement.value
    if not isinstance(call.func, ast.Name) or call.func.id != "run":
        raise ValueError("driver may call only the approved run function")
    if call.args:
        raise ValueError("driver must pass task data as named literal parameters")

    task_data: dict[str, object] = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in _ALLOWED_PARAMETERS:
            raise ValueError("driver contains an unsupported or expanded keyword argument")
        if keyword.arg in task_data:
            raise ValueError(f"driver repeats parameter {keyword.arg!r}")
        if not isinstance(keyword.value, ast.Constant) or type(keyword.value.value) not in (
            str,
            int,
            float,
        ):
            raise ValueError(f"driver parameter {keyword.arg!r} must be a literal value")
        task_data[keyword.arg] = keyword.value.value

    if task_data.get("task_id") != expected_task_id:
        raise ValueError("generated task_id does not match the deterministic request scope")
    task = parse_task(task_data)
    if task.task_id != expected_task_id:
        raise ValueError("validated task_id does not match the deterministic request scope")

    canonical_task = task.model_dump(mode="json")
    arguments = ", ".join(
        f"{name}={value!r}" for name, value in canonical_task.items()
    )
    canonical_source = (
        f"from {_DRIVER_API_MODULE} import run\n"
        f"run({arguments})\n"
    )
    return canonical_source, task


def run_d3_agent(
    request: str,
    client: JsonChatClient,
    output_dir: Path,
    project_root: Path,
    expected_task_id: str,
    *,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Generate, validate, canonicalize, and execute a driver for one supported task."""

    run_id = datetime.now(UTC).strftime("D3-%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "generated_driver_candidate.py"
    validated_path = output_dir / "generated_driver.py"
    prompt = {
        "request": request,
        "expected_task_id": expected_task_id,
        "supported_task_contracts": {
            "poisson_unit_square_smooth": {
                "description": "manufactured -Delta u = 2*pi^2*sin(pi*x)*sin(pi*y) on unit square, zero boundary",
                "parameters": {
                    "maxh": "number in [0.025, 1.0], default 0.2",
                    "order": "integer 1..6, default 2",
                    "grid_samples": "integer 32..512, default 64",
                },
            },
            "ngsolve_tutorial_stokes_flow_around_cylinder": {
                "description": "steady 2D Stokes flow in NGSolve tutorial channel around cylinder, Taylor-Hood P2/P1",
                "parameters": {
                    "maxh": "number in [0.025, 0.2], default 0.05",
                },
            },
        },
        "driver_contract": (
            f"Generate only two Python statements: `from {_DRIVER_API_MODULE} import run` "
            "and a call like `run(task_id='poisson_unit_square_smooth', maxh=0.1, "
            "order=3)`. Pass every setting as its own named argument; never nest them "
            "in a `parameter` or `parameters` dictionary. Use literal strings/numbers only, "
            "the exact expected_task_id, and copy every explicit supported parameter. "
            "Do not add imports, variables, comments with instructions, functions, "
            "filesystem access, shell commands, or other executable statements. "
            "Return JSON with the single key driver_code containing the Python source."
        ),
    }
    system = (
        "Generate a minimal Python main driver for exactly one allowlisted NGSolve task. "
        "The caller will validate and rebuild your source before execution. Do not write "
        "solver implementations. Return only a JSON object with driver_code."
    )
    response = client.complete(system=system, user=json.dumps(prompt, ensure_ascii=False))
    source = response.get("driver_code")
    if not isinstance(source, str):
        return {
            "status": "invalid_driver",
            "error": "model response is missing string field driver_code",
            "run_id": run_id,
        }
    source_path.write_text(source, encoding="utf-8")
    try:
        canonical_source, task = validate_and_canonicalize_driver(
            source, expected_task_id
        )
    except (SyntaxError, TypeError, ValueError) as exc:
        return {
            "status": "invalid_driver",
            "error": str(exc),
            "run_id": run_id,
            "generated_driver_candidate": str(source_path),
            "llm_calls": len(client.calls),
        }

    validated_path.write_text(canonical_source, encoding="utf-8")
    execution_dir = output_dir / "execution"
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE")
        if key in os.environ
    }
    env["NGSOLVE_D3_OUTPUT_DIR"] = str(execution_dir.resolve())
    env["NGSOLVE_D3_RUN_ID"] = run_id
    try:
        returncode, stdout, stderr, timed_out, output_limited = _run_child_bounded(
            [sys.executable, "-I", str(validated_path.resolve())],
            cwd=project_root,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "driver_timeout",
            "error": f"generated driver exceeded {timeout_seconds:g} seconds",
            "run_id": run_id,
            "generated_driver_candidate": str(source_path),
            "validated_driver": str(validated_path),
            "task": task.model_dump(mode="json"),
        }

    if timed_out:
        return {
            "status": "driver_timeout",
            "error": f"generated driver exceeded {timeout_seconds:g} seconds",
            "run_id": run_id,
            "generated_driver_candidate": str(source_path),
            "validated_driver": str(validated_path),
            "task": task.model_dump(mode="json"),
        }
    if output_limited:
        return {
            "status": "driver_output_limit_exceeded",
            "error": (
                "driver output exceeded the bounded limits "
                f"(stdout={_MAX_CHILD_STDOUT_BYTES} bytes, "
                f"stderr={_MAX_CHILD_STDERR_BYTES} bytes)"
            ),
            "run_id": run_id,
            "generated_driver_candidate": str(source_path),
            "validated_driver": str(validated_path),
        }
    lines = [line for line in stdout.splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        result = {}
    if returncode != 0 or not isinstance(result, dict):
        return {
            "status": "driver_execution_failed",
            "error": stderr[-_MAX_CHILD_STDERR_BYTES:]
            or "generated driver returned no JSON result",
            "returncode": returncode,
            "run_id": run_id,
            "generated_driver_candidate": str(source_path),
            "validated_driver": str(validated_path),
            "stdout_tail": stdout[-2000:],
        }
    returned_task = result.get("task")
    if returned_task != task.model_dump(mode="json"):
        return {
            "status": "driver_result_invalid",
            "error": "child task does not match the validated driver task",
            "run_id": run_id,
        }
    try:
        metrics = parse_metrics(task, result.get("metrics"))
        failures = evaluate_task(task, metrics.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        return {
            "status": "driver_result_invalid",
            "error": f"child metrics failed parent validation: {exc}",
            "run_id": run_id,
        }
    return {
        **result,
        "metrics": metrics.model_dump(mode="json"),
        "status": "passed" if not failures else "verification_failed",
        "verification_failures": failures,
        "generated_driver_candidate": str(source_path),
        "validated_driver": str(validated_path),
        "llm_calls": len(client.calls),
    }
