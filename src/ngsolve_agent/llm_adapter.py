"""Small provider-neutral JSON chat adapter for local and OpenAI-compatible models."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, SecretStr

_MAX_LLM_RESPONSE_BYTES = 1_048_576
_MAX_COMPLETION_TOKENS = 256


class LLMSettings(BaseModel):
    """Runtime LLM settings; credentials are read from environment, never serialized."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    provider: Literal["ollama", "openai-compatible"] = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str
    api_key: SecretStr | None = None
    timeout_seconds: float = 120.0

    @classmethod
    def from_environment(cls) -> LLMSettings:
        provider = os.getenv("NGSOLVE_LLM_PROVIDER", "ollama")
        default_url = (
            "http://127.0.0.1:11434"
            if provider == "ollama"
            else "https://api.openai.com/v1"
        )
        return cls(
            provider=provider,
            base_url=os.getenv("NGSOLVE_LLM_BASE_URL", default_url),
            model=os.getenv("NGSOLVE_LLM_MODEL", ""),
            api_key=SecretStr(os.environ["NGSOLVE_LLM_API_KEY"])
            if os.getenv("NGSOLVE_LLM_API_KEY")
            else None,
            timeout_seconds=float(os.getenv("NGSOLVE_LLM_TIMEOUT", "120")),
        )


@dataclass
class JsonChatClient:
    """Make JSON-only chat calls and retain safe per-call metadata for reports."""

    settings: LLMSettings
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, *, system: str, user: str) -> dict[str, Any]:
        started = time.perf_counter()
        if not self.settings.model.strip():
            raise ValueError("set NGSOLVE_LLM_MODEL to the installed/available model name")

        if self.settings.provider == "ollama":
            url = self.settings.base_url.rstrip("/") + "/api/chat"
            body = {
                "model": self.settings.model,
                "stream": False,
                "format": "json",
                "think": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {
                    "temperature": 0,
                    "num_ctx": 4096,
                    "num_predict": _MAX_COMPLETION_TOKENS,
                },
            }
        else:
            url = self.settings.base_url.rstrip("/") + "/chat/completions"
            body = {
                "model": self.settings.model,
                "stream": False,
                "temperature": 0,
                "max_tokens": _MAX_COMPLETION_TOKENS,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }

        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key.get_secret_value()}"
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                response_bytes = response.read(_MAX_LLM_RESPONSE_BYTES + 1)
                if len(response_bytes) > _MAX_LLM_RESPONSE_BYTES:
                    raise ValueError(
                        f"LLM response exceeded {_MAX_LLM_RESPONSE_BYTES} bytes"
                    )
                payload = json.loads(response_bytes.decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"LLM HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"could not reach LLM endpoint {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"LLM request timed out after {self.settings.timeout_seconds:g}s") from exc

        if self.settings.provider == "ollama":
            content = payload.get("message", {}).get("content", "")
            usage = {
                "prompt_tokens": payload.get("prompt_eval_count"),
                "completion_tokens": payload.get("eval_count"),
            }
        else:
            choices = payload.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            usage = payload.get("usage", {})

        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("LLM returned content that is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise TypeError("LLM JSON response must be an object")

        self.calls.append(
            {
                "provider": self.settings.provider,
                "model": self.settings.model,
                "duration_seconds": time.perf_counter() - started,
                "usage": usage,
            }
        )
        return parsed


def make_task_interpreter(client: JsonChatClient):
    """Create D4's constrained natural-language task adapter."""

    def interpret(request: str, previous_errors: list[str]) -> dict[str, Any]:
        system = (
            "Convert the request into exactly one task from this allowlist; never "
            "pretend one PDE is another. Return only a JSON task object including "
            "task_id. Supported tasks: (1) task_id=poisson_unit_square_smooth: the "
            "manufactured -Delta u = 2*pi^2*sin(pi*x)*sin(pi*y) problem on the unit "
            "square with zero Dirichlet boundaries; optional keys maxh (0.025..1, "
            "default 0.2), "
            "order (default 2), grid_samples (default 64). (2) "
            "task_id=ngsolve_tutorial_stokes_flow_around_cylinder: steady "
            "2D Stokes flow in the tutorial channel around a cylinder using Taylor-" 
            "Hood P2/P1 elements; optional keys maxh (0.025..0.2, default 0.05), "
            "Copy explicit mesh settings. Evaluator thresholds are fixed and cannot "
            "be changed by the task. "
            "If the request asks for anything else, return JSON {\"unsupported\": true}."
        )
        user = json.dumps(
            {"request": request, "previous_errors": previous_errors}, ensure_ascii=False
        )
        return client.complete(system=system, user=user)

    return interpret


def make_tool_selector(client: JsonChatClient):
    """Create D1's bounded model-directed selector over its allowlisted tools."""

    def select(request: str, state: dict[str, Any], tools: list[dict[str, Any]]):
        system = (
            "You control one allowlisted Poisson or tutorial Stokes-cylinder solve "
            "through tools. Select the task_id matching the requested physics. "
            "Choose exactly one next tool and return JSON {tool, arguments}. "
            "When calling set_task, use the expected_task_id from state and copy "
            "every explicitly requested parameter that applies to that task. Do not "
            "nest settings under descriptive labels. Omit unspecified fields so "
            "their documented defaults apply. "
            "Use set_task first; then run_solver; then evaluate; call finish only "
            "after status=passed. If a tool result contains an error, choose a valid "
            "recovery action. Do not claim a solve is correct before evaluate passes."
        )
        user = json.dumps(
            {"request": request, "state": state, "available_tools": tools},
            ensure_ascii=False,
        )
        return client.complete(system=system, user=user)

    return select
