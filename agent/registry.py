"""Tool registry: structured inputs/outputs, permissions, timeouts, confirmation and structured logging.

Every tool call from the agent (LLM or rule-based) goes through ToolRegistry.execute(), which enforces:
  * argument validation (pydantic)            -> status "invalid"
  * approved-namespace restriction            -> status "denied"
  * explicit confirmation for mutating tools  -> status "confirmation_required"
  * per-tool timeout                          -> status "timeout"
  * structured logging (name, status, duration, request id; secrets redacted)
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from agent.config import get_settings
from agent.errors import ToolError
from agent.logging_utils import log_event, redact, redact_text
from agent.schemas import ToolResult


@dataclass
class ToolContext:
    request_id: str
    confirmed: bool = False  # set ONLY from the user's request, never by the LLM


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[Any], dict]
    module: str
    mutating: bool = False
    timeout_s: float | None = None

    def spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description + (" [STATE-CHANGING: requires user confirmation]" if self.mutating else ""),
            "input_schema": self.args_model.model_json_schema(),
        }


class _Reject(Exception):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _short(err: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors())


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ---- registration -------------------------------------------------
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def tool(self, *, name: str, description: str, args: type[BaseModel], module: str,
             mutating: bool = False, timeout_s: float | None = None):
        def decorator(fn: Callable[[Any], dict]):
            self.register(Tool(name, description, args, fn, module, mutating, timeout_s))
            return fn
        return decorator

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def describe(self) -> list[dict]:
        return [
            {"name": t.name, "module": t.module, "permission": "write" if t.mutating else "read",
             "description": t.description}
            for t in sorted(self._tools.values(), key=lambda t: (t.module, t.name))
        ]

    # ---- execution ----------------------------------------------------
    def execute(self, name: str, args: dict | None, ctx: ToolContext | None = None) -> ToolResult:
        ctx = ctx or ToolContext(request_id=uuid.uuid4().hex[:12])
        started = time.perf_counter()
        status, data, error = "ok", None, None
        try:
            tool = self._tools.get(name)
            if tool is None:
                raise _Reject("invalid", f"Unknown tool '{name}'")
            try:
                parsed = tool.args_model(**(args or {}))
            except ValidationError as exc:
                raise _Reject("invalid", "Invalid arguments: " + _short(exc)) from exc

            settings = get_settings()
            namespace = getattr(parsed, "namespace", None)
            if namespace is not None and namespace != settings.k8s_namespace:
                raise _Reject("denied", f"Namespace '{namespace}' is not approved; only '{settings.k8s_namespace}' is allowed.")
            if tool.mutating and not ctx.confirmed:
                raise _Reject(
                    "confirmation_required",
                    f"'{name}' changes system state. Explicit user confirmation is required (resend with confirm=true).",
                )
            data = self._run(tool, parsed, tool.timeout_s or settings.tool_timeout_s)
        except _Reject as rej:
            status, error = rej.status, rej.message
        except FuturesTimeout:
            status, error = "timeout", f"Tool '{name}' timed out."
        except ToolError as exc:
            status, error = "error", redact_text(str(exc))
        except Exception as exc:  # unexpected failure: report, never crash the agent
            status, error = "error", redact_text(f"{type(exc).__name__}: {exc}")

        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        result = ToolResult(tool=name, status=status, data=data, error=error,
                            duration_ms=duration_ms, request_id=ctx.request_id)
        log_event("tool_call", tool=name, status=status, duration_ms=duration_ms,
                  request_id=ctx.request_id, args=redact(args or {}), error=error)
        return result

    @staticmethod
    def _run(tool: Tool, parsed: BaseModel, timeout_s: float) -> dict:
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(tool.func, parsed)
            return future.result(timeout=timeout_s)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


REGISTRY = ToolRegistry()
