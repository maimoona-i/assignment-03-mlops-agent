import logging
import time

from pydantic import BaseModel

from agent.errors import ToolError
from agent.registry import ToolContext, ToolRegistry


class Args(BaseModel):
    x: int
    namespace: str | None = None


def make(fn, **kw):
    reg = ToolRegistry()
    reg.tool(name="t", description="d", args=Args, module="test", **kw)(fn)
    return reg


def test_successful_tool_execution():
    reg = make(lambda a: {"double": a.x * 2})
    res = reg.execute("t", {"x": 4}, ToolContext("rid-1"))
    assert res.status == "ok" and res.data == {"double": 8} and res.request_id == "rid-1"


def test_invalid_arguments_are_rejected():
    res = make(lambda a: {}).execute("t", {"x": "not-an-int"})
    assert res.status == "invalid" and "x" in res.error


def test_unknown_tool_is_invalid():
    assert ToolRegistry().execute("nope", {}).status == "invalid"


def test_tool_timeout_is_reported():
    def slow(a):
        time.sleep(0.5)
        return {}

    res = make(slow, timeout_s=0.05).execute("t", {"x": 1})
    assert res.status == "timeout"


def test_service_failures_are_reported_not_raised():
    def broken(a):
        raise ToolError("mlflow unreachable")

    def crash(a):
        raise RuntimeError("kaboom")

    assert make(broken).execute("t", {"x": 1}).status == "error"
    res = make(crash).execute("t", {"x": 1})
    assert res.status == "error" and "RuntimeError" in res.error


def test_mutating_tool_requires_confirmation():
    calls = []
    reg = make(lambda a: calls.append(1) or {"done": True}, mutating=True)
    assert reg.execute("t", {"x": 1}, ToolContext("r", confirmed=False)).status == "confirmation_required"
    assert calls == []
    assert reg.execute("t", {"x": 1}, ToolContext("r", confirmed=True)).status == "ok"


def test_unapproved_namespace_is_denied_even_when_confirmed():
    reg = make(lambda a: {}, mutating=True)
    res = reg.execute("t", {"x": 1, "namespace": "kube-system"}, ToolContext("r", confirmed=True))
    assert res.status == "denied"


def test_structured_logs_have_required_fields_and_no_secrets(caplog):
    caplog.set_level(logging.INFO, logger="mlops_agent")
    reg = make(lambda a: {})
    secret = "sk-abcdef1234567890"
    reg.execute("t", {"x": 1, "api_key": secret, "note": f"Bearer {secret}"}, ToolContext("req-42"))
    rec = next(r for r in caplog.records if r.getMessage() == "tool_call")
    f = rec.fields
    assert f["tool"] == "t" and f["status"] == "ok" and f["request_id"] == "req-42" and "duration_ms" in f
    assert secret not in str(f) and "***REDACTED***" in str(f)
