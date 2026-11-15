"""Agent API (FastAPI): validation in, evidence-based response out.   Run: uvicorn agent.main:app --port 8000"""
from __future__ import annotations

import re

from fastapi import FastAPI, Header

from agent.config import get_settings
from agent.logging_utils import configure_logging
from agent.orchestrator import Agent
from agent.schemas import ChatRequest, ChatResponse

configure_logging()
app = FastAPI(title="MLOps AI Agent", version="1.0.0")
_agent: Agent | None = None


def build_agent() -> Agent:
    import agent.tools  # noqa: F401  (register tools)

    llm = None
    if get_settings().llm_enabled:
        from agent.llm import AnthropicLLM

        llm = AnthropicLLM()
    return Agent(llm=llm)


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "llm" if get_settings().llm_enabled else "rules"}


@app.get("/tools")
def tools() -> list[dict]:
    """Tool catalogue with permissions (read / write)."""
    return get_agent().registry.describe()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_request_id: str | None = Header(default=None)) -> ChatResponse:
    rid = x_request_id if x_request_id and re.fullmatch(r"[\w\-]{1,64}", x_request_id) else None
    return get_agent().handle(req.message, confirm=req.confirm, request_id=rid, degradation_report=req.degradation_report)
