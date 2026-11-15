"""Pydantic schemas: tool inputs/outputs and the chat API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ToolStatus = Literal["ok", "error", "timeout", "invalid", "denied", "confirmation_required"]


class NoArgs(BaseModel):
    """Tool without arguments."""


class ToolResult(BaseModel):
    tool: str
    status: ToolStatus
    data: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0
    request_id: str = ""


class ToolCallRecord(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: ToolStatus
    duration_ms: float = 0.0


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    confirm: bool = Field(
        default=False,
        description="Explicit user confirmation for state-changing operations (restart, scale, trigger DAG).",
    )
    degradation_report: str | None = Field(
        default=None, description="Optional supplied evidence of model degradation (used for retraining advice)."
    )


class ChatResponse(BaseModel):
    request_id: str
    mode: Literal["llm", "rules"]
    answer: str
    facts: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
