"""Structured observations emitted while executing a customer-service plan."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """One normalized event from the user, executor, or retrieval controller."""

    event_id: str = Field(min_length=1)
    event_type: Literal[
        "user_message",
        "assistant_message",
        "tool_call",
        "tool_result",
        "retrieval_result",
        "error",
    ]
    tool_name: Optional[str] = None
    wrapper_name: Optional[str] = None
    request_id: Optional[str] = None
    step_id: Optional[str] = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Optional[str] = None
    success: Optional[bool] = None
