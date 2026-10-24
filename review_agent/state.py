from __future__ import annotations

from typing import Any, Optional, TypedDict


class ReviewState(TypedDict, total=False):
    code: str
    filename: str
    focus: str
    source_hash: str
    static_findings: list[Any]
    tool_statuses: list[Any]
    messages: list[Any]
    draft_findings: list[Any]
    reviewed_findings: list[Any]
    pending_findings: list[Any]
    validation_records: list[Any]
    limitations: list[str]
    events: list[Any]
    plan_rounds: int
    tool_calls: int
    model_calls: int
    http_attempts: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    stage_ms: dict[str, int]
    started_at: float
    force_final: bool
    initial_stop_reason: str
    last_model_result: Any
    report: Any
