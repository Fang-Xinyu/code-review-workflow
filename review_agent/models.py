from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import MAX_EVIDENCE_CHARS, MAX_MODEL_CALLS, MAX_PLAN_ROUNDS, MAX_SOURCE_BYTES, MAX_TOOL_CALLS_PER_ROUND

Severity = Literal["high", "medium", "low"]
Category = Literal["bug", "boundary", "exception", "resource", "security", "maintainability"]
Origin = Literal["ast", "ruff", "model"]
FindingStatus = Literal["static_confirmed", "model_reviewed", "needs_confirmation"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewInput(StrictModel):
    code: str = Field(min_length=1, max_length=MAX_SOURCE_BYTES)
    filename: str = Field(min_length=1, max_length=255)
    focus: str = Field(default="全面检查正确性、异常处理、资源和安全", max_length=1000)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("filename")
    @classmethod
    def python_file(cls, value: str) -> str:
        if not value.lower().endswith(".py"):
            raise ValueError("只支持 .py 文件")
        return value


class FindingDraft(StrictModel):
    title: str = Field(min_length=1, max_length=160)
    severity: Severity
    category: Category
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    evidence: str = Field(min_length=1, max_length=MAX_EVIDENCE_CHARS)
    trigger: str = Field(min_length=1, max_length=1200)
    explanation: str = Field(min_length=1, max_length=2400)
    suggestion: str = Field(min_length=1, max_length=1600)
    rule_id: Optional[str] = Field(default=None, max_length=40)

    @model_validator(mode="after")
    def ordered(self) -> "FindingDraft":
        if self.line_end < self.line_start:
            raise ValueError("line_end 不能小于 line_start")
        return self


class ReviewedFinding(FindingDraft):
    finding_id: str = Field(pattern=r"^F-[0-9a-f]{12}$")
    origin: Origin
    status: FindingStatus
    review_note: str = Field(default="", max_length=1600)
    validation_hint: str = Field(default="", max_length=1600)


class StaticFinding(FindingDraft):
    origin: Literal["ast", "ruff"]
    column_start: int = Field(default=1, ge=1)
    column_end: int = Field(default=1, ge=1)
    scope: str = Field(default="", max_length=1200)
    symbol: Optional[str] = Field(default=None, max_length=200)
    needs_context: bool = False


class ToolStatus(StrictModel):
    name: str
    status: Literal["completed", "unavailable", "timeout", "error", "syntax_error"]
    detail: str = ""
    finding_count: int = Field(default=0, ge=0)


class ValidationRecord(StrictModel):
    action: Literal["accepted", "rejected", "hidden", "relocated", "backfilled", "deduplicated", "truncated", "rule_unlinked", "reflection"]
    finding_ref: str = ""
    detail: str


class ExecutionEvent(StrictModel):
    stage: str
    message: str
    elapsed_ms: Optional[int] = Field(default=None, ge=0)


class ModelAttemptStats(StrictModel):
    elapsed_ms: int = Field(ge=0)
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    answer_chars: int = Field(default=0, ge=0)
    finish_reason: Optional[str] = None
    reasoning_tokens: Optional[int] = Field(default=None, ge=0)
    failed: bool = False


class ModelCallStats(ModelAttemptStats):
    stage: str
    http_attempts: int = Field(ge=0)
    attempts: list[ModelAttemptStats] = Field(default_factory=list)


class UsageStats(StrictModel):
    calls: list[ModelCallStats] = Field(default_factory=list)
    total_ms: int = Field(ge=0)
    stage_ms: dict[str, int] = Field(default_factory=dict)
    model_calls: int = Field(ge=0)
    http_attempts: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    model_call_limit: int = Field(default=MAX_MODEL_CALLS, ge=1)
    plan_rounds: int = Field(default=0, ge=0)
    plan_round_limit: int = Field(default=MAX_PLAN_ROUNDS, ge=1)
    tool_call_limit: int = Field(default=(MAX_PLAN_ROUNDS - 1) * MAX_TOOL_CALLS_PER_ROUND, ge=0)
    initial_stop_reason: Literal["candidate_submitted", "round_limit", "budget_limit", "tool_request_rejected"] = "candidate_submitted"


class ReviewSummary(StrictModel):
    total_formal: int = Field(ge=0)
    total_pending: int = Field(ge=0)
    high: int = Field(ge=0)
    medium: int = Field(ge=0)
    low: int = Field(ge=0)
    text: str


class ReviewReport(StrictModel):
    schema_version: Literal["1.0", "1.1"] = "1.1"
    filename: str
    source_hash: str
    focus: str
    model: str
    summary: ReviewSummary
    findings: list[ReviewedFinding] = Field(default_factory=list, max_length=30)
    pending_findings: list[ReviewedFinding] = Field(default_factory=list, max_length=30)
    validation_records: list[ValidationRecord] = Field(default_factory=list)
    tool_statuses: list[ToolStatus] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    events: list[ExecutionEvent] = Field(default_factory=list)
    usage: UsageStats


class DraftResponse(StrictModel):
    findings: list[FindingDraft] = Field(max_length=60)


class ReflectionChanges(StrictModel):
    # Default factories allow omission without accepting explicit null in JSON/schema.
    title: str = Field(default_factory=lambda: None, min_length=1, max_length=160)
    severity: Severity = Field(default_factory=lambda: None)
    trigger: str = Field(default_factory=lambda: None, min_length=1, max_length=1200)
    explanation: str = Field(default_factory=lambda: None, min_length=1, max_length=2400)
    suggestion: str = Field(default_factory=lambda: None, min_length=1, max_length=1600)

    @field_validator("title", "trigger", "explanation", "suggestion")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("修改内容不能为空白")
        return value


class ReflectionDecision(StrictModel):
    finding_id: str = Field(pattern=r"^F-[0-9a-f]{12}$")
    verdict: Literal["supported", "uncertain", "rejected"]
    note: str = Field(min_length=1, max_length=1600)
    changes: ReflectionChanges = Field(default_factory=ReflectionChanges)


class ReflectionResponse(StrictModel):
    decisions: list[ReflectionDecision]


class PatchEdit(StrictModel):
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    original: str
    replacement: str
    rationale: str = Field(min_length=1, max_length=1600)

    @model_validator(mode="after")
    def ordered(self) -> "PatchEdit":
        if self.line_end < self.line_start:
            raise ValueError("补丁行号倒置")
        return self


class PatchResponse(StrictModel):
    patches: list[PatchEdit] = Field(min_length=1, max_length=3)


class PatchReviewResponse(StrictModel):
    verdict: Literal["supported", "uncertain", "rejected"]
    note: str = Field(min_length=1, max_length=1600, strict=True)


class FollowupResponse(StrictModel):
    answer: str = Field(min_length=1, strict=True)

    @field_validator("answer")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("追问响应缺少有效 answer")
        return value


class PatchValidation(StrictModel):
    valid: bool
    patched_code: Optional[str] = None
    diff: Optional[str] = None
    errors: list[str] = Field(default_factory=list)
    resolved_diagnostics: list[str] = Field(default_factory=list)
    new_diagnostics: list[str] = Field(default_factory=list)
    model_note: Optional[str] = None


JsonDict = dict[str, Any]
