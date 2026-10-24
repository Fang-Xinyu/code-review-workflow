from __future__ import annotations

import json

from pydantic import ValidationError

from .llm import ModelLike, RecordingModel, parse_json_object
from .models import ReflectionResponse, ReviewedFinding, ValidationRecord
from .prompts import REFLECTION_SYSTEM


def reflect(model: ModelLike, candidates: list[ReviewedFinding], code: str | None = None) -> tuple[list[ReviewedFinding], list[ValidationRecord], int, int | None, int | None]:
    targets = [x for x in candidates if x.severity in {"high", "medium"} or x.status == "needs_confirmation"]
    if not targets:
        return candidates, [], 0, None, None
    payload = [{**x.model_dump(), "location": [x.line_start, x.line_end]} for x in targets]
    error = ""
    calls = 0
    input_tokens = output_tokens = None
    for _ in range(2):
        if isinstance(model, RecordingModel):
            model.stage = "reflection" if calls == 0 else "reflection_repair"
        calls += 1
        last_result = model.invoke(REFLECTION_SYSTEM, json.dumps({"source": code, "candidates": payload, "previous_error": error, "reflection_schema": ReflectionResponse.model_json_schema()}, ensure_ascii=False))
        input_tokens = _add_tokens(input_tokens, last_result.input_tokens)
        output_tokens = _add_tokens(output_tokens, last_result.output_tokens)
        try:
            parsed = ReflectionResponse.model_validate(parse_json_object(last_result.content))
            expected = [x.finding_id for x in targets]
            actual = [x.finding_id for x in parsed.decisions]
            if len(actual) != len(set(actual)) or set(actual) != set(expected):
                raise ValueError("裁决 ID 必须完整、唯一且不得新增")
            break
        except (ValidationError, ValueError) as exc:
            error = str(exc)[:700]
    else:
        raise ValueError(f"复核格式两次失败: {error}")
    decisions = {x.finding_id: x for x in parsed.decisions}
    output: list[ReviewedFinding] = []
    records: list[ValidationRecord] = []
    for item in candidates:
        decision = decisions.get(item.finding_id)
        if not decision:
            output.append(item)
            continue
        verdict_text = {"supported": "支持", "uncertain": "待确认", "rejected": "不支持"}[decision.verdict]
        records.append(ValidationRecord(action="reflection", finding_ref=item.finding_id, detail=f"{verdict_text}：{decision.note}"))
        if decision.verdict == "rejected" and item.origin == "model":
            continue
        status = item.status
        if item.origin == "model":
            status = "model_reviewed" if decision.verdict == "supported" else "needs_confirmation"
        elif decision.verdict == "rejected":
            # Keep the deterministic rule signal, but do not present an AI-disputed
            # issue as confirmed. The user can resolve it with project context.
            status = "needs_confirmation"
        elif item.status == "needs_confirmation" and decision.verdict == "supported":
            status = "model_reviewed"
        changes = decision.changes.model_dump(exclude_unset=True)
        severity = changes.get("severity", item.severity)
        merged = {**item.model_dump(), **changes, "severity": severity, "status": status,
                  "review_note": decision.note}
        output.append(ReviewedFinding.model_validate(merged))
    return output, records, calls, input_tokens, output_tokens


def _add_tokens(old: int | None, new: int | None) -> int | None:
    return None if old is None and new is None else (old or 0) + (new or 0)
