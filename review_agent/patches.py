from __future__ import annotations

import ast
import difflib
import json

from pydantic import ValidationError

from .ast_rules import analyze_ast
from .llm import ModelLike, RecordingModel, parse_json_object
from .models import PatchEdit, PatchResponse, PatchReviewResponse, PatchValidation, ReviewedFinding
from .prompts import PATCH_REVIEW_SYSTEM, PATCH_SYSTEM
from .ruff_adapter import diagnostic_key, run_ruff
from .source import source_lines


def apply_and_validate_patches(code: str, patches: list[PatchEdit], filename: str = "input.py",
                               target: ReviewedFinding | None = None, ruff_runner=run_ruff) -> PatchValidation:
    lines = source_lines(code)
    ordered = sorted(patches, key=lambda x: x.line_start)
    errors: list[str] = []
    previous_end = 0
    for patch in ordered:
        if patch.line_end > len(lines):
            errors.append(f"补丁 {patch.line_start}-{patch.line_end} 超出源码范围")
        if patch.line_start <= previous_end:
            errors.append("补丁范围重叠")
        previous_end = patch.line_end
        actual = "\n".join(lines[patch.line_start - 1:patch.line_end])
        if patch.line_end <= len(lines) and actual != patch.original:
            errors.append(f"补丁 {patch.line_start}-{patch.line_end} 的 original 与当前快照不匹配")
    if errors:
        return PatchValidation(valid=False, errors=errors)
    changed = list(lines)
    for patch in reversed(ordered):
        changed[patch.line_start - 1:patch.line_end] = source_lines(patch.replacement)
    patched = "\n".join(changed) + ("\n" if code.endswith("\n") else "")
    try:
        ast.parse(patched)
    except SyntaxError as exc:
        return PatchValidation(valid=False, errors=[f"组合补丁产生语法错误：{exc.msg}（第 {exc.lineno or '?'} 行）"])
    before, before_status = ruff_runner(code, filename)
    after, after_status = ruff_runner(patched, filename)
    if before_status.status not in {"completed", "syntax_error"} or after_status.status not in {"completed", "syntax_error"}:
        errors.append(f"Ruff 前后比较不完整：{before_status.status} -> {after_status.status}")
    before_keys, after_keys = {diagnostic_key(x) for x in before}, {diagnostic_key(x) for x in after}
    def matches(old, current):
        return (old[0], old[1], old[3]) == (current[0], current[1], current[3]) and current[2] in _mapped_lines(old[2], ordered)

    unmatched = set(after_keys)
    preserved = set()
    for old in sorted(before_keys, key=_key_text):
        candidates = [current for current in unmatched if matches(old, current)]
        if candidates:
            unmatched.remove(min(candidates, key=lambda current: current[2]))
            preserved.add(old)
    resolved = sorted(_key_text(old) for old in before_keys - preserved)
    new = sorted(_key_text(current) for current in unmatched)
    if new:
        errors.append(f"补丁引入 {len(new)} 条新 Ruff 诊断")
    if target and target.origin == "ruff" and target.rule_id:
        target_keys = [key for key in before_keys if key[0] == target.rule_id and key[2] == target.line_start]
        exact = [key for key in target_keys if key[3] == target.explanation]
        target_keys = exact or target_keys
        if not target_keys:
            errors.append("修复前未在 Ruff 结果中找到目标诊断")
        elif any(matches(old, current) for old in target_keys for current in after_keys):
            errors.append(f"目标 Ruff 诊断 {target.rule_id} 仍存在")
    if target and target.origin == "ast" and target.rule_id:
        after_ast, _ = analyze_ast(patched)
        if any(x.rule_id == target.rule_id and x.line_start in _mapped_lines(target.line_start, ordered) for x in after_ast):
            errors.append(f"目标 AST 诊断 {target.rule_id} 在修改后对应位置仍存在")
    diff = "\n".join(difflib.unified_diff(source_lines(code), source_lines(patched), fromfile=f"a/{filename}", tofile=f"b/{filename}", lineterm=""))
    return PatchValidation(valid=not errors, patched_code=patched if not errors else None, diff=diff if not errors else None,
                           errors=errors, resolved_diagnostics=resolved, new_diagnostics=new)


def _mapped_lines(line: int, patches: list[PatchEdit]) -> range:
    """Map an old line to its new line, or conservatively to its replacement block."""
    offset = 0
    for patch in patches:
        if line < patch.line_start:
            break
        replacement_count = len(source_lines(patch.replacement))
        if line <= patch.line_end:
            start = patch.line_start + offset
            return range(start, start + replacement_count)
        offset += replacement_count - (patch.line_end - patch.line_start + 1)
    return range(line + offset, line + offset + 1)


def _key_text(key: tuple[str, str | None, int, str]) -> str:
    return f"{key[0]}:{key[1] or '-'}:{key[2]}:{key[3]}"


def generate_patch(model: ModelLike, code: str, goal: str, filename: str = "input.py",
                   target: ReviewedFinding | None = None, ruff_runner=run_ruff) -> PatchValidation:
    if not isinstance(model, RecordingModel):
        model = RecordingModel(model)
    previous_error = ""
    for _ in range(2):
        model.stage = "patch_generation"
        payload = {"filename": filename, "goal": goal, "source": code,
                   "target": target.model_dump() if target else None, "previous_validation_error": previous_error,
                   "schema": PatchResponse.model_json_schema()}
        result = model.invoke(PATCH_SYSTEM, json.dumps(payload, ensure_ascii=False))
        try:
            response = PatchResponse.model_validate(parse_json_object(result.content))
            validation = apply_and_validate_patches(code, response.patches, filename, target, ruff_runner)
        except (ValidationError, ValueError) as exc:
            previous_error = str(exc)[:1000]
            continue
        if not validation.valid:
            previous_error = "; ".join(validation.errors)[:1000]
            continue
        # Review a validated patch once; review format failure must not regenerate code.
        model.stage = "patch_review"
        review = model.invoke(PATCH_REVIEW_SYSTEM, json.dumps({"goal": goal, "original": code,
            "patched": validation.patched_code, "diff": validation.diff}, ensure_ascii=False))
        try:
            decision = PatchReviewResponse.model_validate(parse_json_object(review.content))
        except (ValidationError, ValueError):
            return PatchValidation(valid=False, errors=["补丁独立复核响应格式不合法，补丁正文已隐藏"])
        if decision.verdict != "supported":
            return PatchValidation(valid=False, errors=[f"补丁独立复核未支持：{decision.verdict} {decision.note}"])
        validation.model_note = decision.note
        return validation
    return PatchValidation(valid=False, errors=[f"补丁经一次修正后仍未通过：{previous_error}"])
