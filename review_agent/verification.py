from __future__ import annotations

import ast
import hashlib
import io
import json
import tokenize

from .config import MAX_SOURCE_BYTES, MAX_SOURCE_LINES
from .models import FindingDraft, ReviewedFinding, StaticFinding, ValidationRecord
from .source import source_lines


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def normalize_and_validate(code: str, filename: str, focus: str = "") -> tuple[str, str]:
    if not filename.lower().endswith(".py"):
        raise ValueError("只支持 .py 文件")
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        raise ValueError("源码不能为空")
    if "\x00" in normalized:
        raise ValueError("源码不能包含空字节")
    if len(source_lines(normalized)) > MAX_SOURCE_LINES:
        raise ValueError(f"源码超过 {MAX_SOURCE_LINES} 行")
    if len(code.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("源码 UTF-8 大小超过 100 KB（102400 字节）")
    return normalized, hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def finding_id(origin: str, rule_id: str | None, start: int, evidence: str,
               *, column_start: int = 1, column_end: int = 1,
               symbol: str | None = None, explanation: str = "") -> str:
    raw = json.dumps([origin, rule_id, start, column_start, column_end, evidence, symbol, explanation],
                     ensure_ascii=False).encode("utf-8")
    return "F-" + hashlib.sha256(raw).hexdigest()[:12]


def _positions(code: str, evidence: str) -> list[tuple[int, int, int, int]]:
    results: list[tuple[int, int, int, int]] = []
    pos = 0
    while True:
        idx = code.find(evidence, pos)
        if idx < 0:
            break
        start = code.count("\n", 0, idx) + 1
        end = start + evidence.count("\n")
        results.append((start, end, idx, idx + len(evidence)))
        pos = idx + 1
    return results


def _comment_or_docstring_only(code: str, start: int, end: int) -> bool:
    excerpt = code[start:end]
    start += len(excerpt) - len(excerpt.lstrip())
    end -= len(excerpt) - len(excerpt.rstrip())
    lines = code.split("\n")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)

    def position(row: int, column: int, *, utf8: bool = False) -> int:
        if utf8:
            column = len(lines[row - 1].encode("utf-8")[:column].decode("utf-8"))
        return offsets[row - 1] + column

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        comment_ranges = [(position(*t.start), position(*t.end)) for t in tokens if t.type == tokenize.COMMENT]
        if any(a <= start and end <= b for a, b in comment_ranges):
            return True
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                    doc = body[0]
                    a = position(doc.lineno, doc.col_offset, utf8=True)
                    b = position(doc.end_lineno, doc.end_col_offset, utf8=True)
                    if a <= start and end <= b:
                        return True
    except (SyntaxError, tokenize.TokenError):
        return False
    return False


def ground_model_findings(code: str, drafts: list[FindingDraft], static: list[StaticFinding]) -> tuple[list[ReviewedFinding], list[ValidationRecord]]:
    accepted: list[ReviewedFinding] = []
    records: list[ValidationRecord] = []
    static_keys = {(x.rule_id, x.line_start): x for x in static}
    line_count = max(1, len(source_lines(code)))
    for index, draft in enumerate(drafts):
        ref = f"model[{index}]"
        positions = _positions(code, draft.evidence)
        target = next((p for p in positions if draft.line_start <= p[0] and p[1] <= draft.line_end), None)
        if target is None and len(positions) == 1:
            target = positions[0]
            records.append(ValidationRecord(action="relocated", finding_ref=ref, detail=f"证据唯一匹配，已重定位到 {target[0]}-{target[1]} 行"))
        if target is None:
            detail = "证据不存在" if not positions else "证据出现多次且无法唯一定位"
            records.append(ValidationRecord(action="rejected", finding_ref=ref, detail=detail))
            continue
        if target[1] > line_count or _comment_or_docstring_only(code, target[2], target[3]):
            records.append(ValidationRecord(action="rejected", finding_ref=ref, detail="证据仅位于注释或独立文档字符串中"))
            continue
        rule_id = draft.rule_id
        origin = "model"
        if rule_id:
            matching = [s for (rid, line), s in static_keys.items() if rid == rule_id and s.line_start <= target[1] and s.line_end >= target[0]]
            if matching:
                origin = matching[0].origin
            else:
                records.append(ValidationRecord(action="rule_unlinked", finding_ref=ref, detail=f"{rule_id} 未与本次真实诊断的规则和位置匹配，已移除背书"))
                rule_id = None
        status = "static_confirmed" if origin != "model" else "needs_confirmation"
        severity = draft.severity
        accepted.append(ReviewedFinding(**draft.model_dump(exclude={"rule_id", "line_start", "line_end", "severity"}),
            line_start=target[0], line_end=target[1], severity=severity, rule_id=rule_id,
            finding_id=finding_id(origin, rule_id, target[0], draft.evidence), origin=origin, status=status,
            review_note="", validation_hint="模型候选已通过连续原文定位"))
    return accepted, records


def static_to_reviewed(item: StaticFinding) -> ReviewedFinding:
    status = "needs_confirmation" if item.needs_context else "static_confirmed"
    severity = item.severity
    return ReviewedFinding(**item.model_dump(exclude={"origin", "scope", "symbol", "needs_context", "severity", "column_start", "column_end"}),
        severity=severity, finding_id=finding_id(item.origin, item.rule_id, item.line_start, item.evidence,
            column_start=item.column_start, column_end=item.column_end,
            symbol=item.symbol, explanation=item.explanation),
        origin=item.origin, status=status, review_note="固定静态规则发现",
        validation_hint=item.scope)


def reconcile(code: str, drafts: list[FindingDraft], static: list[StaticFinding]) -> tuple[list[ReviewedFinding], list[ValidationRecord]]:
    model_items, records = ground_model_findings(code, drafts, static)
    merged: dict[tuple[object, ...], ReviewedFinding] = {}
    for static_item in static:
        item = static_to_reviewed(static_item)
        key = (item.finding_id,)
        merged[key] = item
        represented = any(candidate.origin == item.origin and candidate.rule_id == item.rule_id
                          and candidate.line_start <= item.line_end and candidate.line_end >= item.line_start
                          for candidate in model_items)
        if represented:
            records.append(ValidationRecord(action="deduplicated", finding_ref=item.finding_id,
                detail="模型候选与真实静态诊断重合，保留工具原始诊断"))
        else:
            records.append(ValidationRecord(action="backfilled", finding_ref=item.finding_id, detail="模型未写入的真实静态诊断已由程序补回"))
    for item in model_items:
        if item.origin == "model":
            merged[("model", item.rule_id, item.line_start, item.evidence)] = item
    values = sorted(merged.values(), key=lambda x: (SEVERITY_ORDER[x.severity], 0 if x.origin != "model" else 1, x.line_start, x.rule_id or ""))
    if len(values) > 30:
        omitted = len(values) - 30
        values = values[:30]
        records.append(ValidationRecord(action="truncated", detail=f"候选超过 30 条，按严重度、静态优先和行号排序后省略 {omitted} 条"))
    return values, records
