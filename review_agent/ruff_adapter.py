from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
import sys
from typing import Any

from .models import StaticFinding, ToolStatus
from .config import MAX_EVIDENCE_CHARS
from .source import source_lines


# Project severity defaults describe impact, independently of confirmation status.
LOW_IMPACT_RULES = {"F401", "F541", "F841"}
HIGH_IMPACT_RULES = {"F821", "F822", "F823"}


def _syntax_rule(rule_id: str) -> bool:
    return rule_id.startswith("E9") or rule_id == "invalid-syntax"


def ruff_severity(rule_id: str) -> str:
    if _syntax_rule(rule_id) or rule_id in HIGH_IMPACT_RULES:
        return "high"
    if rule_id in LOW_IMPACT_RULES:
        return "low"
    return "medium"


def parse_ruff_json(text: str, code: str) -> list[StaticFinding]:
    payload = json.loads(text or "[]")
    if not isinstance(payload, list):
        raise ValueError("Ruff JSON 顶层必须是列表")
    lines = source_lines(code)
    findings: list[StaticFinding] = []
    for item in payload:
        rule_id = str(item.get("code") or "RUFF")
        loc = item.get("location") or {}
        end = item.get("end_location") or loc
        start_line = max(1, int(loc.get("row", 1)))
        end_line = max(start_line, int(end.get("row", start_line)))
        evidence = "\n".join(lines[start_line - 1:end_line]) or "<EOF>"
        truncated = len(evidence) > MAX_EVIDENCE_CHARS
        if truncated:
            # Start at the diagnostic column so a long prefix cannot hide the issue.
            column = max(0, int(loc.get("column", 1)) - 1)
            evidence = evidence[column:column + MAX_EVIDENCE_CHARS] or evidence[:MAX_EVIDENCE_CHARS]
            end_line = start_line + evidence.count("\n")
        message = str(item.get("message") or "Ruff 诊断")
        findings.append(StaticFinding(
            title=f"{rule_id}: {message}"[:160], severity=ruff_severity(rule_id),
            category="maintainability" if rule_id in LOW_IMPACT_RULES else "bug", line_start=start_line, line_end=end_line, evidence=evidence,
            column_start=max(1, int(loc.get("column", 1))), column_end=max(1, int(end.get("column", 1))),
            trigger="Ruff 对该源码位置产生了确定性诊断。", explanation=message[:2400],
            suggestion=str((item.get("fix") or {}).get("message") or "根据 Ruff 诊断修正该位置。")[:1600],
            rule_id=rule_id, origin="ruff", scope="Ruff F/E9 静态检查" + ("；证据较长，仅展示连续原文片段，完整源码仍参与审查" if truncated else ""),
            symbol=(_extract_symbol(item) or "")[:200] or None, needs_context=False,
        ))
    return findings


def _extract_symbol(item: dict[str, Any]) -> str | None:
    message = str(item.get("message") or "")
    if "`" in message:
        parts = message.split("`")
        return parts[1] if len(parts) >= 3 else None
    if "'" in message:
        parts = message.split("'")
        return parts[1] if len(parts) >= 3 else None
    return None


def run_ruff(code: str, filename: str = "input.py", timeout: float = 11.0) -> tuple[list[StaticFinding], ToolStatus]:
    executable = shutil.which("ruff")
    if executable:
        command = [executable]
    elif importlib.util.find_spec("ruff") is not None:
        command = [sys.executable, "-m", "ruff"]
    else:
        return [], ToolStatus(name="ruff", status="unavailable", detail="Ruff 可执行文件和 Python 模块均不可用")
    command.extend(["check", "--isolated", "--no-cache", "--select", "F,E9", "--output-format", "json", "--stdin-filename", filename, "-"])
    try:
        result = subprocess.run(command, input=code, text=True, encoding="utf-8", capture_output=True, shell=False, timeout=timeout, check=False)
    except FileNotFoundError:
        return [], ToolStatus(name="ruff", status="unavailable", detail="Ruff 可执行文件不可用")
    except subprocess.TimeoutExpired:
        return [], ToolStatus(name="ruff", status="timeout", detail=f"Ruff 超过 {timeout:g} 秒限制")
    except OSError as exc:
        return [], ToolStatus(name="ruff", status="error", detail=f"Ruff 启动失败: {type(exc).__name__}")
    try:
        findings = parse_ruff_json(result.stdout, code)
    except (ValueError, TypeError, json.JSONDecodeError):
        return [], ToolStatus(name="ruff", status="error", detail="Ruff 返回了无法解析的 JSON")
    if result.returncode not in (0, 1):
        return [], ToolStatus(name="ruff", status="error", detail=f"Ruff 异常退出（代码 {result.returncode}）")
    syntax = any(_syntax_rule(f.rule_id or "") for f in findings)
    status = "syntax_error" if syntax else "completed"
    return findings, ToolStatus(name="ruff", status=status, detail="已固定执行 F/E9 规则", finding_count=len(findings))


def diagnostic_key(finding: StaticFinding) -> tuple[str, str | None, int, str]:
    return (finding.rule_id or "", finding.symbol, finding.line_start, finding.explanation)
