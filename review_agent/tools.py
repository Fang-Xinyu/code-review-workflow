from __future__ import annotations

import ast
import json
from collections import Counter
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import MAX_TOOL_OUTPUT_CHARS
from .models import StaticFinding
from .source import source_lines


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadRangeArgs(ToolArgs):
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class SymbolArgs(ToolArgs):
    symbol_name: str = Field(min_length=1, max_length=200)
    context_start_line: Optional[int] = Field(default=None, ge=1)


class StaticQueryArgs(ToolArgs):
    severity: Optional[Literal["high", "medium", "low"]] = None
    rule_id: Optional[str] = None
    offset: int = Field(default=0, ge=0)


TOOL_SCHEMAS = [
    {"name": "read_source_range", "description": "读取当前内存源码的有界行范围", "parameters": ReadRangeArgs.model_json_schema()},
    {"name": "get_symbol_context", "description": "查找唯一函数或类的定义、名称匹配的调用位置；用 context_start_line 分页读取定义上下文", "parameters": SymbolArgs.model_json_schema()},
    {"name": "get_static_findings", "description": "查询本次已执行的 AST/Ruff 结果；用 offset 分页", "parameters": StaticQueryArgs.model_json_schema()},
]


class EvidenceTools:
    def __init__(self, code: str, findings: list[StaticFinding], max_chars: int = MAX_TOOL_OUTPUT_CHARS):
        if max_chars < 512:
            raise ValueError("工具输出限制至少为 512 字符")
        self.code = code
        self.lines = source_lines(code)
        self.findings = findings
        self.max_chars = max_chars
        self.seen: Counter[str] = Counter()

    def execute(self, name: str, arguments: str | dict[str, Any], call_id: str) -> dict[str, Any]:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(args, dict):
                raise ValueError("参数必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError) as exc:
            return self._error(call_id, "invalid_json", str(exc))
        signature = json.dumps([name, args], ensure_ascii=False, sort_keys=True)
        self.seen[signature] += 1
        if self.seen[signature] > 1:
            return self._error(call_id, "duplicate_call", "相同工具和参数已调用，不重复返回内容")
        mapping: dict[str, tuple[type[ToolArgs], Callable[[ToolArgs], dict[str, Any]]]] = {
            "read_source_range": (ReadRangeArgs, self._range),
            "get_symbol_context": (SymbolArgs, self._symbol),
            "get_static_findings": (StaticQueryArgs, self._static),
        }
        if name not in mapping:
            return self._error(call_id, "unknown_tool", f"未知工具: {name}")
        schema, function = mapping[name]
        try:
            parsed = schema.model_validate(args)
            result = function(parsed)
        except ValidationError as exc:
            return self._error(call_id, "invalid_arguments", exc.errors(include_url=False)[0]["msg"])
        except ValueError as exc:
            return self._error(call_id, "invalid_range", str(exc))
        result.update({"ok": True, "tool_call_id": call_id, "tool": name})
        return self._bounded(result)

    def _bounded(self, result: dict[str, Any]) -> dict[str, Any]:
        """Bound the complete JSON, keeping data structured and truncation explicit."""
        def size():
            return len(json.dumps(result, ensure_ascii=False))

        if size() <= self.max_chars:
            return result
        result["truncated"] = True
        # Trim complete source lines first, retaining their actual returned range.
        field = "content" if "content" in result else "context" if "context" in result else None
        if field:
            lines = result[field].split("\n")
            while len(lines) > 1 and size() > self.max_chars:
                lines.pop()
                result[field] = "\n".join(lines)
                start = result.get("returned_start_line", result.get("start_line", 1))
                result["returned_end_line"] = start + len(lines) - 1
                result["next_start_line"] = result["returned_end_line"] + 1
        if "findings" in result:
            while result["findings"] and size() > self.max_chars:
                result["findings"].pop()
                result["next_offset"] = result["offset"] + len(result["findings"])
            if not result["findings"] and result["count"] > result["offset"]:
                result["partial_item"] = True
        # Oversized individual lines/diagnostics and error messages remain explicitly partial.
        if size() > self.max_chars:
            result["partial_item"] = True
            if field:
                result["next_start_line"] = result.get("returned_start_line", result.get("start_line", 1))
            strings = [(result, key) for key, value in result.items() if isinstance(value, str)]
            if isinstance(result.get("error"), dict):
                strings.extend((result["error"], key) for key, value in result["error"].items() if isinstance(value, str))
            while size() > self.max_chars and any(container[key] for container, key in strings):
                container, key = max(strings, key=lambda pair: len(pair[0][pair[1]]))
                container[key] = container[key][:len(container[key]) // 2]
        if size() > self.max_chars:
            return {"ok": False, "truncated": True, "error": {"code": "output_limit", "message": "结果元数据超过输出上限，请缩小查询范围"}}
        return result

    def _error(self, call_id: str, code: str, message: str) -> dict[str, Any]:
        return self._bounded({"ok": False, "tool_call_id": call_id, "error": {"code": code, "message": message}})

    def _range(self, args: ReadRangeArgs) -> dict[str, Any]:
        if args.end_line < args.start_line or args.end_line > len(self.lines) or args.end_line - args.start_line + 1 > 120:
            raise ValueError("行号必须在源码内，顺序正确且一次不超过 120 行")
        content = "\n".join(f"{i:4d} | {self.lines[i - 1]}" for i in range(args.start_line, args.end_line + 1))
        return {"start_line": args.start_line, "end_line": args.end_line,
                "returned_start_line": args.start_line, "returned_end_line": args.end_line, "content": content}

    def _symbol(self, args: SymbolArgs) -> dict[str, Any]:
        try:
            tree = ast.parse(self.code)
        except SyntaxError:
            return {"found": False, "error": "syntax_error"}
        definitions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == args.symbol_name]
        if len(definitions) != 1:
            return {"found": False, "error": "not_found" if not definitions else "ambiguous", "matches": len(definitions)}
        node = definitions[0]
        all_calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == args.symbol_name) or (isinstance(n.func, ast.Attribute) and n.func.attr == args.symbol_name))]
        calls = all_calls[:20]
        start, end = node.lineno, getattr(node, "end_lineno", node.lineno)
        lo = args.context_start_line or max(1, start - 3)
        context_end = min(len(self.lines), end + 3)
        if lo > context_end or lo < max(1, start - 3):
            raise ValueError("上下文起点必须在符号定义及附近范围内")
        hi = min(context_end, lo + 119)
        return {"found": True, "kind": type(node).__name__, "start_line": start, "end_line": end,
                "signature": self.lines[start - 1][:500], "call_lines": calls,
                "call_count": len(all_calls), "call_lines_truncated": len(all_calls) > len(calls),
                "signature_truncated": len(self.lines[start - 1]) > 500,
                "returned_start_line": lo, "returned_end_line": hi,
                "next_start_line": hi + 1 if hi < context_end else None,
                "truncated": hi < context_end or len(all_calls) > len(calls) or len(self.lines[start - 1]) > 500,
                "context": "\n".join(f"{i:4d} | {self.lines[i - 1]}" for i in range(lo, hi + 1))}

    def _static(self, args: StaticQueryArgs) -> dict[str, Any]:
        selected = [f for f in self.findings if (not args.severity or f.severity == args.severity) and (not args.rule_id or f.rule_id == args.rule_id)]
        page = selected[args.offset:args.offset + 30]
        next_offset = args.offset + len(page)
        return {"count": len(selected), "offset": args.offset, "findings": [f.model_dump() for f in page],
                "next_offset": next_offset if next_offset < len(selected) else None,
                "truncated": next_offset < len(selected)}
