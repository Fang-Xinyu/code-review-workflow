from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable

from .models import StaticFinding
from .config import MAX_EVIDENCE_CHARS
from .source import source_lines


@dataclass(frozen=True)
class Rule:
    rule_id: str
    title: str
    severity: str
    category: str
    explanation: str
    suggestion: str
    scope: str
    needs_context: bool = False


RULES = {
    "AST001": Rule("AST001", "Python 语法错误", "high", "bug", "源码无法被 Python 语法树解析。", "修正语法后再进行完整审查。", "语法解析"),
    "AST101": Rule("AST101", "可变默认参数", "medium", "bug", "默认对象在函数定义时创建，调用之间会共享状态。", "使用 None 作默认值，并在函数内创建新对象。", "函数定义"),
    "AST102": Rule("AST102", "裸 except", "medium", "exception", "裸 except 会捕获 SystemExit、KeyboardInterrupt 等不应被普通业务处理吞掉的异常。", "捕获能够处理的具体异常类型。", "异常处理"),
    "AST103": Rule("AST103", "异常被静默忽略", "medium", "exception", "捕获后仅执行 pass，失败原因和程序状态都不可见。", "仅捕获预期异常，并记录、转换或重新抛出。", "异常处理"),
    "AST104": Rule("AST104", "字面量除零", "high", "bug", "使用数值执行该表达式时，零分母会触发 ZeroDivisionError。", "修正分母，或在运算前验证其非零。", "算术表达式"),
    "AST105": Rule("AST105", "finally 中 return", "medium", "exception", "finally 中的 return 可覆盖 try/except 中的返回值或待抛出异常。", "将 return 移出 finally，finally 只负责清理。", "try/finally"),
    "AST106": Rule("AST106", "动态代码执行", "medium", "security", "eval/exec 会执行动态内容，当内容可被外部影响时会形成代码执行风险。", "改用明确的解析器或受限数据格式。", "调用点", True),
    "AST107": Rule("AST107", "文件句柄可能未关闭", "medium", "resource", "open() 的结果未交给 with，且当前作用域未看到对该变量的 close() 调用。", "使用 with open(...) as handle 管理文件生命周期。", "当前语法作用域", True),
    "AST108": Rule("AST108", "subprocess 使用 shell=True", "medium", "security", "shell=True 会将命令交给 shell 解析，当参数可被外部影响时存在注入风险。", "传递参数列表并保持 shell=False。", "subprocess 调用点", True),
    "AST109": Rule("AST109", "捕获 Exception 范围过宽", "low", "maintainability", "宽泛捕获可能隐藏编程错误，也使恢复策略不明确。", "缩小到实际能处理的异常类型。", "异常处理"),
}


def _segment(code: str, node: ast.AST) -> str:
    value = ast.get_source_segment(code, node)
    if value:
        return value
    lines = source_lines(code)
    return lines[max(0, getattr(node, "lineno", 1) - 1)] if lines else ""


def _finding(code: str, node: ast.AST, rule_id: str, *, symbol: str | None = None) -> StaticFinding:
    rule = RULES[rule_id]
    segment = _segment(code, node)
    evidence = segment[:MAX_EVIDENCE_CHARS]
    start = max(1, getattr(node, "lineno", 1))
    scope = rule.scope + ("；证据较长，仅展示连续原文前段，完整源码仍参与审查" if len(segment) > len(evidence) else "")
    return StaticFinding(
        title=rule.title, severity=rule.severity, category=rule.category,
        line_start=start,
        line_end=start + evidence.count("\n"),
        column_start=getattr(node, "col_offset", 0) + 1,
        column_end=getattr(node, "end_col_offset", 0) + 1,
        evidence=evidence, trigger=rule.explanation,
        explanation=rule.explanation, suggestion=rule.suggestion,
        rule_id=rule.rule_id, origin="ast", scope=scope,
        symbol=symbol[:200] if symbol else None, needs_context=rule.needs_context,
    )


def _is_zero(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)) and node.value == 0


def _is_numeric_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        node = node.operand
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def _walk_without_nested_scopes(node: ast.AST) -> Iterable[ast.AST]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        for child in ast.iter_child_nodes(current):
            if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            stack.append(child)


class Visitor(ast.NodeVisitor):
    def __init__(self, code: str):
        self.code = code
        self.findings: list[StaticFinding] = []
        self.finally_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_defaults(node)
        self._resource_checks(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _function_defaults(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            constructor = (isinstance(default, ast.Call) and isinstance(default.func, ast.Name)
                           and default.func.id in {"list", "dict", "set"} and not default.args and not default.keywords)
            if isinstance(default, (ast.List, ast.Dict, ast.Set)) or constructor:
                self.findings.append(_finding(self.code, default, "AST101", symbol=node.name))

    def _resource_checks(self, scope: ast.AST) -> None:
        closed: set[str] = set()
        for item in _walk_without_nested_scopes(scope):
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == "close" and isinstance(item.func.value, ast.Name):
                closed.add(item.func.value.id)
        for item in _walk_without_nested_scopes(scope):
            if isinstance(item, (ast.Assign, ast.AnnAssign)):
                value = item.value
                target = item.targets[0] if isinstance(item, ast.Assign) and len(item.targets) == 1 else getattr(item, "target", None)
                if _is_open_call(value) and isinstance(target, ast.Name) and target.id not in closed:
                    self.findings.append(_finding(self.code, value, "AST107", symbol=target.id))
            elif isinstance(item, ast.Expr) and _is_open_call(item.value):
                self.findings.append(_finding(self.code, item.value, "AST107"))

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.findings.append(_finding(self.code, node, "AST102"))
        elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
            self.findings.append(_finding(self.code, node.type, "AST109", symbol=node.name))
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self.findings.append(_finding(self.code, node, "AST103"))
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        numeric_modulo = isinstance(node.op, ast.Mod) and _is_numeric_literal(node.left)
        if (isinstance(node.op, (ast.Div, ast.FloorDiv)) or numeric_modulo) and _is_zero(node.right):
            self.findings.append(_finding(self.code, node, "AST104"))
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        for child in node.finalbody:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            for item in _walk_without_nested_scopes(child):
                if isinstance(item, ast.Return):
                    self.findings.append(_finding(self.code, item, "AST105"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
            self.findings.append(_finding(self.code, node, "AST106", symbol=node.func.id))
        if _is_subprocess_call(node) and any(k.arg == "shell" and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords):
            self.findings.append(_finding(self.code, node, "AST108"))
        self.generic_visit(node)


def _is_open_call(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open"


def _is_subprocess_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"


def analyze_ast(code: str) -> tuple[list[StaticFinding], str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        lines = source_lines(code)
        line = min(max(1, exc.lineno or 1), len(lines))
        while line > 1 and not lines[line - 1].strip():
            line -= 1
        evidence = lines[line - 1] or "<EOF>"
        rule = RULES["AST001"]
        finding = StaticFinding(title=rule.title, severity="high", category="bug", line_start=line,
            line_end=line, evidence=evidence[:MAX_EVIDENCE_CHARS],
            trigger=f"第 {exc.lineno or line} 行：{exc.msg}"[:1200], explanation=rule.explanation,
            suggestion=rule.suggestion, rule_id="AST001", origin="ast", scope=rule.scope)
        return [finding], "syntax_error"
    visitor = Visitor(code)
    visitor._resource_checks(tree)
    visitor.visit(tree)
    unique = {(x.rule_id, x.line_start, x.column_start, x.line_end, x.column_end, x.evidence): x for x in visitor.findings}
    return list(unique.values()), "completed"
