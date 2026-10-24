import ast
import difflib
import io
import tokenize
import textwrap

from review_agent.ast_rules import analyze_ast
from review_agent.patches import apply_and_validate_patches
from review_agent.ruff_adapter import run_ruff
from review_agent.source import source_lines
from review_agent.verification import normalize_and_validate
from schemas import StaticResult, digest


def mapped_lines(line, patches):
    offset = 0
    for p in sorted(patches, key=lambda p: p.line_start):
        if line < p.line_start:
            break
        count = len(source_lines(p.replacement))
        if line <= p.line_end:
            return range(p.line_start + offset, p.line_start + offset + count)
        offset += count - (p.line_end - p.line_start + 1)
    return range(line + offset, line + offset + 1)


def evidence_tokens(evidence):
    try:
        ignored = {tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER}
        return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(textwrap.dedent(evidence)).readline)
                if t.type not in ignored]
    except (tokenize.TokenError, IndentationError):
        return None


def compare_diagnostics(before, after, patches):
    """One-to-one matching preserves multiplicity; changed lines need exact evidence/columns."""
    unmatched = list(after)
    matches = []
    for old in before:
        changed = any(p.line_start <= old.line_start <= p.line_end for p in patches)
        match = next((new for new in unmatched if
                      (old.origin, old.rule_id, old.symbol, old.explanation) ==
                      (new.origin, new.rule_id, new.symbol, new.explanation)
                      and new.line_start in mapped_lines(old.line_start, patches)
                      and (not changed or (old.evidence, old.column_start, old.column_end) ==
                           (new.evidence, new.column_start, new.column_end)
                           or (old.origin == "ast" and old.evidence != new.evidence
                               and evidence_tokens(old.evidence) is not None
                               and evidence_tokens(old.evidence) == evidence_tokens(new.evidence)))), None)
        if match is not None:
            unmatched.remove(match)
            matches.append((old, match))
    return unmatched, matches


def function_names(code):
    try:
        return {(type(n).__name__, n.name) for n in ast.walk(ast.parse(code))
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    except SyntaxError:
        return set()


def executable_ast(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def suppressions(code):
    try:
        return [t.string for t in tokenize.generate_tokens(io.StringIO(code).readline)
                if t.type == tokenize.COMMENT and any(s in t.string.lower() for s in ("noqa", "ruff:", "type: ignore"))]
    except (tokenize.TokenError, IndentationError):
        return []


def pending_evidence_preserved(before, after, evidence, patches):
    before_count, after_count = before.count(evidence), after.count(evidence)
    if before_count == after_count and before_count:
        return True
    if after_count > before_count:
        return False
    # A guard inserted between unchanged original lines breaks a contiguous
    # excerpt without deleting or rewriting those lines. Keep their identities
    # and order, and constrain each to the corresponding patch-mapped region.
    matcher = difflib.SequenceMatcher(a=source_lines(before), b=source_lines(after), autojunk=False)
    equal_lines = {}
    for tag, start, end, new_start, _ in matcher.get_opcodes():
        if tag == "equal":
            equal_lines.update((old + 1, new_start + old - start + 1) for old in range(start, end))
    if not before_count:
        # A prior insertion may have fragmented background evidence. Without a
        # unique current span, fail closed on rewrites rather than lose protection.
        return all(line in equal_lines and equal_lines[line] in mapped_lines(line, patches)
                   for line in range(1, len(source_lines(before)) + 1))
    offset = 0
    while True:
        position = before.find(evidence, offset)
        if position < 0:
            break
        start = before.count("\n", 0, position) + 1
        end = start + evidence.count("\n")
        if any(line not in equal_lines or equal_lines[line] not in mapped_lines(line, patches)
               for line in range(start, end + 1)):
            return False
        offset = position + len(evidence)
    return True


def validate_patch(task, proposal, ruff_runner=run_ruff):
    out = StaticResult(status="failed", base_source_hash=task.source_hash)
    if (proposal.task_id != task.task_id or proposal.base_source_hash != task.source_hash
            or sorted(proposal.goal_ids) != sorted(g.goal_id for g in task.goals)):
        out.errors.append("补丁任务、目标或源码哈希不匹配")
        return out
    if proposal.decision != "propose":
        out.errors.append("未提出补丁")
        return out
    if len(source_lines(task.source)) > 2 and any(
            p.line_start == 1 and p.line_end == len(source_lines(task.source)) for p in proposal.patches):
        out.errors.append("必须提交局部修改，不能整文件重写")
        return out
    check = apply_and_validate_patches(task.source, proposal.patches, task.filename, ruff_runner=ruff_runner)
    if not check.valid:
        out.errors = [*check.errors, *check.new_diagnostics]
        return out
    code = check.patched_code
    out.candidate_code, out.candidate_hash, out.diff = code, digest(code), check.diff
    try:
        normalize_and_validate(code, task.filename)
    except ValueError as exc:
        out.errors.append(str(exc))
    if code == task.source:
        out.errors.append("补丁没有实际修改")
    try:
        if executable_ast(task.source) == executable_ast(code):
            out.errors.append("仅修改注释、文档或格式，未修改可执行代码，不能视为缺陷修复")
    except SyntaxError:
        pass  # Syntax repairs are checked by the existing AST validation.
    if function_names(task.source) - function_names(code):
        out.errors.append("不允许删除有效函数或类")
    if suppressions(code) != suppressions(task.source):
        out.errors.append("不允许修改检查抑制配置")
    before_ast, bstatus = analyze_ast(task.source)
    after_ast, astatus = analyze_ast(code)
    before_ruff, br = ruff_runner(task.source, task.filename)
    after_ruff, ar = ruff_runner(code, task.filename)
    out.tool_statuses = [br, ar]
    from review_agent.models import ToolStatus
    out.tool_statuses += [ToolStatus(name="ast_before", status=bstatus), ToolStatus(name="ast_after", status=astatus)]
    out.syntax = "passed" if astatus == "completed" else "failed"
    out.before, out.after = before_ast + before_ruff, after_ast + after_ruff
    if ar.status != "completed" or br.status not in ("completed", "syntax_error"):
        out.status = "error"
        out.errors.append("Ruff 不可用或比较不完整")
        return out
    out.new_diagnostics, matches = compare_diagnostics(out.before, out.after, proposal.patches)
    if out.new_diagnostics:
        out.errors.append("出现新增或无法可靠匹配的 AST/Ruff 诊断")
        out.errors.extend(f"{d.rule_id}，当前候选第 {d.line_start} 行：{d.explanation}" for d in out.new_diagnostics)
    addressed = {goal for patch in proposal.patch_targets for goal in patch.goal_ids}
    for target in task.targets:
        if target.kind != "review_issue" or target.goal_id not in addressed:
            continue
        for finding in target.findings:
            if finding.origin == "model":
                continue
            # Old review positions are never interpreted as current positions.
            current = [s for s in out.before if s.origin == finding.origin and s.rule_id == finding.rule_id
                       and (target.source_hash != task.source_hash or s.line_start == finding.line_start)
                       and (s.evidence == finding.evidence or finding.origin == "ast"
                            and evidence_tokens(finding.evidence) is not None
                            and evidence_tokens(s.evidence) == evidence_tokens(finding.evidence))]
            if len(current) != 1:
                out.errors.append("无法唯一定位指定静态目标（包括同列诊断不确定性）")
            elif any(old == current[0] for old, _ in matches) or any(
                    n.origin == current[0].origin and n.rule_id == current[0].rule_id
                    and n.line_start in mapped_lines(current[0].line_start, proposal.patches)
                    for n in out.after):
                out.errors.append("指定静态目标仍存在或列级对应无法确认")
            else:
                out.target_checks.append(f"{target.goal_id}: 静态目标信号已消失")
    for pending in task.pending_findings:
        if not pending_evidence_preserved(task.source, code, pending.evidence, proposal.patches):
            out.errors.append("待确认事项的原文证据被改写或删除")
        elif task.source.count(pending.evidence) != code.count(pending.evidence):
            out.target_checks.append("待确认证据的原始行按顺序保留；插入新行仅改变片段连续性，未将待确认事项视为解决")
    out.status = "failed" if out.errors else "passed"
    return out


def separate_pending_conflicts(task, proposal):
    """Defer complete linked goals; never keep half of a dependent patch group."""
    lines = source_lines(task.source)
    blocked = set()
    for index, patch in enumerate(proposal.patches):
        if "\n".join(lines[patch.line_start - 1:patch.line_end]) != patch.original:
            return None
        changed = list(lines)
        changed[patch.line_start - 1:patch.line_end] = source_lines(patch.replacement)
        candidate = "\n".join(changed) + ("\n" if task.source.endswith("\n") else "")
        if any(not pending_evidence_preserved(task.source, candidate, pending.evidence, [patch])
               for pending in task.pending_findings):
            blocked.add(index)
    if not blocked:
        return None
    goals = set()
    while True:
        goals.update(g for link in proposal.patch_targets if link.patch_index in blocked for g in link.goal_ids)
        expanded = {link.patch_index for link in proposal.patch_targets if goals.intersection(link.goal_ids)}
        if expanded <= blocked:
            break
        blocked |= expanded
    data = proposal.model_dump()
    keep = [i for i in range(len(proposal.patches)) if i not in blocked]
    if not keep:
        return None
    remap = {old: new for new, old in enumerate(keep)}
    data["patches"] = [data["patches"][i] for i in keep]
    data["patch_targets"] = [{**link, "patch_index": remap[link["patch_index"]]}
                             for link in data["patch_targets"] if link["patch_index"] in remap]
    data["unaddressed"].update({g: "程序暂缓：与待确认事项共享代码，相关补丁整组未应用；其他独立修改继续验证。" for g in goals})
    data["summary"] = "保留可独立应用的修改；与待确认事项冲突的目标已暂缓。原方案说明：" + data["summary"][:3800]
    return type(proposal).model_validate(data)


def cumulative_diff(original, candidate):
    return "\n".join(difflib.unified_diff(source_lines(original), source_lines(candidate),
                                         fromfile="a/solution.py", tofile="b/solution.py", lineterm=""))
