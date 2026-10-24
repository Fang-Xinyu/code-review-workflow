import json
from pydantic import ValidationError

from review_agent.ast_rules import analyze_ast
from review_agent.llm import RecordingModel, parse_json_object
from review_agent.ruff_adapter import run_ruff
from review_agent.tools import EvidenceTools, TOOL_SCHEMAS
from review_agent.source import source_lines
from schemas import FixTask, FixProposal
from validation.patch_checks import executable_ast
from .prompts import FIXER


class FixProtocolError(ValueError):
    """Only program-generated diagnostics, never raw model/provider text."""


def locate_patches(source, proposal):
    """Correct only unambiguous exact-text locations; never fuzzy-match code."""
    lines = source_lines(source)
    for patch in proposal.patches:
        original = source_lines(patch.original)
        if "\n".join(lines[patch.line_start - 1:patch.line_end]) != patch.original:
            positions = [i for i in range(len(lines) - len(original) + 1)
                         if lines[i:i + len(original)] == original]
            if not original or len(positions) != 1:
                continue
            patch.line_start = positions[0] + 1
            patch.line_end = positions[0] + len(original)
        # Remove unchanged context, preserving an original anchor for insertions.
        replacement = source_lines(patch.replacement)
        while len(original) > 1 and len(replacement) > 1 and original[0] == replacement[0]:
            original.pop(0)
            replacement.pop(0)
            patch.line_start += 1
        while len(original) > 1 and len(replacement) > 1 and original[-1] == replacement[-1]:
            original.pop()
            replacement.pop()
            patch.line_end -= 1
        patch.original = "\n".join(original)
        patch.replacement = "\n".join(replacement)
    return proposal


def documentation_only(source, proposal):
    if proposal.decision != "propose":
        return False
    lines = source_lines(source)
    patches = sorted(proposal.patches, key=lambda patch: patch.line_start)
    previous_end = 0
    for patch in patches:
        if (patch.line_start <= previous_end or patch.line_end > len(lines)
                or "\n".join(lines[patch.line_start - 1:patch.line_end]) != patch.original):
            return False  # Leave malformed locations to the static validator.
        previous_end = patch.line_end
    changed = list(lines)
    for patch in reversed(patches):
        changed[patch.line_start - 1:patch.line_end] = source_lines(patch.replacement)
    try:
        return executable_ast(source) == executable_ast("\n".join(changed))
    except (SyntaxError, ValueError, TypeError):
        return False


class Fixer:
    def __init__(self, model):
        self.base_model = model
        self.recording = None
        self.observations = []

    def propose(self, task: FixTask) -> FixProposal:
        self.recording = RecordingModel(self.base_model, max_calls=2)
        self.recording.stage = "fixer"
        self.observations = []
        static, _ = analyze_ast(task.source)
        ruff, _ = run_ruff(task.source)
        tools = EvidenceTools(task.source, [*static, *ruff])
        payload = {"task": task.model_dump(), "schema": FixProposal.model_json_schema(),
                   "current_source_numbered": "\n".join(f"{i}: {line}" for i, line in enumerate(source_lines(task.source), 1))}
        for index in range(2):
            result = self.recording.invoke(FIXER, json.dumps(payload, ensure_ascii=False), TOOL_SCHEMAS if index == 0 else None)
            if result.tool_calls:
                if index == 1 or len(result.tool_calls) > 5:
                    raise FixProtocolError("Fixer 工具请求超出预算")
                for call in result.tool_calls:
                    self.observations.append(tools.execute(call.get("name", ""), call.get("args", {}), call.get("id", "")))
                payload["tool_observations"] = self.observations
                payload["instruction"] = "提交最终结果，禁止工具调用"
                continue
            try:
                data = parse_json_object(result.content)
                # Missing bookkeeping never implies a goal was fixed. Preserve
                # omitted goals explicitly as unaddressed; IDs still validate.
                if isinstance(data.get("patch_targets"), list) and isinstance(data.get("unaddressed", {}), dict):
                    linked = {g for item in data["patch_targets"] if isinstance(item, dict)
                              and isinstance(item.get("goal_ids"), list) for g in item["goal_ids"] if isinstance(g, str)}
                    unaddressed = data.setdefault("unaddressed", {})
                    for goal in task.goals:
                        if goal.goal_id not in linked:
                            unaddressed.setdefault(goal.goal_id, "模型未为此目标关联补丁，本轮未处理。")
                proposal = FixProposal.model_validate(data)
                if (proposal.task_id != task.task_id or proposal.base_source_hash != task.source_hash
                        or sorted(proposal.goal_ids) != sorted(g.goal_id for g in task.goals)):
                    raise FixProtocolError("任务、源码快照或目标标识不匹配；必须逐字复制当前 task 的标识")
            except ValueError as exc:
                detail = str(exc) if isinstance(exc, FixProtocolError) else "JSON 格式无效"
                if isinstance(exc, ValidationError):
                    detail = "；".join(".".join(map(str, e["loc"])) + ": " + e["type"]
                                      for e in exc.errors(include_input=False, include_context=False)[:5])
                    known = {"重复目标", "拒绝修复必须无补丁并说明缺失信息", "propose 必须包含补丁",
                             "每处补丁必须恰好关联一次目标", "补丁目标不完整或包含未知目标"}
                    for error in exc.errors(include_input=False):
                        message = str(error.get("ctx", {}).get("error", ""))
                        if message in known:
                            detail += "；" + message
                if result.finish_reason == "length":
                    detail = "模型输出达到长度上限，JSON 被截断"
                if index:
                    raise FixProtocolError("修改方案未通过协议检查：" + detail) from None
                payload["format_error"] = "上次输出不符合 schema：" + detail + "；在最后一次调用修正格式并提交"
                continue
            proposal = locate_patches(task.source, proposal)
            if documentation_only(task.source, proposal):
                if index:
                    raise FixProtocolError("修改方案仅改变注释、文档或格式，未修改可执行代码")
                payload["rejected_proposal"] = proposal.model_dump()
                payload["repair_feedback"] = (
                    "上次补丁只修改注释、文档或格式，没有修改可执行代码，不能解决代码缺陷。"
                    "请重新对照 task.tests、测试来源及目标证据，在最后一次调用提交对应逻辑修复；"
                    "若必要约定确实不足，返回 cannot_fix 并说明缺失信息。禁止把解释现有错误当成修复。")
                continue
            return proposal
        raise ValueError("Fixer 未提交结果")
