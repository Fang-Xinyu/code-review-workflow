from __future__ import annotations

import re

from .ast_rules import RULES


SEVERITIES = {"high": "高", "medium": "中", "low": "低"}
CONFIRMATIONS = {"static_confirmed": "规则检查已确认", "model_reviewed": "AI 复核通过", "needs_confirmation": "需要人工确认"}
ORIGINS = {"ast": "Python 语法结构检查（AST）", "ruff": "Python 静态检查工具（Ruff）", "model": "AI 分析"}
STAGES = {"validate_input": "输入检查", "static_analysis": "基础检查", "agent_reasoning": "AI 分析",
          "tool_execution": "读取补充证据", "parse_draft": "整理问题", "reconcile_and_ground": "核对源码证据",
          "reflection": "复核问题", "finalize_report": "生成报告"}
STAGES.update({"draft_repair": "修正初审格式", "reflection_repair": "修正复核格式",
               "followup": "报告追问", "patch_generation": "生成补丁", "patch_review": "复核补丁"})
STOP_REASONS = {"candidate_submitted": "模型提前提交候选", "round_limit": "到达初审轮次上限，要求提交候选",
                "budget_limit": "为格式修正与独立复核预留额度，停止取证",
                "tool_request_rejected": "收尾时仍请求工具，已停止取证并保留静态发现"}
TOOL_STATUSES = {"completed": "已完成", "unavailable": "不可用", "timeout": "超时", "error": "检查失败", "syntax_error": "发现语法错误"}
ACTIONS = {"accepted": "证据通过", "rejected": "排除问题", "hidden": "隐藏问题", "relocated": "修正位置",
           "backfilled": "补回规则发现", "deduplicated": "合并重复问题", "truncated": "限制展示数量",
           "rule_unlinked": "移除无依据的规则引用", "reflection": "AI 复核结果"}
RULE_LABELS = {rule_id: rule.title for rule_id, rule in RULES.items()}
RULE_LABELS.update({
    "F401": "导入但未使用", "F403": "使用星号导入，无法确定名称来源",
    "F405": "名称可能来自星号导入", "F541": "格式化字符串没有占位符",
    "F601": "字典包含重复的字面量键", "F602": "字典包含重复的变量键",
    "F811": "重复定义未使用的名称", "F821": "未定义变量",
    "F822": "__all__ 中包含未定义名称", "F823": "局部变量在赋值前被引用",
    "F841": "局部变量赋值后未使用", "F901": "错误地抛出 NotImplemented",
    "E902": "源码读取错误", "E999": "Python 语法错误",
    "invalid-syntax": "Python 语法错误",
})


def finding_basis(item) -> str:
    if not item.rule_id:
        return ""
    label = RULE_LABELS.get(item.rule_id, item.title.removeprefix(f"{item.rule_id}: "))
    return f"{item.rule_id}：{escape_markdown_tildes(label)}"


_MARKDOWN_TOKENS = re.compile(r"(`+)([\s\S]*?)\1|\\[\s\S]|~")


def escape_markdown_tildes(value: object) -> str:
    """Escape prose tildes while preserving code spans, fences and existing escapes.

    Apply at Markdown output boundaries, never to stored report data or source.
    """
    return _MARKDOWN_TOKENS.sub(
        lambda match: r"\~" if match.group(0) == "~" else match.group(0),
        str(value),
    )
