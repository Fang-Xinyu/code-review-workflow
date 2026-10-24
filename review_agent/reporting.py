from __future__ import annotations

import re

from .models import ReviewedFinding, ReviewReport
from .ui_text import (
    ACTIONS,
    CONFIRMATIONS,
    ORIGINS,
    SEVERITIES,
    STAGES,
    STOP_REASONS,
    TOOL_STATUSES,
    escape_markdown_tildes,
    finding_basis,
)


def report_json(report: ReviewReport) -> str:
    return report.model_dump_json(indent=2)


def _finding_md(item: ReviewedFinding, reflection_detail: str = "") -> str:
    basis = f"- 检查依据：{finding_basis(item)}\n" if item.rule_id else ""
    reflection = f"- AI 复核结论：{escape_markdown_tildes(reflection_detail)}\n" if reflection_detail else ""
    validation = f"- 程序核验依据：{escape_markdown_tildes(item.validation_hint)}\n" if item.validation_hint else ""
    review_note = f"- 审查说明：{escape_markdown_tildes(item.review_note)}\n" if item.review_note and not reflection_detail else ""
    fence = "`" * max(3, max((len(part) for part in re.findall(r'`+', item.evidence)), default=0) + 1)
    return f"""### 【{SEVERITIES[item.severity]}严重度】{escape_markdown_tildes(item.title)} · 第 {item.line_start}-{item.line_end} 行

- 问题编号：{item.finding_id}
- 确认情况：{CONFIRMATIONS[item.status]}
{basis}{reflection}{review_note}{validation}- 触发：{escape_markdown_tildes(item.trigger)}
- 解释：{escape_markdown_tildes(item.explanation)}
- 建议：{escape_markdown_tildes(item.suggestion)}

{fence}python
{item.evidence}
{fence}
"""


def report_markdown(report: ReviewReport) -> str:
    reflections = {x.finding_ref: x.detail for x in report.validation_records if x.action == "reflection"}

    def grouped_findings(items):
        groups = []
        for origin, label in ORIGINS.items():
            entries = [_finding_md(x, reflections.get(x.finding_id, "")) for x in items if x.origin == origin]
            if entries:
                # Source headings sit above the individual problem headings.
                groups.append(f"### {label}\n\n" + "\n".join(entry.replace("### ", "#### ", 1) for entry in entries))
        return "\n".join(groups) or "无。\n"

    formal = grouped_findings(report.findings)
    pending = grouped_findings(report.pending_findings)
    tools = "\n".join(f"- {ORIGINS.get(x.name, x.name)}：{TOOL_STATUSES[x.status]} · 发现 {x.finding_count} 条。{escape_markdown_tildes(x.detail)}" for x in report.tool_statuses) or "- 无"
    records = "\n".join(f"- {ACTIONS[x.action]} {x.finding_ref}：{escape_markdown_tildes(x.detail)}" for x in report.validation_records) or "- 无"
    limitations = "\n".join(f"- {escape_markdown_tildes(x)}" for x in report.limitations) or "- 无"
    usage = report.usage
    compatibility_note = "旧版本报告的新增统计字段可能为兼容默认值，不能据此恢复原运行的真实轮数及结束原因。\n" if report.schema_version == "1.0" else ""
    events = "\n".join(f"- {STAGES.get(event.stage, event.stage)}：{escape_markdown_tildes(event.message)}" for event in report.events) or "- 无"
    timings = "\n".join(f"- {STAGES.get(stage, stage)}：{ms / 1000:.2f} 秒" for stage, ms in usage.stage_ms.items()) or "- 无"
    def available(value):
        return "不可用" if value is None else str(value)

    call_rows = "\n".join(
        f"| {i} | {STAGES.get(x.stage, x.stage)} | {x.elapsed_ms / 1000:.3f} | {x.http_attempts} | "
        f"{available(x.input_tokens)} | {available(x.output_tokens)} | {x.answer_chars} | "
        f"{available(x.reasoning_tokens)} | {escape_markdown_tildes(available(x.finish_reason)).replace('|', '&#124;')} | "
        f"{'失败' if x.failed else '完成'} |"
        for i, x in enumerate(usage.calls, 1))
    call_table = ("| 调用 | 阶段 | 秒 | HTTP尝试 | 输入token | 输出token | 答案字符 | 推理token | finish_reason | 状态 |\n"
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n" + call_rows) if call_rows else "无逐次记录。"
    return f"""# TraceReview 审查报告

- 文件：{escape_markdown_tildes(report.filename)}
- SHA-256：{report.source_hash}
- 模型：{escape_markdown_tildes(report.model)}
- 审查目标：{escape_markdown_tildes(report.focus)}
- 数据格式版本：{report.schema_version}

## 最终摘要

{compatibility_note}
{escape_markdown_tildes(report.summary.text)}

确认情况与严重度独立。AI复核通过不代表已运行行为测试；未保留问题不代表代码已证明正确。

## 已确认问题

{formal}

## 需要人工确认

{pending}

## 工具覆盖

{tools}

## 校验记录

{records}

## 已知限制

{limitations}

## 执行轨迹

{events}

## 运行统计

- 总用时：{usage.total_ms / 1000:.2f} 秒
- 模型调用：{usage.model_calls}/{usage.model_call_limit} 次（含格式修正、独立复核和失败调用；HTTP 重试另计）
- 初审轮数：{usage.plan_rounds}/{usage.plan_round_limit} 轮
- 初审结束原因：{STOP_REASONS[usage.initial_stop_reason]}
- HTTP 请求：{usage.http_attempts} 次
- 补充取证：{usage.tool_calls}/{usage.tool_call_limit} 次（不含固定AST/Ruff检查）
- 输入 token 数：{usage.input_tokens if usage.input_tokens is not None else '不可用'}
- 输出 token 数：{usage.output_tokens if usage.output_tokens is not None else '不可用'}

以上统计仅属于生成此报告的一次审查，不包含连接测试、后续追问或补丁。未返回的token用量不可据此推断为零。

### 阶段用时

{timings}

### 逐次模型调用

{call_table}
"""


def snapshot_is_stale(code: str, report: ReviewReport, filename: str | None = None) -> bool:
    from .verification import normalize_and_validate
    if filename is not None and filename != report.filename:
        return True
    try:
        _, digest = normalize_and_validate(code, report.filename)
    except ValueError:
        return True
    return digest != report.source_hash
