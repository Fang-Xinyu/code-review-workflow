from __future__ import annotations

import json
import time
from typing import Any, Callable

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from .ast_rules import analyze_ast
from .config import MAX_PLAN_ROUNDS as PLAN_ROUND_LIMIT, MAX_TOOL_CALLS_PER_ROUND as TOOL_ROUND_LIMIT
from .llm import ModelLike, ModelResult, RecordingModel, parse_json_object
from .models import (
    DraftResponse,
    ExecutionEvent,
    ReviewInput,
    ReviewReport,
    ReviewSummary,
    ToolStatus,
    UsageStats,
)
from .prompts import INITIAL_SYSTEM
from .reflection import reflect
from .ruff_adapter import run_ruff
from .state import ReviewState
from .tools import TOOL_SCHEMAS, EvidenceTools
from .verification import normalize_and_validate, reconcile
from .source import source_lines


class ReviewAgent:
    MAX_PLAN_ROUNDS = PLAN_ROUND_LIMIT
    MAX_TOOL_CALLS_PER_ROUND = TOOL_ROUND_LIMIT

    def __init__(self, model: ModelLike, model_name: str = "unknown", ruff_runner: Callable[..., Any] = run_ruff,
                 progress_callback: Callable[[str, int], None] | None = None):
        self._base_model = model
        self.model = RecordingModel(model)
        self.model_name = model_name
        self.ruff_runner = ruff_runner
        self.progress_callback = progress_callback
        self._tools: EvidenceTools | None = None
        self.graph = self._build_graph()

    def review(self, code: str, filename: str = "input.py", focus: str = "全面检查正确性、异常处理、资源和安全") -> ReviewReport:
        self.model = RecordingModel(self._base_model)
        initial: ReviewState = {"code": code, "filename": filename, "focus": focus, "started_at": time.perf_counter(),
            "messages": [], "validation_records": [], "limitations": [], "events": [], "stage_ms": {},
            "plan_rounds": 0, "tool_calls": 0, "model_calls": 0, "http_attempts": 0,
            "input_tokens": None, "output_tokens": None, "force_final": False}
        result = self.graph.invoke(initial, {"recursion_limit": 32})
        return result["report"]

    def _timed(self, name: str, function: Callable[[ReviewState], dict[str, Any]]) -> Callable[[ReviewState], dict[str, Any]]:
        def wrapper(state: ReviewState) -> dict[str, Any]:
            if self.progress_callback:
                self.progress_callback(name, state.get("plan_rounds", 0) + (1 if name == "agent_reasoning" else 0))
            self.model.stage = name
            start = time.perf_counter()
            update = function(state)
            elapsed = int((time.perf_counter() - start) * 1000)
            timings = dict(state.get("stage_ms", {}))
            timings[name] = timings.get(name, 0) + elapsed
            update["stage_ms"] = timings
            events = update.get("events", [])
            if events and events[-1].stage == name:
                events[-1] = events[-1].model_copy(update={"elapsed_ms": elapsed})
            if "report" in update:
                update["report"].events = events
                update["report"].usage.stage_ms = timings
                update["report"].usage.total_ms = int((time.perf_counter() - state["started_at"]) * 1000)
            return update
        return wrapper

    def _build_graph(self):
        workflow = StateGraph(ReviewState)
        workflow.add_node("validate_input", self._timed("validate_input", self._validate))
        workflow.add_node("static_analysis", self._timed("static_analysis", self._static))
        workflow.add_node("agent_reasoning", self._timed("agent_reasoning", self._reason))
        workflow.add_node("tool_execution", self._timed("tool_execution", self._execute_tools))
        workflow.add_node("parse_draft", self._timed("parse_draft", self._parse))
        workflow.add_node("reconcile_and_ground", self._timed("reconcile_and_ground", self._reconcile))
        workflow.add_node("reflection", self._timed("reflection", self._reflect))
        workflow.add_node("finalize_report", self._timed("finalize_report", self._finalize))
        workflow.set_entry_point("validate_input")
        workflow.add_edge("validate_input", "static_analysis")
        workflow.add_edge("static_analysis", "agent_reasoning")
        workflow.add_conditional_edges("agent_reasoning", self._after_reason, {"tools": "tool_execution", "parse": "parse_draft"})
        workflow.add_edge("tool_execution", "agent_reasoning")
        workflow.add_edge("parse_draft", "reconcile_and_ground")
        workflow.add_edge("reconcile_and_ground", "reflection")
        workflow.add_edge("reflection", "finalize_report")
        workflow.add_edge("finalize_report", END)
        return workflow.compile()

    def _event(self, state: ReviewState, stage: str, message: str) -> list[ExecutionEvent]:
        return [*state.get("events", []), ExecutionEvent(stage=stage, message=message)]

    def _validate(self, state: ReviewState) -> dict[str, Any]:
        code, digest = normalize_and_validate(state["code"], state["filename"], state.get("focus", ""))
        validated = ReviewInput(code=code, filename=state["filename"], focus=state.get("focus", ""), source_hash=digest)
        return {"code": validated.code, "filename": validated.filename, "focus": validated.focus,
                "source_hash": validated.source_hash,
                "events": self._event(state, "validate_input", f"输入通过：{len(source_lines(code))} 行，SHA-256 {digest[:12]}")}

    def _static(self, state: ReviewState) -> dict[str, Any]:
        ast_findings, ast_status = analyze_ast(state["code"])
        ruff_findings, ruff_status = self.ruff_runner(state["code"], state["filename"])
        ast_tool = ToolStatus(name="ast", status=ast_status, detail="已执行内置确定性规则", finding_count=len(ast_findings))
        findings = [*ast_findings, *ruff_findings]
        self._tools = EvidenceTools(state["code"], findings)
        limitations = list(state.get("limitations", []))
        if ruff_status.status != "completed":
            limitations.append(f"Ruff 状态为 {ruff_status.status}，不能解释为无 Ruff 问题。")
        return {"static_findings": findings, "tool_statuses": [ast_tool, ruff_status], "limitations": limitations,
                "events": self._event(state, "static_analysis", f"AST {len(ast_findings)} 条，Ruff {len(ruff_findings)} 条")}

    def _reason(self, state: ReviewState) -> dict[str, Any]:
        rounds = state.get("plan_rounds", 0) + 1
        force = rounds >= self.MAX_PLAN_ROUNDS or self.model.remaining <= 4
        numbered = "\n".join(f"{i:4d} | {line}" for i, line in enumerate(source_lines(state["code"]), 1))
        static = [x.model_dump() for x in state.get("static_findings", [])]
        observations = state.get("messages", [])
        instruction = "已到预算上限，禁止再调工具，立即返回最终 JSON。" if force else f"每轮最多调用 {self.MAX_TOOL_CALLS_PER_ROUND} 次工具追加取证，或直接返回最终 JSON。"
        user = json.dumps({"filename": state["filename"], "focus": state["focus"], "source": numbered,
            "static_findings": static, "tool_observations": observations, "instruction": instruction,
            "remaining_model_calls": self.model.remaining, "draft_schema": DraftResponse.model_json_schema()}, ensure_ascii=False)
        result = self.model.invoke(INITIAL_SYSTEM, user, None if force else TOOL_SCHEMAS)
        stop_reason = ("round_limit" if rounds >= self.MAX_PLAN_ROUNDS else "budget_limit") if force else "candidate_submitted"
        update = self._usage_update(state, result)
        update.update({"plan_rounds": rounds, "last_model_result": result, "force_final": force,
            "initial_stop_reason": stop_reason,
            "events": self._event(state, "agent_reasoning", f"第 {rounds} 轮模型决策")})
        if force and result.tool_calls:
            update["limitations"] = [*state.get("limitations", []), "模型在规划轮次上限后仍请求工具，追加取证已停止。"]
            update["last_model_result"] = ModelResult(content='{"findings":[]}', tool_calls=[])
            update["initial_stop_reason"] = "tool_request_rejected"
        return update

    def _usage_update(self, state: ReviewState, result: ModelResult) -> dict[str, Any]:
        return self.model.totals()

    def _after_reason(self, state: ReviewState) -> str:
        result: ModelResult = state["last_model_result"]
        return "tools" if result.tool_calls else "parse"

    def _execute_tools(self, state: ReviewState) -> dict[str, Any]:
        assert self._tools is not None
        result: ModelResult = state["last_model_result"]
        selected = result.tool_calls[:self.MAX_TOOL_CALLS_PER_ROUND]
        skipped = result.tool_calls[self.MAX_TOOL_CALLS_PER_ROUND:]
        observations = list(state.get("messages", []))
        limitations = list(state.get("limitations", []))
        tool_details = []
        for index, call in enumerate(selected):
            observation = self._tools.execute(call.get("name", ""), call.get("args", {}), call.get("id") or f"call-{state.get('tool_calls', 0)+index+1}")
            observations.append(observation)
            name = str(call.get("name", "unknown"))[:80]
            outcome = "返回数据" if observation.get("ok") else observation.get("error", {}).get("code", "error")
            tool_details.append(f"{name}：{outcome}{'，已截断' if observation.get('truncated') else ''}")
            if observation.get("truncated"):
                limitations_detail = f"工具 {observation.get('tool', 'unknown')} 输出已截断；需按返回的继续位置缩小范围取证。"
                limitations.append(limitations_detail)
        if skipped:
            detail = f"第 {state.get('plan_rounds', 0)} 轮工具请求超过 {self.MAX_TOOL_CALLS_PER_ROUND} 次上限，已跳过 {len(skipped)} 次请求。"
            limitations.append(detail)
            for index, call in enumerate(skipped, len(selected)):
                observations.append(self._tools._error(
                    call.get("id") or f"call-{state.get('tool_calls', 0)+index+1}", "tool_call_limit", detail))
        return {"messages": observations, "limitations": limitations,
                "tool_calls": state.get("tool_calls", 0) + len(selected),
                "events": self._event(state, "tool_execution", f"执行 {len(selected)} 次追加取证；" + "；".join(tool_details))}

    def _parse(self, state: ReviewState) -> dict[str, Any]:
        result: ModelResult = state["last_model_result"]
        error = ""
        calls_update: dict[str, Any] = {}
        for attempt in range(2):
            try:
                parsed = DraftResponse.model_validate(parse_json_object(result.content))
                return {**calls_update, "draft_findings": parsed.findings,
                        "events": self._event(state, "parse_draft", f"候选 JSON 通过，共 {len(parsed.findings)} 条")}
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)[:700]
                if attempt == 1:
                    raise ValueError(f"初审结构化 JSON 两次失败: {error}") from None
                self.model.stage = "draft_repair"
                result = self.model.invoke(INITIAL_SYSTEM, json.dumps({"task": "修复上一个 JSON，不要改变候选语义", "schema_error": error,
                    "invalid_output": result.content[:8000], "schema": DraftResponse.model_json_schema()}, ensure_ascii=False))
                calls_update = self._usage_update(state, result)
        raise AssertionError("unreachable")

    def _reconcile(self, state: ReviewState) -> dict[str, Any]:
        findings, records = reconcile(state["code"], state.get("draft_findings", []), state.get("static_findings", []))
        return {"reviewed_findings": findings, "validation_records": [*state.get("validation_records", []), *records],
                "events": self._event(state, "reconcile_and_ground", f"合并校验后保留 {len(findings)} 条")}

    def _reflect(self, state: ReviewState) -> dict[str, Any]:
        targets = [item for item in state.get("reviewed_findings", [])
                   if item.severity in {"high", "medium"} or item.status == "needs_confirmation"]
        if not targets:
            return {**self.model.totals(), "events": self._event(state, "reflection", "无需独立复核：没有高、中严重度或待确认候选")}
        try:
            findings, records, *_ = reflect(self.model, state.get("reviewed_findings", []), state["code"])
            return {"reviewed_findings": findings, "validation_records": [*state.get("validation_records", []), *records],
                    **self.model.totals(),
                    "events": self._event(state, "reflection", f"选择性复核完成，保留 {len(findings)} 条")}
        except Exception as exc:
            downgraded = [x.model_copy(update={"status": "needs_confirmation",
                "review_note": "复核服务失败，改为待确认，严重度保留"}) if x.origin == "model" else x for x in state.get("reviewed_findings", [])]
            return {"reviewed_findings": downgraded, "limitations": [*state.get("limitations", []), f"选择性复核失败（{type(exc).__name__}）；真实静态发现保留，模型意见改为待确认，严重度保留。"],
                    **self.model.totals(),
                    "events": self._event(state, "reflection", "复核失败，已按安全规则降级")}

    def _finalize(self, state: ReviewState) -> dict[str, Any]:
        items = state.get("reviewed_findings", [])
        formal = [x for x in items if x.status != "needs_confirmation"]
        pending = [x for x in items if x.status == "needs_confirmation"]
        counts = {level: sum(x.severity == level for x in items) for level in ("high", "medium", "low")}
        summary = ReviewSummary(total_formal=len(formal), total_pending=len(pending), **counts,
            text=f"发现 {len(formal)} 条已确认问题和 {len(pending)} 条需要人工确认的问题。")
        total = int((time.perf_counter() - state["started_at"]) * 1000)
        events = self._event(state, "finalize_report", "报告已从最终问题重建")
        report = ReviewReport(filename=state["filename"], source_hash=state["source_hash"], focus=state["focus"], model=self.model_name,
            summary=summary, findings=formal, pending_findings=pending, validation_records=state.get("validation_records", []),
            tool_statuses=state.get("tool_statuses", []), limitations=state.get("limitations", []), events=events,
            usage=UsageStats(calls=self.model.calls, total_ms=total, stage_ms=state.get("stage_ms", {}),
                **self.model.totals(), tool_calls=state.get("tool_calls", 0), model_call_limit=self.model.max_calls,
                plan_rounds=state.get("plan_rounds", 0), plan_round_limit=self.MAX_PLAN_ROUNDS,
                tool_call_limit=(self.MAX_PLAN_ROUNDS - 1) * self.MAX_TOOL_CALLS_PER_ROUND,
                initial_stop_reason=state.get("initial_stop_reason", "candidate_submitted")))
        return {"report": report, "events": events}
