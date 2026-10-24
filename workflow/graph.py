import shutil
import time
import uuid
from pathlib import Path

from langgraph.graph import END, StateGraph

from agents.fixer import Fixer, FixProtocolError
from agents.reviewer import Reviewer, AssessmentProtocolError
from agents.model_errors import describe_model_failure
from reporting import export_report, new_run_directory, save_json
from review_agent.llm import RecordingModel, ModelRequestError
from schemas import (Inputs, FixTask, Candidate, Attempt, Event, Feedback, WorkflowReport, digest)
from validation.patch_checks import validate_patch, cumulative_diff, mapped_lines, separate_pending_conflicts
from validation.test_runner import TestRunner
from .routing import review_complete, review_goals, failure_goals, compare_tests, comparable_unattributed, CODE_FAILURES
from .state import WorkflowState


class Workflow:
    def __init__(self, model, model_name="unknown", runner=None, progress_callback=None):
        self.base_model = model
        self.model_name = model_name
        self.runner = runner or TestRunner()
        self.progress_callback = progress_callback
        self.directory = None
        self.retain_run_files = False
        self.gateway = None
        self.reviewer = None
        self.fixer = None
        self.last_state = None
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        nodes = {"initialize": self.initialize, "preflight": self.preflight, "generate_tests": self.generate_tests, "baseline": self.baseline,
                 "initial_review": self.initial_review, "select": self.select, "fixer": self.propose,
                 "static": self.static, "tests": self.tests, "final_review": self.final_review,
                 "assessment": self.assessment, "finish": self.finish}
        for name, fn in nodes.items():
            graph.add_node(name, self._node(name, fn))
        graph.set_entry_point("initialize")
        edges = {"initialize": {"preflight": "preflight", "initial_review": "initial_review"},
                 "preflight": {"initial_review": "initial_review"},
                 "generate_tests": {"baseline": "baseline"}, "baseline": {"select": "select"},
                 "initial_review": {"baseline": "baseline", "generate_tests": "generate_tests"}, "select": {"fixer": "fixer", "final_review": "final_review"},
                 "fixer": {"static": "static"}, "static": {"tests": "tests", "select": "select"},
                 "tests": {"select": "select", "final_review": "final_review"},
                 "final_review": {"assessment": "assessment", "select": "select"}, "assessment": {}}
        for name, mapping in edges.items():
            graph.add_conditional_edges(name, lambda s: s["route"], {**mapping, "finish": "finish"})
        graph.add_edge("finish", END)
        return graph.compile()

    def _node(self, name, fn):
        def invoke(state):
            s = dict(state)
            role = "Reviewer" if "review" in name or name in ("assessment", "generate_tests") else "Fixer" if name == "fixer" else "Validation" if name in ("static", "tests", "baseline", "preflight") else "Workflow"
            started = time.monotonic()
            self.gateway.stage = name

            def event(phase, message):
                e = Event(node=name, phase=phase, role=role, attempt=s.get("attempt_count", 0),
                          source_hash=s["working_hash"], elapsed_ms=int((time.monotonic() - started) * 1000), message=message)
                s["events"] = [*s.get("events", []), e]
                if self.progress_callback:
                    self.progress_callback(e.model_dump())
            event("start", "阶段开始")
            try:
                fn(s)
                message = s.get("stop_reason") or s.get("route", "阶段完成")
                if name in ("static", "tests") and s.get("attempt_history"):
                    message = s["attempt_history"][-1].reason or message
                event("complete", message)
            except Exception as exc:
                # Exception bodies can contain credentials or provider responses; never persist them.
                s.update(status="error", stop_reason=f"{name} 阶段失败（{type(exc).__name__}）", route="finish")
                if isinstance(exc, ModelRequestError):
                    s["stop_reason"] += "；" + describe_model_failure(exc)
                if isinstance(exc, AssessmentProtocolError):
                    s["stop_reason"] += "；" + str(exc)
                if isinstance(exc, FixProtocolError):
                    s["stop_reason"] += "；" + str(exc)
                    s["attempt_history"][-1].reason = str(exc)
                if name in ("initial_review", "final_review"):
                    s["tool_stats_complete"] = False
                event("failed", s["stop_reason"])
            finally:
                if name == "fixer" and s.get("attempt_history"):
                    s["attempt_history"][-1].tool_observations = list(self.fixer.observations)
                    s["tool_calls"] += len(self.fixer.observations)
                self._save_attempts(s)
                self.last_state = s
            return s
        return invoke

    def run(self, inputs: Inputs, output_dir=None):
        if not isinstance(inputs, Inputs):
            inputs = Inputs.model_validate(inputs)
        run_id = uuid.uuid4().hex
        self.directory = new_run_directory(output_dir or Path("runs") / run_id)
        try:
            generate = inputs.mode == "run" and not inputs.tests
            review_limit = min(3, inputs.max_attempts)
            self.gateway = RecordingModel(self.base_model, max_calls=8 if inputs.mode == "review" else
                                          9 + 8 * review_limit + 2 * inputs.max_attempts + 2 * int(generate))
            self.reviewer = Reviewer(self.gateway, self.model_name)
            self.fixer = Fixer(self.gateway)
            state = WorkflowState(inputs=inputs, run_id=run_id, original_code=inputs.source,
                                  original_hash=digest(inputs.source), working_code=inputs.source,
                                  working_hash=digest(inputs.source), tests_hash=digest(inputs.tests),
                                  tests_origin="none" if inputs.mode == "review" else "generated" if generate else "user",
                                  requirement_hash=digest(inputs.requirement), attempt_count=0,
                                  max_attempts=inputs.max_attempts, attempt_history=[], goal_history=[],
                                  review_count=0, review_limit=review_limit, pending_history=[],
                                  included_tasks=[], handled=[], feedback=[], events=[], coverage_flags=[],
                                  seen_candidate_hashes=[digest(inputs.source)], tool_calls=0, tool_stats_complete=True, environment={},
                                  status="needs_attention", stop_reason="", route="finish", goals=[], targets=[])
            state = self.graph.invoke(state, {"recursion_limit": 128})
            report = self.make_report(state)
            export_report(self.directory, report)
            return report
        finally:
            if output_dir is None and not self.retain_run_files:
                shutil.rmtree(self.directory)

    def stop(self, s, reason, status="needs_attention"):
        s.update(route="finish", status=status, stop_reason=reason)

    def initialize(self, s):
        directory = self.directory / "inputs"
        directory.mkdir()
        for filename, text in (("solution.py", s["original_code"]), ("test_solution.py", s["inputs"].tests),
                               ("requirement.md", s["inputs"].requirement)):
            if filename == "test_solution.py" and s["tests_origin"] == "generated":
                continue
            (directory / filename).write_text(text, encoding="utf-8")
            (directory / filename).chmod(0o400)
        s["route"] = "initial_review" if s["inputs"].mode == "review" else "preflight"

    def preflight(self, s):
        s["environment"] = self.runner.preflight()
        s["route"] = "initial_review"

    def generate_tests(self, s):
        generated = self.reviewer.generate_tests(s["original_code"], s["inputs"].requirement, s["initial_review"])
        s["generated_tests"] = generated
        # Freeze before the baseline; the Fixer never generates or changes this suite.
        s["inputs"] = Inputs.model_validate(s["inputs"].model_dump() | {"tests": generated.tests})
        s["tests_hash"] = digest(generated.tests)
        path = self.directory / "inputs" / "test_solution.py"
        path.write_text(generated.tests, encoding="utf-8")
        path.chmod(0o400)
        save_json(self.directory / "test_generation.json", generated)
        s["route"] = "baseline"

    def baseline(self, s):
        test = self.runner.run_tests(s["original_code"], s["inputs"].tests, self.directory / "baseline")
        self.check_test(s, test, s["original_hash"])
        s["baseline_test"] = s["working_test_result"] = test
        save_json(self.directory / "baseline" / "result.json", test)
        if comparable_unattributed(test) and s["goals"]:
            note = "原始测试存在尚未归因的失败；仅按已确认审查问题修复，不将这些失败视为修改授权"
            s["coverage_flags"].append(note)
            s["feedback"] = [Feedback(candidate_hash=s["original_hash"], current_snapshot=s["original_hash"],
                                      rolled_back=False, test_result=test, errors=[note])]
            s["route"] = "select"
        elif test.status not in ("passed", "failed") or (test.status == "failed" and test.failure_kind not in CODE_FAILURES):
            details = []
            if test.skipped:
                details.append(f"{len(test.skipped)} 个检查被跳过")
            if test.xfailed:
                details.append(f"{len(test.xfailed)} 个检查标记为预期失败")
            if not test.complete or test.truncated:
                details.append("执行报告不完整或被截断")
            self.stop(s, "修复前检查未满足执行条件：" + "；".join(details or [test.failure_kind]) + "；尚未开始修改代码", "error" if test.status == "error" else "needs_attention")
        else:
            report = s["initial_review"]
            if not s["goals"] and test.status == "passed":
                self.stop(s, "存在待人工确认事项" if report.pending_findings else "基线完整通过，初审无须修改",
                          "needs_attention" if report.pending_findings else "no_repair_needed")
            else:
                s["route"] = "select"

    def check_test(self, s, test, source_hash):
        if test.source_hash != source_hash or test.tests_hash != s["tests_hash"]:
            raise ValueError("执行结果哈希不匹配")

    def initial_review(self, s):
        report = self.reviewer.review(s["original_code"], "solution.py", s["inputs"].requirement)
        s["initial_review"] = report
        s["current_review"] = report
        s["pending_history"] = list(report.pending_findings)
        save_json(self.directory / "initial_review.json", report)
        s["tool_calls"] += report.usage.tool_calls
        if s["inputs"].mode == "review":
            self.stop(s, "仅审查完成；未执行行为测试", "review_completed")
            return
        if not review_complete(report, allow_syntax=True):
            s["coverage_flags"].append("初审覆盖不完整，暂停自动修复")
            self.stop(s, "初审工具、取证或复核覆盖不足")
            return
        s["goals"], s["targets"] = review_goals(report)
        s["route"] = "generate_tests" if s["tests_origin"] == "generated" else "baseline"

    def select(self, s):
        if s["attempt_count"] >= s["max_attempts"]:
            test = s["working_test_result"]
            self.stop(s, f"已达到 {s['max_attempts']} 次修复上限；当前保留版本测试通过 {len(test.passed)}/{len(test.collected)} 项，"
                      f"仍有 {len(test.failed)} 项失败；已保留的修改可以查看和下载")
            return
        goals, targets = failure_goals(s["working_test_result"])
        generated_without_contract = s["tests_origin"] == "generated" and not s["inputs"].requirement.strip()
        if generated_without_contract:
            # Model-created assertions are not user authorization to redefine behavior.
            goals, targets = [], []
        remaining = [(g, t) for g, t in zip(s["goals"], s["targets"]) if g.goal_id not in s["handled"]]
        # A single defect may fail many tests. Prioritize distinct confirmed
        # review issues, with the full failing test report as supporting context.
        goals = [g for g, _ in remaining] + goals
        targets = [t for _, t in remaining] + targets
        if not goals:
            if generated_without_contract and s["review_count"] < s["review_limit"]:
                s["route"] = "final_review"
                return
            self.stop(s, "已确认目标已尝试处理；仍有未归因测试失败，不能交付修复版本"
                      if comparable_unattributed(s["working_test_result"]) else "没有可确定执行的目标；需求或失败归因不足")
            return
        attempt = s["attempt_count"] + 1
        feedback = [f.model_copy(update={"current_snapshot": s["working_hash"]}) for f in s["feedback"][-2:]]
        if not feedback:
            feedback = [Feedback(candidate_hash=s["working_hash"], current_snapshot=s["working_hash"],
                                 rolled_back=False, test_result=s["working_test_result"])]
        reference = s.get("current_review", s["initial_review"])
        task = FixTask(task_id=s["run_id"] + f"-{attempt:02d}", attempt=attempt,
                       source=s["working_code"], source_hash=s["working_hash"], requirement=s["inputs"].requirement,
                       requirement_hash=s["requirement_hash"], tests_hash=s["tests_hash"],
                       tests=s["inputs"].tests, tests_origin=s["tests_origin"], goals=goals[:5], targets=targets[:5],
                       reference_review=reference, review_source_hash=reference.source_hash,
                       pending_findings=s.get("pending_history", s["initial_review"].pending_findings), feedback=feedback)
        s.update(active_task=task, active_goals=task.goals, attempt_count=attempt, route="fixer")
        s["goal_history"] = [*s["goal_history"], task]
        s["attempt_history"] = [*s["attempt_history"], Attempt(task=task, next_source_hash=s["working_hash"])]

    def propose(self, s):
        proposal = self.fixer.propose(s["active_task"])
        s["attempt_history"][-1].proposal = proposal
        if (proposal.task_id != s["active_task"].task_id or proposal.base_source_hash != s["working_hash"]
                or sorted(proposal.goal_ids) != sorted(g.goal_id for g in s["active_goals"])):
            raise ValueError("Fixer 任务、目标或快照标识不匹配")
        if proposal.decision == "cannot_fix":
            s["attempt_history"][-1].reason = proposal.summary
            self.stop(s, "Fixer 拒绝猜测：" + "; ".join(proposal.missing_information))
        else:
            s["route"] = "static"

    def static(self, s):
        attempt = s["attempt_history"][-1]
        static = validate_patch(s["active_task"], attempt.proposal)
        if static.errors and set(static.errors) == {"待确认事项的原文证据被改写或删除"}:
            filtered = separate_pending_conflicts(s["active_task"], attempt.proposal)
            if filtered:
                attempt.unfiltered_proposal = attempt.proposal
                attempt.proposal = filtered
                static = validate_patch(s["active_task"], filtered)
        attempt.static = static
        if static.candidate_hash:
            if static.candidate_hash in s["seen_candidate_hashes"]:
                attempt.reason = "重复候选，停止" + ("：" + "; ".join(static.errors) if static.errors else "")
                self.stop(s, attempt.reason)
                return
            s["seen_candidate_hashes"] = [*s["seen_candidate_hashes"], static.candidate_hash]
        if static.status != "passed":
            attempt.reason = "静态验证未通过：" + "; ".join(static.errors)
            s["feedback"] = [*s["feedback"], Feedback(candidate_hash=static.candidate_hash or s["working_hash"],
                current_snapshot=s["working_hash"], diff=static.diff or "", rolled_back=True,
                errors=static.errors, rejected_candidate=static.candidate_code)]
            if static.status == "error":
                self.stop(s, "静态工具执行异常", "error")
            else:
                s["route"] = "select"
            return
        s["candidate"] = Candidate(source_hash=static.candidate_hash, code=static.candidate_code, diff=static.diff)
        s["route"] = "tests"

    def tests(self, s):
        attempt = s["attempt_history"][-1]
        candidate = s["candidate"]
        test = self.runner.run_tests(candidate.code, s["inputs"].tests,
                                     self.directory / "attempts" / f'{s["attempt_count"]:02d}' / "execution")
        self.check_test(s, test, candidate.source_hash)
        attempt.tests = test
        retained, regression, reason, stop = compare_tests(s["baseline_test"], s["working_test_result"], test,
                                                         allow_unattributed=bool(s["goals"]))
        attempt.regression, attempt.reason, attempt.retained = regression, reason, retained
        if retained:
            s["final_review"] = None
            s["working_code"], s["working_hash"], s["working_test_result"] = candidate.code, candidate.source_hash, test
            addressed = {g for p in attempt.proposal.patch_targets for g in p.goal_ids}
            s["handled"] = sorted(set(s["handled"]) | addressed)
            included = attempt.task.model_copy(update={"goals": [g for g in attempt.task.goals if g.goal_id in addressed],
                                                       "targets": [t for t in attempt.task.targets if t.goal_id in addressed]})
            s["included_tasks"] = [*s["included_tasks"], included]
        attempt.next_source_hash = s["working_hash"]
        s["feedback"] = [*s["feedback"], Feedback(candidate_hash=candidate.source_hash, current_snapshot=s["working_hash"],
            diff=candidate.diff, rolled_back=not retained, rejected_candidate=None if retained else candidate.code, test_result=test)]
        if stop:
            self.stop(s, reason, "error" if test.status == "error" else "needs_attention")
        elif retained and test.status == "passed":
            remaining = any(g.goal_id not in s["handled"] for g in s["goals"])
            s["route"] = "select" if remaining and s["attempt_count"] < s["max_attempts"] else "final_review"
        else:
            s["route"] = "select"

    def final_review(self, s):
        s["review_count"] += 1
        s["final_review"] = self.reviewer.review(s["working_code"], "solution.py", s["inputs"].requirement)
        s["tool_calls"] += s["final_review"].usage.tool_calls
        s["pending_history"] += [f for f in s["final_review"].pending_findings if f not in s["pending_history"]]
        save_json(self.directory / "final_review.json", s["final_review"])
        save_json(self.directory / "reviews" / f'{s["review_count"]:02d}.json', s["final_review"])
        if (s["final_review"].findings and review_complete(s["final_review"])
                and s["attempt_count"] < s["max_attempts"] and s["review_count"] < s["review_limit"]):
            s["current_review"] = s["final_review"]
            s["goals"], s["targets"] = review_goals(s["final_review"])
            s["handled"] = []
            s["route"] = "select"
            return
        if s["working_test_result"].status != "passed":
            self.stop(s, f"当前固定测试仍有 {len(s['working_test_result'].failed)} 项失败；"
                      "复审未能提供可继续执行的确认目标，或修复/复审额度已耗尽；"
                      "请核对剩余代码问题与生成测试的约定，不能为迎合模型生成的断言擅自改变功能")
            return
        s["route"] = "assessment"

    def assessment(self, s):
        # Consolidate retried goals into first included task; preserve full contexts separately.
        seen, tasks = set(), []
        for task in s["included_tasks"]:
            goals = [g for g in task.goals if g.goal_id not in seen]
            seen.update(g.goal_id for g in goals)
            if goals:
                data = task.model_dump()
                data["goals"] = [g.model_dump() for g in goals]
                tasks.append(data)
        candidate = Candidate(code=s["working_code"], source_hash=s["working_hash"],
                              diff=cumulative_diff(s["original_code"], s["working_code"]))
        context = {"run_id": s["run_id"], "original_source": s["original_code"], "tasks": tasks,
                   "attempts": [a.model_dump() for a in s["attempt_history"]], "requirement": s["inputs"].requirement}
        assessment = self.reviewer.assess_fix(context, candidate, s["attempt_history"][-1].static,
                                             s["working_test_result"], s["final_review"])
        s["fix_assessment"] = assessment
        final = s["final_review"]
        complete = review_complete(final)
        supported = all(a.verdict == "resolved" for a in assessment.assessments)
        new_findings = [f for f in final.findings if not self.matches_initial(s, f)]
        if supported and complete and not new_findings:
            s["accepted_candidate"] = candidate
            manual = s.get("pending_history") or s["initial_review"].pending_findings or final.pending_findings
            outstanding = self.outstanding(s)
            self.stop(s, "候选获接受；仍有剩余或人工事项" if outstanding or manual else "目标获支持，测试完整通过，最终审查完整",
                      "needs_attention" if outstanding or manual else "verified")
        else:
            reasons = []
            unresolved = [a for a in assessment.assessments if a.verdict != "resolved"]
            if unresolved:
                reasons.append(f"{len(unresolved)} 个修复目标未获确认")
            if not complete:
                reasons.append("最终审查覆盖不完整")
            if new_findings:
                reasons.append(f"最终复审发现 {len(new_findings)} 个未与初审对应的确认问题，详见修复后的情况")
            if s["attempt_count"] >= s["max_attempts"]:
                reasons.append(f"已达到 {s['max_attempts']} 轮修复上限")
            elif s["review_count"] >= s["review_limit"]:
                reasons.append(f"已达到 {s['review_limit']} 次完整复审上限")
            self.stop(s, "固定测试已全部通过，但" + "；".join(reasons) + "；保留修改代码，尚未全部通过最终核验")

    def matches_initial(self, s, final):
        for old in s["initial_review"].findings:
            if (old.origin, old.rule_id, old.explanation) != (final.origin, final.rule_id, final.explanation):
                continue
            positions = {old.line_start}
            for a in s["attempt_history"]:
                if a.retained:
                    positions = {new for line in positions for new in mapped_lines(line, a.proposal.patches)}
            if final.line_start in positions and old.evidence == final.evidence:
                return True
        return False

    def outstanding(self, s):
        final, initial = s.get("final_review"), s.get("initial_review")
        if not final:
            return list(initial.findings) if initial else []
        # Only the review of the current snapshot can describe current defects.
        # Initial goals remain in the attempt history, not stale current findings.
        findings = list(final.findings)
        for target in s["targets"]:
            if target.goal_id not in s["handled"]:
                findings += [f for f in target.findings if f not in findings and f.evidence in s["working_code"]]
        return findings

    def finish(self, s):
        # Review the retained partial snapshot once, even when its tests failed.
        # This is diagnostic only and never promotes it to accepted_candidate.
        if (s["working_hash"] != s["original_hash"] and not s.get("final_review")
                and s["review_count"] < s["review_limit"]
                and not any(e.node == "final_review" and e.source_hash == s["working_hash"] for e in s["events"])):
            self.gateway.stage = "final_review"
            s["review_count"] += 1
            started = time.monotonic()
            def review_event(phase, message):
                event = Event(node="final_review", phase=phase, role="Reviewer", attempt=s["attempt_count"],
                              source_hash=s["working_hash"], elapsed_ms=int((time.monotonic() - started) * 1000), message=message)
                s["events"] = [*s["events"], event]
                if self.progress_callback:
                    self.progress_callback(event.model_dump())
            review_event("start", "复审已保留的部分修改；不会将未通过测试的版本标为成功")
            try:
                s["final_review"] = self.reviewer.review(s["working_code"], "solution.py", s["inputs"].requirement)
                s["tool_calls"] += s["final_review"].usage.tool_calls
                save_json(self.directory / "final_review.json", s["final_review"])
                save_json(self.directory / "reviews" / f'{s["review_count"]:02d}.json', s["final_review"])
                review_event("complete", "部分修改复审完成，原停止原因和修改代码已保留")
            except Exception:
                s["tool_stats_complete"] = False
                s["coverage_flags"].append("保留的部分修改版未能完成复审；初审问题不能直接判定为仍然存在")
                review_event("failed", "部分修改复审未完成，原停止原因和修改代码已保留")
        s["route"] = "finish"

    def _save_attempts(self, s):
        for index, a in enumerate(s.get("attempt_history", []), 1):
            directory = self.directory / "attempts" / f"{index:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            save_json(directory / "attempt.json", a)
            save_json(directory / "task.json", a.task)
            (directory / "base.py").write_text(a.task.source, encoding="utf-8")
            if a.proposal:
                save_json(directory / "proposal.json", a.proposal)
            if a.unfiltered_proposal:
                save_json(directory / "proposal_before_filter.json", a.unfiltered_proposal)
            if a.static:
                save_json(directory / "static.json", a.static)
                if a.static.candidate_code is not None:
                    (directory / "candidate.py").write_text(a.static.candidate_code, encoding="utf-8")
                    (directory / "candidate.diff").write_text(a.static.diff or "", encoding="utf-8")
            if a.tests:
                save_json(directory / "tests.json", a.tests)

    def make_report(self, s):
        initial, final = s.get("initial_review"), s.get("final_review")
        totals = self.gateway.totals()
        manual = list(s.get("pending_history", initial.pending_findings if initial else []))
        if final:
            manual += [p for p in final.pending_findings if p not in manual]
        limitations = [*s["coverage_flags"], *(initial.limitations if initial else []), *(final.limitations if final else [])]
        if s["tests_origin"] == "generated":
            limitations.append("测试由模型根据原始代码、已确认审查问题及可用约定生成；通过仅表示满足这些生成测试，不等同于用户指定验收或完整正确性"
                               if s.get("generated_tests") else "未提供用户测试；尚未完成测试生成，不存在行为验收结果")
        if not s["tool_stats_complete"]:
            limitations.append("审查中断，工具调用总数无法完整确认；逻辑模型与 HTTP 调用记录仍保留")
        if not s.get("accepted_candidate") and s["attempt_history"]:
            limitations.append("尝试产物未作为交付代码接受")
        partial = None
        if not s.get("accepted_candidate") and s["working_hash"] != s["original_hash"]:
            partial = Candidate(code=s["working_code"], source_hash=s["working_hash"],
                                diff=cumulative_diff(s["original_code"], s["working_code"]))
        return WorkflowReport(run_id=s["run_id"], mode=s["inputs"].mode, status=s["status"], reason=s["stop_reason"],
            source_hash=s["original_hash"], tests_hash=s["tests_hash"], requirement_hash=s["requirement_hash"],
            requirement=s["inputs"].requirement, working_hash=s["working_hash"], baseline_test=s.get("baseline_test"),
            tests_origin=s["tests_origin"], generated_tests=s.get("generated_tests"),
            initial_review=initial, attempts=s["attempt_history"], final_review=final, fix_assessment=s.get("fix_assessment"),
            accepted_candidate=s.get("accepted_candidate"), partial_candidate=partial, outstanding_findings=self.outstanding(s),
            manual_items=manual, limitations=limitations,
            goals_verified=bool(s.get("fix_assessment")) and all(a.verdict == "resolved" for a in s["fix_assessment"].assessments),
            tests_passed=bool(s.get("working_test_result")) and s["working_test_result"].status == "passed",
            review_complete=review_complete(final) if final else
                            review_complete(initial) if initial and s["working_hash"] == s["original_hash"] else False,
            events=s["events"], model_calls=self.gateway.calls, model_call_limit=self.gateway.max_calls,
            http_attempts=totals["http_attempts"], input_tokens=totals["input_tokens"], output_tokens=totals["output_tokens"],
            tool_calls=s["tool_calls"] if s["tool_stats_complete"] else None, environment=s["environment"])
