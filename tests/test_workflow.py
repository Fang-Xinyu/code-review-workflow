
import pytest

from schemas import Inputs, digest
from workflow.graph import Workflow
from conftest import ScriptedModel, FakeRunner, patch, test_result


def run(tmp_path, inputs, model, runner):
    wf = Workflow(model, runner=runner)
    report = wf.run(inputs, tmp_path / "output")
    return wf, report


def test_clean_no_fixer(tmp_path, inputs):
    model = ScriptedModel()
    _, report = run(tmp_path, inputs, model, FakeRunner())
    assert report.status == "no_repair_needed"
    assert not report.attempts
    assert len(report.model_calls) == 1


def test_real_core_and_workflow_repair(tmp_path, inputs):
    wf, r = run(tmp_path, inputs, ScriptedModel([[patch()]]), FakeRunner(["failed", "passed"]))
    assert r.status == "verified", r.reason
    assert r.tests_passed and r.goals_verified and r.review_complete
    assert r.attempts[0].task.tests == inputs.tests
    assert r.attempts[0].task.tests_origin == "user"
    assert r.accepted_candidate and r.accepted_candidate.code.endswith("return a + b\n")
    assert (wf.directory / "inputs" / "solution.py").read_text() == inputs.source
    assert (wf.directory / "result" / "accepted_solution.py").exists()
    assert len(r.model_calls) == 4
    assert r.http_attempts == 4
    assert [e.node for e in r.events if e.phase == "start"] == ["initialize", "preflight", "initial_review", "baseline", "select", "fixer", "static", "tests", "final_review", "assessment", "finish"]


def test_review_only_never_executes(tmp_path, inputs):
    runner = FakeRunner(preflight_error=True)
    _, r = run(tmp_path, inputs.model_copy(update={"mode": "review"}), ScriptedModel(), runner)
    assert r.status == "review_completed"
    assert not runner.calls and not r.tests_passed and not r.accepted_candidate


def test_environment_stops_before_model(tmp_path, inputs):
    _, r = run(tmp_path, inputs, ScriptedModel(), FakeRunner(preflight_error=True))
    assert r.status == "error" and not r.model_calls and not r.attempts


@pytest.mark.parametrize("verdict", ["unresolved", "uncertain"])
def test_tests_pass_not_enough(tmp_path, inputs, verdict):
    _, r = run(tmp_path, inputs, ScriptedModel([[patch()]], verdict=verdict), FakeRunner(["failed", "passed"]))
    assert r.status == "needs_attention" and r.tests_passed
    assert not r.accepted_candidate
    assert not (tmp_path / "output" / "result" / "accepted_solution.py").exists()


def test_attempt_limit_invalid_static(tmp_path, inputs):
    _, r = run(tmp_path, inputs, ScriptedModel([[patch(original="wrong")]] * 4), FakeRunner(["failed"]))
    assert r.status == "needs_attention"
    assert len(r.attempts) == 4 and not r.accepted_candidate
    assert r.model_call_limit == 41
    assert all(a.tests is None for a in r.attempts)


def test_cannot_fix(tmp_path, inputs):
    _, r = run(tmp_path, inputs, ScriptedModel(), FakeRunner(["failed"]))
    assert r.status == "needs_attention"
    assert len(r.attempts) == 1
    assert r.attempts[0].proposal.decision == "cannot_fix"


def test_duplicate_candidate_stops(tmp_path, inputs):
    _, r = run(tmp_path, inputs, ScriptedModel([[patch()], [patch()]]), FakeRunner(["failed", "failed"]))
    assert len(r.attempts) == 2
    assert "重复候选" in r.reason
    assert not r.accepted_candidate


def test_rollback_feedback_snapshot(tmp_path, inputs):
    def baseline(source, tests):
        return test_result(source, tests, passed=("a",), failed=("b",))
    def regression(source, tests):
        return test_result(source, tests, passed=("b",), failed=("a",))
    def final(source, tests):
        return test_result(source, tests, passed=("a", "b"))
    wf, r = run(tmp_path, inputs, ScriptedModel([[patch()], [patch(replacement="    return (a + b)")]]),
                FakeRunner([baseline, regression, final]))
    assert r.status == "verified"
    first, second = r.attempts
    assert not first.retained and first.regression == ["a"]
    assert second.task.source == inputs.source
    feedback = second.task.feedback[0]
    assert feedback.rolled_back and feedback.current_snapshot == digest(inputs.source)
    assert feedback.candidate_hash == first.tests.source_hash != feedback.current_snapshot
    assert feedback.test_result.source_hash == feedback.candidate_hash
    assert "a + b" in r.accepted_candidate.code
    assert wf.last_state["original_code"] == inputs.source


def test_partial_progress_then_cumulative_assessment(tmp_path, inputs):
    def baseline(source, tests):
        return test_result(source, tests, passed=("a",), failed=("b", "c"))
    def improved(source, tests):
        return test_result(source, tests, passed=("a", "b"), failed=("c",))
    def passed(source, tests):
        return test_result(source, tests, passed=("a", "b", "c"))
    _, r = run(tmp_path, inputs, ScriptedModel([[patch()], [patch(original="    return a + b", replacement="    result = a + b\n    return result")]]),
               FakeRunner([baseline, improved, passed]))
    assert r.status == "verified", r.reason
    assert all(a.retained for a in r.attempts)
    assert r.attempts[1].task.source_hash == r.attempts[0].tests.source_hash
    # 'c' retried across two retained tasks; appears once in final assessment.
    ids = [a.goal_id for a in r.fix_assessment.assessments]
    assert len(ids) == 2 and len(set(ids)) == 2


def test_test_collection_change_stops(tmp_path, inputs):
    def baseline(source, tests):
        return test_result(source, tests, passed=("a",), failed=("b",))
    def reduced(source, tests):
        return test_result(source, tests, passed=("a",))
    _, r = run(tmp_path, inputs, ScriptedModel([[patch()]]), FakeRunner([baseline, reduced]))
    assert r.status == "needs_attention" and "身份集合" in r.reason
    assert not r.attempts[0].retained


def test_model_error_sanitized_and_counted(tmp_path, inputs):
    _, r = run(tmp_path, inputs, ScriptedModel([RuntimeError("API secret: must-not-appear")]), FakeRunner(["failed"]))
    assert r.status == "error"
    assert r.model_calls[-1].failed
    assert "must-not-appear" not in r.model_dump_json()


def test_pending_not_automatic(tmp_path):
    inputs = Inputs(source="def execute(text):\n    return eval(text)\n", tests="def test_a(): pass", requirement="执行表达式")
    # Reflection supports pending rule and makes it model_reviewed; return uncertain to preserve pending.
    from review_agent.llm import ModelResult
    import json
    class Pending(ScriptedModel):
        def invoke(self, system, user, tools=None):
            if "independent-reflection" in system:
                p = json.loads(user)
                return ModelResult(json.dumps({"decisions": [{"finding_id": f["finding_id"], "verdict": "uncertain", "note": "输入信任边界未知"} for f in p["candidates"]]}), [])
            return super().invoke(system, user, tools)
    _, r = run(tmp_path, inputs, Pending(), FakeRunner())
    assert r.status == "needs_attention" and r.manual_items
    assert not r.attempts


def test_multiple_batches_without_intermediate_review(tmp_path):
    source = "\n".join(f"def f{i}(x=[]):\n    return x\n" for i in range(6))
    inputs = Inputs(source=source, tests="def test_a(): pass", requirement="每个函数未传列表时返回独立列表，显式传入时返回原列表。")
    def batch(task):
        patches = []
        links = []
        from review_agent.source import source_lines
        lines = source_lines(task["source"])
        for i, goal in enumerate(task["goals"]):
            # Locate the function anew using evidence and unique function context.
            finding = task["targets"][i]["findings"][0]
            fn = finding["line_start"] // 3
            line = next(j for j, text in enumerate(lines, 1) if text.startswith(f"def f{fn}("))
            original = "\n".join(lines[line-1:line+1])
            patches.append({"line_start": line, "line_end": line+1, "original": original,
                            "replacement": f"def f{fn}(x=None):\n    return [] if x is None else x", "rationale": "默认列表独立"})
            links.append({"patch_index": len(patches)-1, "goal_ids": [goal["goal_id"]]})
        return dict(task_id=task["task_id"], base_source_hash=task["source_hash"], goal_ids=[g["goal_id"] for g in task["goals"]],
                    decision="propose", patches=patches, patch_targets=links, summary="修复默认列表", missing_information=[])
    _, r = run(tmp_path, inputs, ScriptedModel([batch, batch]), FakeRunner(["passed", "passed", "passed"]))
    assert r.status == "verified", r.reason
    assert len(r.attempts) == 2
    assert len(r.attempts[0].task.goals) == 5
    assert len(r.fix_assessment.assessments) == 6
    assert len([e for e in r.events if e.node == "final_review" and e.phase == "start"]) == 1


def test_output_never_overwrites(tmp_path, inputs):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "solution.py").write_text(inputs.source)
    with pytest.raises(FileExistsError):
        Workflow(ScriptedModel(), runner=FakeRunner()).run(inputs, out)
    assert (out / "solution.py").read_text() == inputs.source


def test_global_limit_enforced():
    from review_agent.llm import RecordingModel, ModelBudgetExceeded
    model = RecordingModel(ScriptedModel(), max_calls=25)
    for _ in range(25):
        model.invoke("review", "{}")
    with pytest.raises(ModelBudgetExceeded):
        model.invoke("review", "{}")
    assert len(model.calls) == 25


def test_final_coverage_insufficient_rejects(tmp_path, inputs, monkeypatch):
    from agents.reviewer import Reviewer
    original = Reviewer.review
    count = 0
    def limited(self, code, filename, requirement):
        nonlocal count
        count += 1
        report = original(self, code, filename, requirement)
        if count == 2:
            report.limitations.append("取证覆盖受限")
        return report
    monkeypatch.setattr(Reviewer, "review", limited)
    _, r = run(tmp_path, inputs, ScriptedModel([[patch()]]), FakeRunner(["failed", "passed"]))
    assert r.fix_assessment and not r.accepted_candidate
    assert r.status == "needs_attention" and not r.review_complete


def test_initial_coverage_insufficient_no_fix(tmp_path, inputs, monkeypatch):
    from agents.reviewer import Reviewer
    original = Reviewer.review
    def limited(self, code, filename, requirement):
        report = original(self, code, filename, requirement)
        report.limitations.append("工具不可用")
        return report
    monkeypatch.setattr(Reviewer, "review", limited)
    _, r = run(tmp_path, inputs, ScriptedModel(), FakeRunner())
    assert r.status == "needs_attention" and not r.attempts


def test_initial_pending_preserved_when_omitted_final(tmp_path, inputs, monkeypatch):
    from agents.reviewer import Reviewer
    from review_agent.models import ReviewedFinding
    original = Reviewer.review
    count = 0
    pending = ReviewedFinding(title="待确认接口约定", severity="low", category="maintainability", line_start=1, line_end=1,
                              evidence="def add(a, b):", trigger="约定不明", explanation="约定不明", suggestion="核对约定",
                              finding_id="F-000000000001", origin="model", status="needs_confirmation")
    def review(self, code, filename, requirement):
        nonlocal count
        count += 1
        report = original(self, code, filename, requirement)
        if count == 1:
            report.pending_findings.append(pending)
        return report
    monkeypatch.setattr(Reviewer, "review", review)
    _, r = run(tmp_path, inputs, ScriptedModel([[patch()]]), FakeRunner(["failed", "passed"]))
    assert r.status == "needs_attention" and r.accepted_candidate
    assert pending in r.manual_items
    assert not r.final_review.pending_findings


def test_new_final_confirmed_rejects(tmp_path, inputs, monkeypatch):
    from agents.reviewer import Reviewer
    from review_agent.models import ReviewedFinding
    original = Reviewer.review
    count = 0
    def review(self, code, filename, requirement):
        nonlocal count
        count += 1
        report = original(self, code, filename, requirement)
        if count == 2:
            report.findings.append(ReviewedFinding(title="新确认问题", severity="high", category="bug", line_start=2, line_end=2,
                evidence="return a + b", trigger="新条件", explanation="新机制", suggestion="检查",
                finding_id="F-000000000002", origin="model", status="model_reviewed"))
        return report
    monkeypatch.setattr(Reviewer, "review", review)
    _, r = run(tmp_path, inputs, ScriptedModel([[patch()]]), FakeRunner(["failed", "passed"]))
    assert r.status == "needs_attention" and not r.accepted_candidate
    assert r.outstanding_findings


def test_unhandled_targets_remain_even_when_final_omits(tmp_path, monkeypatch):
    from agents.reviewer import Reviewer
    original_review = Reviewer.review
    count = 0
    def omit_final(self, code, filename, requirement):
        nonlocal count
        count += 1
        report = original_review(self, code, filename, requirement)
        if count == 2:
            report.findings = []
        return report
    monkeypatch.setattr(Reviewer, "review", omit_final)
    source = "\n".join(f"def f{i}(x=[]):\n    return x\n" for i in range(6))
    inputs = Inputs(source=source, tests="def test_a(): pass", requirement="默认列表独立，显式列表不复制。", max_attempts=1)
    def batch(task):
        patches, links = [], []
        lines = task["source"].split("\n")
        for i, target in enumerate(task["targets"]):
            line = target["findings"][0]["line_start"]
            patches.append({"line_start": line, "line_end": line + 1,
                            "original": "\n".join(lines[line-1:line+1]),
                            "replacement": lines[line-1].replace("x=[]", "x=None") + "\n    return [] if x is None else x",
                            "rationale": "默认列表独立"})
            links.append({"patch_index": i, "goal_ids": [target["goal_id"]]})
        return dict(task_id=task["task_id"], base_source_hash=task["source_hash"], goal_ids=[g["goal_id"] for g in task["goals"]],
                    decision="propose", patches=patches, patch_targets=links, summary="修复五个目标", missing_information=[])
    _, report = run(tmp_path, inputs, ScriptedModel([batch]), FakeRunner(["passed", "passed"]))
    assert report.accepted_candidate
    assert report.status == "needs_attention"
    assert len(report.outstanding_findings) == 1
    assert len(report.fix_assessment.assessments) == 5


@pytest.mark.parametrize("preflight_error", [False, True])
def test_default_run_cleans_files(tmp_path, monkeypatch, inputs, preflight_error):
    monkeypatch.chdir(tmp_path)
    previous = tmp_path / "runs" / "existing" / "keep.txt"
    previous.parent.mkdir(parents=True)
    previous.write_text("existing data")
    wf = Workflow(ScriptedModel(), runner=FakeRunner(preflight_error=preflight_error))
    report = wf.run(inputs)
    assert report.status == ("error" if preflight_error else "no_repair_needed")
    assert not wf.directory.exists()
    assert previous.read_text() == "existing data"


def test_default_run_cleans_on_exception(tmp_path, monkeypatch, inputs):
    monkeypatch.chdir(tmp_path)
    wf = Workflow(ScriptedModel(), runner=FakeRunner())
    def fail(*args):
        raise RuntimeError("unexpected failure")
    monkeypatch.setattr(wf.graph, "invoke", fail)
    with pytest.raises(RuntimeError, match="unexpected failure"):
        wf.run(inputs)
    assert not wf.directory.exists()


def test_comment_only_cannot_be_accepted(tmp_path, inputs):
    model = ScriptedModel([[patch(replacement='    """减法。"""\n    return a - b')]])
    wf = Workflow(model, runner=FakeRunner(["failed", "passed"]))
    report = wf.run(inputs, tmp_path / "output")
    assert report.accepted_candidate is None
    assert not (wf.directory / "result" / "accepted_solution.py").exists()
    assert report.attempts[0].proposal.decision == "cannot_fix"
    assert "repair_feedback" in model.calls[-1][1]
