from agents.reviewer import Reviewer
from conftest import FakeRunner, ScriptedModel, patch
from review_agent.models import ReviewedFinding
from workflow.graph import Workflow


def install_review(monkeypatch, persistent=False, manual=False):
    original = Reviewer.review
    calls = []
    def review(self, source, filename, requirement=""):
        report = original(self, source, filename, requirement)
        calls.append(source)
        if len(calls) == 2 or persistent and len(calls) > 1:
            report.findings = [ReviewedFinding(
                finding_id="F-123456789abc", origin="model", status="model_reviewed", severity="low",
                category="bug", title="Further confirmed issue", line_start=2, line_end=2,
                evidence=source.splitlines()[1], trigger="Input condition", explanation="Additional confirmed defect",
                suggestion="Correct the return expression", review_note="Fixture evidence")]
        if manual and len(calls) == 2:
            report.pending_findings = [ReviewedFinding(
                finding_id="F-abcdef123456", origin="model", status="needs_confirmation", severity="low",
                category="boundary", title="Manual policy", line_start=1, line_end=1,
                evidence=source.splitlines()[0], trigger="Unknown policy", explanation="Needs user decision",
                suggestion="Confirm policy", review_note="Not authorized for repair")]
        return report
    monkeypatch.setattr(Reviewer, "review", review)
    return calls


def test_review_findings_return_to_fixer_with_current_snapshot(monkeypatch, tmp_path, inputs):
    calls = install_review(monkeypatch)
    model = ScriptedModel([[patch()], [patch(original="    return a + b", replacement="    return sum((a, b))")]])
    report = Workflow(model, runner=FakeRunner(["failed", "passed", "passed"])).run(inputs, tmp_path / "run")
    assert report.status == "verified", report.reason
    assert len(calls) == 3 and len(report.attempts) == 2
    assert report.attempts[1].task.reference_review.source_hash == report.attempts[0].next_source_hash
    assert report.attempts[1].task.source == calls[1]
    assert all(a.tests.tests_hash == report.tests_hash for a in report.attempts)
    assert (tmp_path / "run/reviews/01.json").exists()
    assert (tmp_path / "run/reviews/02.json").exists()


def test_review_loop_has_three_review_limit(monkeypatch, tmp_path, inputs):
    calls = install_review(monkeypatch, persistent=True)
    model = ScriptedModel([[patch()],
                           [patch(original="    return a + b", replacement="    return sum((a, b))")],
                           [patch(original="    return sum((a, b))", replacement="    return (a + b) + 0")]])
    report = Workflow(model, runner=FakeRunner(["failed", "passed", "passed", "passed"])).run(
        inputs.model_copy(update={"max_attempts": 8}), tmp_path / "run")
    assert len(calls) == 4 and len(report.attempts) == 3
    assert report.partial_candidate and not report.accepted_candidate
    assert "3 次完整复审上限" in report.reason
    assert report.model_call_limit == 49


def test_intermediate_manual_item_survives_later_omission(monkeypatch, tmp_path, inputs):
    install_review(monkeypatch, manual=True)
    model = ScriptedModel([[patch()], [patch(original="    return a + b", replacement="    return sum((a, b))")]])
    report = Workflow(model, runner=FakeRunner(["failed", "passed", "passed"])).run(inputs, tmp_path / "run")
    assert report.accepted_candidate and report.status == "needs_attention"
    assert not report.final_review.pending_findings
    assert [f.title for f in report.manual_items] == ["Manual policy"]


def test_generated_assertion_alone_cannot_authorize_code_change(tmp_path):
    from schemas import Inputs
    from test_optional_tests import GenerationModel
    model = GenerationModel(proposals=[[patch()]])
    report = Workflow(model, runner=FakeRunner(["failed"])).run(
        Inputs(source="def add(a, b):\n    return a + b\n"), tmp_path / "run")
    assert report.status == "needs_attention"
    assert not report.attempts and not report.accepted_candidate
    assert report.final_review and not report.final_review.findings
    assert "生成测试的约定" in report.reason


def test_final_assessment_can_cover_all_eight_rounds():
    from pydantic import ValidationError
    import pytest
    from schemas import FixAssessment, digest
    items = [dict(task_id=f'task-{i // 5}', goal_id=f'goal-{i}', verdict='resolved',
                  reason='依据最终代码核对') for i in range(40)]
    result = FixAssessment(run_id='eight-rounds', candidate_hash=digest('candidate'), assessments=items)
    assert len(result.assessments) == 40
    with pytest.raises(ValidationError):
        FixAssessment(run_id='eight-rounds', candidate_hash=digest('candidate'), assessments=items + [items[0]])
