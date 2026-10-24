import json

import pytest

from agents.fixer import Fixer
from agents.reviewer import Reviewer
from conftest import ScriptedModel, FakeRunner
from review_agent.llm import ModelResult, RecordingModel
from review_agent.models import FindingDraft, ReviewedFinding
from schemas import Inputs, FixTask, Goal, TestTarget, Failure, digest
from validation.patch_checks import validate_patch
from workflow.graph import Workflow

SOURCE = "def divide(a, b):\n    return a / b\n"
REQUIREMENT = "divide(a, b) 在 b 等于 0 时抛出 ValueError；其他情况下返回 a / b。"
CANDIDATE = 'def divide(a, b):\n    if b == 0:\n        raise ValueError("division by zero")\n    return a / b\n'
DRAFT = dict(title="零分母异常不符合需求", severity="medium", category="exception", line_start=1, line_end=2,
             evidence=SOURCE.rstrip("\n"), trigger="divide(1, 0)",
             explanation="需求要求抛出 ValueError，当前抛出 ZeroDivisionError。", suggestion="增加零分母检查", rule_id=None)


def division_patch(candidate=CANDIDATE):
    return {"line_start": 1, "line_end": 2, "original": SOURCE.rstrip("\n"),
            "replacement": candidate.rstrip("\n"), "rationale": "根据明确需求增加零分母检查，其他分支保留正常除法"}


class DivisionModel(ScriptedModel):
    """A regression model that requires the contract to be present at reflection."""
    def __init__(self, force_uncertain=False, proposals=None):
        super().__init__(proposals=proposals if proposals is not None else [[division_patch()]])
        self.force_uncertain = force_uncertain
        self.reflection_contracts = []

    def invoke(self, system, user, tools=None):
        payload = json.loads(user)
        if "independent-reflection" in system:
            self.calls.append((system, payload, tools))
            contract = payload.get("requirement", "")
            self.reflection_contracts.append(contract)
            verdict = "supported" if contract and not self.force_uncertain else "uncertain"
            return ModelResult(json.dumps({"decisions": [{"finding_id": f["finding_id"], "verdict": verdict,
                               "note": "依据收到的固定需求核对；未提供需求时保留不确定性"} for f in payload["candidates"]]}), [])
        if "draft_schema" in payload:
            self.calls.append((system, payload, tools))
            findings = [] if "if b == 0:" in payload.get("source", "") else [DRAFT]
            return ModelResult(json.dumps({"findings": findings}), [])
        return super().invoke(system, user, tools)


def make_division_task():
    review = Reviewer(DivisionModel(force_uncertain=True)).review(SOURCE, "solution.py", REQUIREMENT)
    failure = Failure(nodeid="test_solution.py::test_zero", phase="call", exception_type="ZeroDivisionError",
                      traceback="solution.py:2: ZeroDivisionError: division by zero", location="/input/solution.py")
    return FixTask(task_id="division-01", attempt=1, source=SOURCE, source_hash=digest(SOURCE),
                   requirement=REQUIREMENT, requirement_hash=digest(REQUIREMENT), tests_hash=digest("fixed suite"),
                   goals=[Goal(goal_id="T-zero", kind="test_failure", description="满足明确零分母契约")],
                   targets=[TestTarget(goal_id="T-zero", source_hash=digest(SOURCE), failures=[failure])],
                   reference_review=review, review_source_hash=review.source_hash, pending_findings=review.pending_findings)


def test_full_review_reflection_receives_requirement():
    model = DivisionModel()
    gateway = RecordingModel(model, max_calls=25)
    reviewer = Reviewer(gateway)
    report = reviewer.review(SOURCE, "solution.py", REQUIREMENT)
    assert report.findings and not report.pending_findings
    assert report.findings[0].status == "model_reviewed"
    assert model.reflection_contracts == [REQUIREMENT]
    assert len(gateway.calls) == report.usage.model_calls == 2
    assert all(payload["requirement"] == REQUIREMENT for _, payload, _ in model.calls)
    second = reviewer.review(SOURCE, "solution.py", "")
    assert second.pending_findings and not second.findings
    assert model.reflection_contracts == [REQUIREMENT, ""]


def test_inserted_guard_preserves_pending_original_lines():
    task = make_division_task()
    assert task.pending_findings
    proposal = Fixer(ScriptedModel([[division_patch()]])).propose(task)
    result = validate_patch(task, proposal)
    assert result.status == "passed", result.errors
    assert result.candidate_code == CANDIDATE
    assert task.pending_findings[0].evidence not in result.candidate_code
    assert any("原始行按顺序保留" in check for check in result.target_checks)
    assert task.pending_findings[0].status == "needs_confirmation"


@pytest.mark.parametrize("candidate", [
    "def divide(a, b):\n    return 0\n",
    "def divide(a, b):\n    return a / abs(b)\n",
    "def divide(a, b):\n    if b == 0:\n        return 0\n    return a // b\n",
])
def test_pending_line_rewrite_still_rejected(candidate):
    task = make_division_task()
    proposal = Fixer(ScriptedModel([[division_patch(candidate)]] )).propose(task)
    result = validate_patch(task, proposal)
    assert result.status == "failed"
    assert "待确认事项的原文证据被改写或删除" in result.errors


def test_partial_line_evidence_rewrite_not_allowed(task):
    pending = ReviewedFinding(**FindingDraft.model_validate({**DRAFT, "line_start": 2, "line_end": 2,
                                    "evidence": "return a - b"}).model_dump(), finding_id="F-000000000009",
                              origin="model", status="needs_confirmation")
    task = task.model_copy(update={"pending_findings": [pending]})
    proposal = Fixer(ScriptedModel([[{"line_start": 2, "line_end": 2, "original": "    return a - b",
                                     "replacement": "    return a + b", "rationale": "按需求修复"}]] )).propose(task)
    assert validate_patch(task, proposal).status == "failed"


def test_identical_failed_static_candidate_stops_before_four_rounds(tmp_path):
    inputs = Inputs(source=SOURCE, tests="def test_a(): pass", requirement=REQUIREMENT)
    wrong = division_patch("def divide(a, b):\n    return a // b\n")
    report = Workflow(DivisionModel(force_uncertain=True, proposals=[[wrong]] * 4), runner=FakeRunner(["failed"])).run(inputs, tmp_path / "run")
    assert len(report.attempts) == 2 and "重复候选" in report.reason
    assert not report.accepted_candidate
    assert all(a.tests is None for a in report.attempts)


def test_uncertain_item_remains_manual_after_accepted_insertion(tmp_path):
    inputs = Inputs(source=SOURCE, tests="def test_a(): pass", requirement=REQUIREMENT)
    report = Workflow(DivisionModel(force_uncertain=True), runner=FakeRunner(["failed", "passed"])).run(inputs, tmp_path / "run")
    assert report.accepted_candidate and report.tests_passed
    assert report.status == "needs_attention" and report.manual_items
    assert all(t.kind == "test_failure" for t in report.attempts[0].task.targets)


def test_fragmented_background_evidence_does_not_disable_protection():
    from review_agent.models import PatchEdit
    from validation.patch_checks import pending_evidence_preserved
    evidence = SOURCE.rstrip("\n")
    rewrite = PatchEdit(line_start=4, line_end=4, original="    return a / b", replacement="    return a // b", rationale="修改")
    assert not pending_evidence_preserved(CANDIDATE, CANDIDATE.replace("return a / b", "return a // b"), evidence, [rewrite])
    insertion = PatchEdit(line_start=4, line_end=4, original="    return a / b", replacement="    # preserve normal division\n    return a / b", rationale="插入说明")
    assert pending_evidence_preserved(CANDIDATE, CANDIDATE.replace("    return a / b", insertion.replacement), evidence, [insertion])
