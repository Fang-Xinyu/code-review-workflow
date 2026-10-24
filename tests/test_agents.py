import json
import pytest

from agents.fixer import Fixer
from agents.reviewer import Reviewer
from review_agent.llm import ModelResult, RecordingModel, ModelBudgetExceeded
from conftest import ScriptedModel, patch


def test_direct_proposal(task):
    fixer = Fixer(ScriptedModel([[patch()]]))
    assert fixer.propose(task).patches[0].replacement.endswith("a + b")
    assert len(fixer.recording.calls) == 1


def test_tools_then_submit(task):
    result = ModelResult(content="", tool_calls=[{"name": "read_source_range", "args": {"start_line": 1, "end_line": 2}, "id": "t"}])
    model = ScriptedModel([result, [patch()]])
    fixer = Fixer(model)
    fixer.propose(task)
    assert len(fixer.recording.calls) == 2
    assert fixer.observations[0]["ok"]
    assert model.calls[-1][2] is None


def test_format_repair(task):
    fixer = Fixer(ScriptedModel([ModelResult("invalid json", []), [patch()]]))
    fixer.propose(task)
    assert len(fixer.recording.calls) == 2


@pytest.mark.parametrize("count", [1, 6])
def test_tools_limit(task, count):
    result = ModelResult("", [{"name": "read_source_range", "args": {"start_line": 1, "end_line": 2}}] * count)
    fixer = Fixer(ScriptedModel([result, result]))
    with pytest.raises(ValueError):
        fixer.propose(task)
    assert len(fixer.recording.calls) <= 2
    assert len(fixer.observations) <= 5


def test_failure_counts_and_global_gateway(task):
    global_model = RecordingModel(ScriptedModel([RuntimeError("secret provider body")]), max_calls=1)
    fixer = Fixer(global_model)
    with pytest.raises(RuntimeError):
        fixer.propose(task)
    assert global_model.calls[0].failed
    assert fixer.recording.calls[0].failed
    with pytest.raises(ModelBudgetExceeded):
        global_model.invoke("", "")
    assert len(global_model.calls) == 1


def test_real_review_backfills_and_resets():
    reviewer = Reviewer(ScriptedModel())
    code = "def append(x, items=[]):\n    items.append(x)\n    return items\n"
    r = reviewer.review(code, "solution.py", "默认列表独立，显式列表原地追加")
    assert any(f.rule_id == "AST101" and f.status == "static_confirmed" for f in r.findings)
    assert any(v.action == "backfilled" for v in r.validation_records)
    assert r.usage.model_calls <= 8
    clean = reviewer.review("x = 1\n", "solution.py", "")
    assert clean.usage.model_calls == 1
    assert not clean.findings


def test_assessment_is_single_call(task):
    from schemas import Candidate, StaticResult
    from conftest import test_result
    code = task.source
    candidate = Candidate(code=code, source_hash=task.source_hash, diff="")
    model = ScriptedModel()
    reviewer = Reviewer(model)
    context = {"run_id": "run", "tasks": [task.model_dump()]}
    result = reviewer.assess_fix(context, candidate, StaticResult(status="passed", base_source_hash=task.source_hash),
                                 test_result(code), task.reference_review)
    assert len(result.assessments) == 1
    assert len(reviewer.assessment_model.calls) == 1
    assert model.calls[0][2] is None
    assert model.calls[0][1]["required_assessment_ids"] == [
        {"task_id": task.task_id, "goal_id": g.goal_id} for g in task.goals]


def test_assessment_invalid_no_retry(task):
    from schemas import Candidate, StaticResult
    from conftest import test_result
    class Bad:
        def invoke(self, *args):
            return ModelResult(json.dumps({"run_id": "bad", "candidate_hash": task.source_hash, "assessments": []}), [])
    reviewer = Reviewer(Bad())
    with pytest.raises(ValueError):
        reviewer.assess_fix({"run_id": "run", "tasks": [task.model_dump()]}, Candidate(code=task.source, source_hash=task.source_hash, diff=""),
                            StaticResult(status="passed", base_source_hash=task.source_hash), test_result(task.source), task.reference_review)
    assert len(reviewer.assessment_model.calls) == 1


def test_http_attempts_distinct_from_logical_calls():
    from review_agent.models import ModelAttemptStats
    class Retried:
        attempts = 0
        request_stats = []
        def invoke(self, *args):
            self.attempts += 3
            self.request_stats = [ModelAttemptStats(elapsed_ms=1, failed=True),
                                  ModelAttemptStats(elapsed_ms=1, failed=True),
                                  ModelAttemptStats(elapsed_ms=1, input_tokens=8, output_tokens=2)]
            return ModelResult('{"findings":[]}', [], input_tokens=8, output_tokens=2)
    model = RecordingModel(Retried())
    model.invoke("review", "{}")
    assert model.totals() == {"model_calls": 1, "http_attempts": 3, "input_tokens": 8, "output_tokens": 2}
    assert len(model.calls[0].attempts) == 3


def test_fixer_corrects_documentation_only_with_fixed_test_evidence():
    from schemas import FixTask, Goal, TestTarget, Failure, digest
    from conftest import ROOT
    from validation.patch_checks import validate_patch
    source = (ROOT / "examples/division/solution.py").read_text()
    suite = (ROOT / "examples/division/test_solution.py").read_text()
    review = Reviewer(ScriptedModel()).review(source, "solution.py")
    failure = Failure(nodeid="test_solution.py::test_zero", phase="call",
                      exception_type="ZeroDivisionError", traceback="division by zero")
    task = FixTask(task_id="division", attempt=1, source=source, source_hash=digest(source),
                   requirement_hash=digest(""), tests_hash=digest(suite), tests=suite, tests_origin="user",
                   goals=[Goal(goal_id="zero", kind="test_failure", description="满足零分母测试")],
                   targets=[TestTarget(goal_id="zero", source_hash=digest(source), failures=[failure])],
                   reference_review=review, review_source_hash=review.source_hash)
    documentation = patch(original="    return a / b", replacement='    """零分母会抛 ZeroDivisionError。"""\n    return a / b')
    repair = patch(original="    return a / b", replacement='    if b == 0:\n        raise ValueError("分母不能为零")\n    return a / b')
    model = ScriptedModel([[documentation], [repair]])
    fixer = Fixer(model)
    proposal = fixer.propose(task)
    assert len(fixer.recording.calls) == 2
    first, second = [payload for _, payload, _ in model.calls]
    assert first["task"]["tests"] == suite
    assert first["task"]["tests_origin"] == "user"
    assert "pytest.raises(ValueError)" in first["task"]["tests"]
    assert "repair_feedback" in second and "rejected_proposal" in second
    assert model.calls[1][2] is None
    assert "raise ValueError" in proposal.patches[0].replacement
    assert validate_patch(task, proposal).status == "passed"
    with pytest.raises(ValueError, match="固定测试内容与哈希不一致"):
        FixTask.model_validate(task.model_dump() | {"tests": "def test_other(): pass"})


def test_fixer_repeated_comment_only_stops_within_budget(task):
    from agents.fixer import FixProtocolError
    docs = patch(replacement="    return a - b  # 只说明原来的减法")
    fixer = Fixer(ScriptedModel([[docs], [docs]]))
    with pytest.raises(FixProtocolError, match="未修改可执行代码"):
        fixer.propose(task)
    assert len(fixer.recording.calls) == 2
