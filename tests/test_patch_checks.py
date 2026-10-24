import pytest
from schemas import FixProposal, FixTask, ReviewTarget, Goal, digest
from agents.fixer import Fixer
from agents.reviewer import Reviewer
from conftest import ScriptedModel, patch
from validation.patch_checks import validate_patch, compare_diagnostics, mapped_lines
from review_agent.ast_rules import analyze_ast


@pytest.mark.parametrize("changes", [{"task_id": "other"}, {"base_source_hash": "0" * 64}, {"goal_ids": ["unknown"]}])
def test_bound_proposal(task, changes):
    p = Fixer(ScriptedModel([[patch()]])).propose(task)
    # Construct invalid transport fields to verify validation boundary.
    p = p.model_copy(update=changes)
    assert validate_patch(task, p).status == "failed"


@pytest.mark.parametrize("patches", [[patch(original="wrong")], [patch(), patch()], [patch(replacement="    return (")],
                                    [patch(replacement="    return a - b")], [patch(replacement="    return missing")],
                                    [patch(replacement="    return a / 0")], [patch(replacement="    return a + b  # noqa")],
                                    [patch(replacement="    return a + b\n" + "#" * 103000)]])
def test_reject_invalid_and_new_diagnostics(task, patches):
    p = Fixer(ScriptedModel([patches])).propose(task)
    result = validate_patch(task, p)
    assert result.status == "failed"
    assert result.errors


def test_good_patch(task):
    p = Fixer(ScriptedModel([[patch()]])).propose(task)
    result = validate_patch(task, p)
    assert result.status == "passed"
    assert result.candidate_hash == digest(result.candidate_code)
    assert "-    return a - b" in result.diff


def test_ten_patch_limit(task):
    p = Fixer(ScriptedModel([[patch()]])).propose(task)
    with pytest.raises(ValueError):
        FixProposal.model_validate(p.model_dump() | {"patches": [patch()] * 11})


def test_associations(task):
    p = Fixer(ScriptedModel([[patch()]])).propose(task)
    for change in ({"patch_targets": []}, {"patch_targets": [{"patch_index": 0, "goal_ids": ["bad"]}]}):
        with pytest.raises(ValueError):
            FixProposal.model_validate(p.model_dump() | change)


def test_multi_diagnostic_counts():
    before, _ = analyze_ast("def f(x=[], y=[]):\n    return x, y\n")
    assert len(before) == 2
    new, matches = compare_diagnostics(before[:1], before, [])
    assert len(matches) == 1 and len(new) == 1
    assert before[0].column_start != before[1].column_start


def test_line_mapping(task):
    p = Fixer(ScriptedModel([[patch(replacement="    x = a + b\n    return x")]])).propose(task)
    assert list(mapped_lines(2, p.patches)) == [2, 3]
    assert list(mapped_lines(3, p.patches)) == [4]


def test_static_target_persists(task):
    source = "def f(x=[]):\n    return x\n"
    r = Reviewer(ScriptedModel()).review(source, "solution.py", "默认列表独立")
    target = r.findings[0]
    data = task.model_dump() | {"source": source, "source_hash": digest(source),
        "goals": [Goal(goal_id="G", kind="review_issue", description="默认列表独立").model_dump()],
        "targets": [ReviewTarget(goal_id="G", source_hash=digest(source), findings=[target]).model_dump()]}
    t = FixTask.model_validate(data)
    proposal = Fixer(ScriptedModel([[patch(line=2, original="    return x", replacement="    return x[:]")]])).propose(t)
    assert validate_patch(t, proposal).status == "failed"


def test_pending_evidence_guard(task):
    from review_agent.models import ReviewedFinding
    pending = ReviewedFinding(title="待确认", severity="low", category="bug", line_start=2, line_end=2,
                              evidence="return a - b", trigger="未知需求", explanation="未知需求", suggestion="核对", finding_id="F-000000000000",
                              origin="model", status="needs_confirmation")
    t = task.model_copy(update={"pending_findings": [pending]})
    proposal = Fixer(ScriptedModel([[patch()]])).propose(t)
    assert "待确认" in ";".join(validate_patch(t, proposal).errors)
    with pytest.raises(ValueError):
        ReviewTarget(goal_id="G", source_hash=t.source_hash, findings=[pending])


@pytest.mark.parametrize("replacement", [
    "    return a - b  # 已修复",
    '    """说明原有行为。"""\n    return a - b',
    "    return (a - b)",
])
def test_documentation_only_is_not_a_fix(task, replacement):
    proposal = Fixer(ScriptedModel([[patch()]])).propose(task)
    proposal.patches[0].replacement = replacement
    result = validate_patch(task, proposal)
    assert result.status == "failed"
    assert any("未修改可执行代码" in error for error in result.errors)
