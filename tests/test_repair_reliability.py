import pytest

from agents.fixer import Fixer, locate_patches
from conftest import ScriptedModel, patch
from schemas import GeneratedTests, FixTask, digest
from agents.reviewer import Reviewer
from workflow.routing import review_goals
from validation.patch_checks import validate_patch
from validation.patch_checks import compare_diagnostics
from review_agent.ast_rules import analyze_ast
from review_agent.models import PatchEdit


def test_stale_line_number_uses_unique_current_original(task):
    proposal = Fixer(ScriptedModel([[patch(line=1)]])).propose(task)
    assert proposal.patches[0].line_start == 2
    assert validate_patch(task, proposal).status == "passed"


def test_ambiguous_original_never_relocated(task):
    proposal = Fixer(ScriptedModel([[patch()]])).propose(task)
    proposal.patches[0].line_start = proposal.patches[0].line_end = 1
    locate_patches("def f():\n    return a - b\ndef g():\n    return a - b\n", proposal)
    assert proposal.patches[0].line_start == 1


def test_unicode_string_separator_does_not_change_line_numbers(task):
    proposal = Fixer(ScriptedModel([[patch()]])).propose(task)
    locate_patches("label = 'a\u2028b'\ndef f():\n    return a - b\n", proposal)
    assert proposal.patches[0].line_start == 3


@pytest.mark.parametrize("code", [
    "import pytest\ndef test_a(): pytest.skip('unknown')",
    "from pytest import skip as omit\ndef test_a(): omit('unknown')",
    "import pytest\n@pytest.mark.xfail\ndef test_a(): assert False",
])
def test_generated_suite_cannot_skip_checks(code):
    with pytest.raises(ValueError, match="不能包含跳过"):
        GeneratedTests(tests=code, rationale="test")


@pytest.mark.parametrize("indent", [False, True])
def test_multiline_static_target_after_previous_insertion(indent):
    original = "def load():\n    try:\n        return int('bad')\n    except:\n        pass\n"
    review = Reviewer(ScriptedModel()).review(original, "solution.py")
    goals, targets = review_goals(review)
    pairs = [(g, t) for g, t in zip(goals, targets) if t.findings[0].rule_id == "AST102"]
    assert len(pairs) == 1
    current = "# inserted by an earlier accepted patch\n" + original
    if indent:
        current = "def load():\n    if True:\n" + "\n".join("    " + line for line in original.splitlines()[1:]) + "\n"
    task = FixTask(task_id="multiline", attempt=2, source=current, source_hash=digest(current),
                   requirement_hash=digest(""), tests_hash=digest(""), goals=[pairs[0][0]], targets=[pairs[0][1]],
                   reference_review=review, review_source_hash=review.source_hash)
    prefix = "    " if indent else ""
    proposal = Fixer(ScriptedModel([[dict(line_start=5, line_end=6, original=prefix + "    except:\n" + prefix + "        pass",
                                        replacement=prefix + "    except ValueError:\n" + prefix + "        raise", rationale="Preserve propagation")]])).propose(task)
    result = validate_patch(task, proposal)
    assert result.status == "passed", result.errors


def test_existing_indented_diagnostics_are_not_new_but_duplicates_are():
    before = "def f():\n    try:\n        return int('x')\n    except:\n        pass\n"
    after = "def f():\n    if True:\n        try:\n            return int('x')\n        except:\n            pass\n"
    edit = PatchEdit(line_start=1, line_end=5, original=before.rstrip(), replacement=after.rstrip(), rationale="Indent")
    old, _ = analyze_ast(before)
    current, _ = analyze_ast(after)
    assert not compare_diagnostics(old, current, [edit])[0]
    duplicated, _ = analyze_ast(after + before.replace("f()", "g()"))
    assert compare_diagnostics(old, duplicated, [edit])[0]


def test_unchanged_context_trimmed_without_changing_replacement(task):
    edit = dict(line_start=1, line_end=2, original=task.source.rstrip(),
                replacement="def add(a, b):\n    return a + b", rationale="Fix")
    proposal = Fixer(ScriptedModel([[edit]])).propose(task)
    assert proposal.patches[0].line_start == proposal.patches[0].line_end == 2
    result = validate_patch(task, proposal)
    assert result.status == "passed"
    assert result.candidate_code == "def add(a, b):\n    return a + b\n"


def test_failed_final_review_not_retried_in_finish(monkeypatch, tmp_path, inputs):
    from conftest import FakeRunner
    from workflow.graph import Workflow
    original_review = Reviewer.review
    calls = []
    def review(self, *args, **kwargs):
        calls.append(True)
        if len(calls) > 1:
            raise RuntimeError("private-provider-body")
        return original_review(self, *args, **kwargs)
    monkeypatch.setattr(Reviewer, "review", review)
    report = Workflow(ScriptedModel([[patch()]]), runner=FakeRunner(["failed", "passed"])).run(inputs, tmp_path / "run")
    assert len(calls) == 2
    assert report.partial_candidate and not report.accepted_candidate
    assert "private-provider-body" not in report.model_dump_json()


def test_missing_display_summary_does_not_discard_valid_patch(task):
    def response(data):
        return dict(task_id=data["task_id"], base_source_hash=data["source_hash"], goal_ids=["G1"],
                    decision="propose", patches=[patch()], patch_targets=[dict(patch_index=0, goal_ids=["G1"])])
    proposal = Fixer(ScriptedModel([response])).propose(task)
    assert proposal.summary.startswith("模型未提供")
    assert validate_patch(task, proposal).status == "passed"


def test_stale_snapshot_identity_repaired_within_existing_call_budget(task):
    def stale(data):
        return dict(task_id=data["task_id"], base_source_hash="0" * 64, goal_ids=["G1"],
                    decision="propose", patches=[patch()], patch_targets=[dict(patch_index=0, goal_ids=["G1"])])
    fixer = Fixer(ScriptedModel([stale, [patch()]]))
    proposal = fixer.propose(task)
    assert proposal.base_source_hash == task.source_hash
    assert len(fixer.recording.calls) == 2
