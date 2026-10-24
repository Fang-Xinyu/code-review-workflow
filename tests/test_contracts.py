import pytest
from pydantic import ValidationError
from schemas import Inputs, FixProposal, FixTask, Feedback, digest
from review_agent.source import source_lines
from conftest import ScriptedModel
from agents.fixer import Fixer


@pytest.mark.parametrize("source", [" ", "x='\x00'", "x=1\n" * 2001, "#" + "中" * 35000])
def test_source_limits(source):
    with pytest.raises(ValueError):
        Inputs(mode="review", source=source)


def test_raw_bytes_before_normalization():
    # CRLF bytes exceed the boundary while normalized text does not.
    source = "#" * 101000 + "\r\n" * 900
    with pytest.raises(ValueError):
        Inputs(mode="review", source=source)


def test_lf_unicode_and_syntax():
    inputs = Inputs(mode="review", source="x='a\u2028b'\r\ny=\r\n")
    assert len(source_lines(inputs.source)) == 2
    assert "\r" not in inputs.source
    assert Inputs(mode="review", source="x=1\n" * 2000)


@pytest.mark.parametrize("values", [{"filename": "../solution.py"}, {"tests": "x="}, {"tests": " "},
                                    {"requirement": "x" * 1001}, {"max_attempts": 9},
                                    {"tests": "pytest_plugins = ['evil']"}, {"tests": "def pytest_configure(config): pass"}])
def test_invalid_inputs(inputs, values):
    with pytest.raises(ValueError):
        Inputs(**(inputs.model_dump() | values))


def test_extra_fields_forbidden(inputs):
    with pytest.raises(ValidationError):
        Inputs(**inputs.model_dump(), extra=True)


def test_task_hash_and_goals(task):
    data = task.model_dump()
    for change in ({"source_hash": "0" * 64}, {"requirement_hash": "0" * 64}, {"goals": data["goals"] * 6},
                   {"targets": []}, {"review_source_hash": "0" * 64}):
        with pytest.raises(ValueError):
            FixTask.model_validate(data | change)


def test_proposal_cannot_fix(task):
    proposal = Fixer(ScriptedModel()).propose(task)
    assert proposal.decision == "cannot_fix"
    with pytest.raises(ValueError):
        FixProposal.model_validate(proposal.model_dump() | {"missing_information": []})
    with pytest.raises(ValueError):
        FixProposal.model_validate(proposal.model_dump() | {"decision": "propose"})


def test_feedback_hash_mismatch(task):
    from conftest import test_result
    with pytest.raises(ValueError):
        Feedback(candidate_hash="0" * 64, current_snapshot=task.source_hash, rolled_back=True,
                 test_result=test_result(task.source))
    feedback = Feedback(candidate_hash=task.source_hash, current_snapshot=digest("different"), rolled_back=True)
    with pytest.raises(ValueError):
        FixTask.model_validate(task.model_dump() | {"feedback": [feedback.model_dump()]})
