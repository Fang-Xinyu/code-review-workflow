"""Opt-in real Docker integration; model protocol fixture is clearly separate."""
import os
import pytest

from conftest import ROOT, ScriptedModel
from schemas import Inputs
from validation.test_runner import TestRunner
from workflow.graph import Workflow

pytestmark = [pytest.mark.container, pytest.mark.skipif(os.getenv("CODEFLOW_CONTAINER_TESTS") != "1",
                reason="需 CODEFLOW_CONTAINER_TESTS=1 和已准备的 Docker 固定镜像")]


@pytest.fixture
def runner():
    runner = TestRunner()
    runner.preflight()
    return runner


def test_missing_expected_exception_reaches_fixer(runner, tmp_path):
    source = "def checked_path(path):\n    return path\n"
    tests = '''import pytest
from solution import checked_path

def test_safe():
    assert checked_path("report.txt") == "report.txt"

def test_reject_parent():
    with pytest.raises(ValueError):
        checked_path("../secret")
'''
    patch = {"line_start": 2, "line_end": 2, "original": "    return path",
             "replacement": '    if ".." in path.split("/"):\n        raise ValueError("parent path")\n    return path',
             "rationale": "按固定测试拒绝父目录路径，保留普通文件名行为"}
    report = Workflow(ScriptedModel([[patch]]), runner=runner).run(Inputs(source=source, tests=tests), tmp_path / "run")
    assert report.baseline_test.status == "failed" and report.baseline_test.failure_kind == "assertion"
    assert any(f.exception_type == "Failed" for f in report.baseline_test.failures)
    assert report.status == "verified", report.reason
    assert len(report.attempts) == 1 and report.accepted_candidate
    assert report.tests_passed and report.goals_verified


def test_probe_and_sample_baselines(runner, tmp_path):
    assert runner.environment["probe"] == "passed"
    for name, expected in (("division", "failed"), ("mutable_default", "failed"), ("clean", "passed")):
        directory = ROOT / "examples" / name
        result = runner.run_tests((directory / "solution.py").read_text(),
                                  (directory / "test_solution.py").read_text(), tmp_path / name)
        assert result.status == expected
        assert result.complete and result.collected


@pytest.mark.parametrize("name", ["division", "mutable_default", "clean"])
def test_end_to_end_with_real_isolation(runner, tmp_path, name):
    directory = ROOT / "examples" / name
    inputs = Inputs(source=(directory / "solution.py").read_text(), tests=(directory / "test_solution.py").read_text(),
                    requirement=(directory / "requirement.md").read_text())
    proposals = []
    if name == "division":
        proposals = [[{"line_start": 2, "line_end": 2, "original": "    return a / b",
                       "replacement": "    if b == 0:\n        raise ValueError('zero denominator')\n    return a / b", "rationale": "零除数契约"}]]
    elif name == "mutable_default":
        proposals = [[{"line_start": 1, "line_end": 1, "original": "def append_item(item, items=[]):",
                       "replacement": "def append_item(item, items=None):\n    if items is None:\n        items = []", "rationale": "保留显式列表原地追加"}]]
    report = Workflow(ScriptedModel(proposals), runner=runner).run(inputs, tmp_path / "run")
    assert report.status == ("no_repair_needed" if name == "clean" else "verified"), report.reason
    assert report.tests_passed
    if name != "clean":
        assert report.accepted_candidate and report.fix_assessment


def test_timeout_cleanup_and_no_key(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "never-forward-this-key")
    test = "import os\nimport time\ndef test_wait():\n    assert 'DEEPSEEK_API_KEY' not in os.environ\n    time.sleep(30)\n"
    runner.timeout = 2
    result = runner.run_tests("x = 1", test, tmp_path / "timeout")
    assert result.failure_kind == "timeout"
    rc, output, *_ = runner.sandbox.command("ps", "-a", "--filter", f"name={result.run_id}", "--format={{.Names}}")
    assert rc == 0 and not output.strip()


def test_syntax_and_skip(runner, tmp_path):
    syntax = runner.run_tests("def f(:\n", "from solution import f\ndef test_a(): pass\n", tmp_path / "syntax")
    assert syntax.status == "failed" and syntax.failure_kind == "source_syntax"
    skipped = runner.run_tests("x=1", "import pytest\n@pytest.mark.skip\ndef test_a(): pass\n", tmp_path / "skip")
    assert skipped.status == "inconclusive"


@pytest.mark.parametrize("force_uncertain", [False, True])
def test_division_with_requirement_and_overlapping_pending_evidence(runner, tmp_path, force_uncertain):
    from test_pending_insertions import DivisionModel
    directory = ROOT / "examples/division"
    inputs = Inputs(source=(directory / "solution.py").read_text(), tests=(directory / "test_solution.py").read_text(),
                    requirement=(directory / "requirement.md").read_text())
    report = Workflow(DivisionModel(force_uncertain=force_uncertain), runner=runner).run(inputs, tmp_path / "run")
    assert report.baseline_test.status == "failed"
    assert len(report.attempts) == 1
    assert report.attempts[0].static.status == "passed"
    assert report.attempts[0].tests.status == "passed"
    assert len(report.attempts[0].tests.passed) == 3
    assert report.accepted_candidate and report.goals_verified
    assert report.status == ("needs_attention" if force_uncertain else "verified"), report.reason
    assert bool(report.manual_items) == force_uncertain
