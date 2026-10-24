import sys

import pytest

from conftest import test_result
from schemas import digest, TestResult
from validation.result_parser import parse_report
from validation.sandbox import Sandbox, SandboxError, bounded_process, REPORT_LIMIT
from validation.test_runner import TestRunner
from workflow.routing import compare_tests


def payload(*, failures=None, phases=None, nodes=None, exit_code=0):
    return {"version": 1, "python": "3.12.8", "pytest": "8.3.5", "complete": True, "truncated": False,
            "collected": ["test_solution.py::test_a"] if nodes is None else nodes,
            "results": {"test_solution.py::test_a": phases or [{"phase": p, "outcome": "passed", "duration_ms": 1} for p in ("setup", "call", "teardown")]},
            "failures": failures or [], "exit_code": exit_code, "duration_ms": 3}


def parse(p, **kwargs):
    return parse_report(p, run_id="run", source_hash=digest("source"), tests_hash=digest("tests"), environment_id="fixture", **kwargs)


def test_machine_report_pass():
    assert parse(payload()).status == "passed"


@pytest.mark.parametrize("change", [{"complete": False}, {"truncated": True}, {"collected": []},
                                    {"results": {}}, {"collected": ["a", "a"]}, {"version": 2},
                                    {"exit_code": 3}])
def test_incomplete_not_success(change):
    assert parse(payload() | change).status != "passed"


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_phase_failure(phase):
    phases = [{"phase": p, "outcome": "failed" if p == phase else "passed", "duration_ms": 0} for p in ("setup", "call", "teardown")]
    failure = {"nodeid": "test_solution.py::test_a", "phase": phase, "exception_type": "AssertionError", "traceback": "assert x == y"}
    r = parse(payload(failures=[failure], phases=phases, exit_code=1))
    assert r.status == ("failed" if phase == "call" else "inconclusive")


@pytest.mark.parametrize("exception,location,kind", [("ValueError", "/input/solution.py", "source_exception"),
    ("SyntaxError", "/input/solution.py", "source_syntax"), ("ModuleNotFoundError", "/input/solution.py", "test_input"),
    ("ValueError", "/input/test_solution.py", "unknown")])
def test_failure_classification(exception, location, kind):
    failure = {"nodeid": "test_solution.py::test_a", "phase": "call", "exception_type": exception,
               "traceback": "machine traceback evidence", "location": location}
    phases = [{"phase": p, "outcome": "failed" if p == "call" else "passed", "duration_ms": 0} for p in ("setup", "call", "teardown")]
    assert parse(payload(failures=[failure], phases=phases, exit_code=1)).failure_kind == kind


@pytest.mark.parametrize("phase,frames,expected", [
    ("call", ["/input/test_solution.py", "/usr/local/lib/python3.12/site-packages/_pytest/outcomes.py"], "assertion"),
    ("setup", ["/input/test_solution.py", "/usr/local/lib/python3.12/site-packages/_pytest/outcomes.py"], "unknown"),
    ("call", ["/input/test_solution.py"], "unknown"),
    ("call", ["/usr/local/lib/python3.12/site-packages/_pytest/outcomes.py"], "unknown"),
])
def test_pytest_failed_outcome_is_classified_from_frames(phase, frames, expected):
    failure = {"nodeid": "test_solution.py::test_a", "phase": phase, "exception_type": "Failed",
               "traceback": "not used for classification", "location": "/input/test_solution.py", "source_frames": frames}
    phases = [{"phase": p, "outcome": "failed" if p == phase else "passed", "duration_ms": 0} for p in ("setup", "call", "teardown")]
    result = parse(payload(failures=[failure], phases=phases, exit_code=1))
    assert result.failure_kind == expected
    assert result.status == ("failed" if expected == "assertion" else "inconclusive")


@pytest.mark.parametrize("outcome,xfail", [("skipped", False), ("skipped", True), ("passed", True)])
def test_skip_xfail_not_pass(outcome, xfail):
    phases = [{"phase": p, "outcome": outcome if p == "call" else "passed", "duration_ms": 0, "wasxfail": xfail and p == "call"}
              for p in ("setup", "call", "teardown")]
    assert parse(payload(phases=phases)).status == "inconclusive"


def test_log_truncation_and_no_report():
    assert parse(payload(), truncated=True).status == "inconclusive"
    assert parse(None).status == "error"
    rc, stdout, stderr, truncated, timeout = bounded_process([sys.executable, "-c", "import sys; print('x'*100000); sys.stderr.write('y'*100000)"])
    assert rc == 0 and len(stdout) == len(stderr) == 65536 and truncated and not timeout


def test_bounded_process_timeout():
    result = bounded_process([sys.executable, "-c", "import time; time.sleep(10)"], timeout=.05)
    assert result[-1]


def test_no_host_fallback(tmp_path):
    sandbox = Sandbox()
    sandbox.docker = None
    runner = TestRunner(sandbox)
    with pytest.raises(SandboxError):
        runner.preflight()
    with pytest.raises(SandboxError):
        runner.run_tests("raise RuntimeError()", "assert False", tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_container_command_boundaries(tmp_path):
    sandbox = Sandbox()
    sandbox.image_id = "sha256:fixed"
    command = sandbox.create_command("task", tmp_path)
    for flag in ("--pull=never", "--network=none", "--read-only", "--user=65532:65532", "--cap-drop=ALL",
                 "--security-opt=no-new-privileges:true", "--cpus=1", "--memory=512m", "--pids-limit=64"):
        assert flag in command
    assert len([s for s in command if s.startswith("type=bind")]) == 1
    assert command[-1] == sandbox.image_id
    assert "--log-opt=compress=false" in command
    assert not any("API_KEY" in s or "docker.sock" in s for s in command)


class StubSandbox:
    image_id = "fixture"
    def __init__(self, behavior="timeout"):
        self.behavior = behavior
        self.cleaned = []
    def create_command(self, name, path):
        return ["create", name]
    def check_container(self, name):
        return {"fixture": True}
    def command(self, *args, **kwargs):
        if args[0] in ("create", "start", "kill"):
            return 0, "", "", False, False
        if args[0] == "inspect":
            return 0, "true", "", False, False
        if args[0] == "logs":
            return 0, "", "", False, False
        if self.behavior == "bad_report":
            return 0, "invalid", "", False, False
        if self.behavior == "large_report":
            assert kwargs["limit"] == REPORT_LIMIT
            return 0, "", "", True, False
        return 3, "", "", False, False
    def cleanup(self, name):
        self.cleaned.append(name)


@pytest.mark.parametrize("behavior", ["timeout", "bad_report", "large_report"])
def test_cleanup_on_failure(tmp_path, behavior):
    sandbox = StubSandbox(behavior)
    runner = TestRunner(sandbox, timeout=.05)
    runner.ready = True
    result = runner.run_tests("x=1", "def test_a(): pass", tmp_path / "exec")
    assert result.status == "error"
    assert len(sandbox.cleaned) == 1


def test_regression_partial_and_reduction():
    baseline = test_result("original", passed=("a",), failed=("b", "c"))
    working = baseline
    improved = test_result("improved", passed=("a", "b"), failed=("c",))
    retained, regression, _, stop = compare_tests(baseline, working, improved)
    assert retained and not regression and not stop
    regressed = test_result("regression", passed=("b", "c"), failed=("a",))
    assert compare_tests(baseline, improved, regressed)[1] == ["a"]
    reduced = test_result("reduced", passed=("a", "b"))
    assert compare_tests(baseline, working, reduced)[3]
    unchanged = test_result("changed-no-progress", passed=("a",), failed=("b", "c"))
    assert not compare_tests(baseline, working, unchanged)[0]


def test_syntax_collection_exception():
    baseline = test_result("broken", passed=(), kind="source_syntax", collected=[])
    candidate = test_result("fixed")
    assert compare_tests(baseline, baseline, candidate)[0]


def test_forged_pass_contract():
    with pytest.raises(ValueError):
        TestResult(run_id="r", source_hash=digest("a"), tests_hash=digest("b"), environment_id="fixture",
                   status="passed", failure_kind="none", complete=True, exit_code=0)


def test_source_syntax_baseline_is_incomplete():
    failure = {"nodeid": "test_solution.py", "phase": "collect", "exception_type": "SyntaxError",
               "traceback": "invalid syntax", "location": "/input/solution.py"}
    result = parse(payload(failures=[failure], nodes=[], exit_code=2) | {"results": {}})
    assert result.status == "failed" and result.failure_kind == "source_syntax"
    assert not result.complete


def test_docker_desktop_detected_without_updated_path(tmp_path, monkeypatch):
    from pathlib import Path
    import validation.sandbox as module
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    executable = tmp_path / ".docker/bin/docker"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    assert module.find_docker() == str(executable)
    assert module.Sandbox().docker == str(executable)
