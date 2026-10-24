import json
from pathlib import Path

import pytest

from review_agent.llm import ModelResult
from schemas import Inputs, TestResult, Phase, Failure, digest, FixTask, Goal, TestTarget
from agents.reviewer import Reviewer


class ScriptedModel:
    """Protocol fixture; never imported by product entry points."""
    def __init__(self, proposals=None, verdict="resolved", draft=None):
        self.proposals = list(proposals or [])
        self.verdict = verdict
        self.draft = draft or []
        self.calls = []

    def invoke(self, system, user, tools=None):
        payload = json.loads(user)
        self.calls.append((system, payload, tools))
        if "independent-reflection" in system:
            content = {"decisions": [{"finding_id": f["finding_id"], "verdict": "supported", "note": "静态证据支持"}
                                     for f in payload["candidates"]]}
        elif "你负责最小局部修复" in system:
            value = self.proposals.pop(0) if self.proposals else "cannot_fix"
            if isinstance(value, Exception):
                raise value
            if isinstance(value, ModelResult):
                return value
            task = payload["task"]
            ids = [g["goal_id"] for g in task["goals"]]
            if value == "cannot_fix":
                content = dict(task_id=task["task_id"], base_source_hash=task["source_hash"], goal_ids=ids,
                               decision="cannot_fix", patches=[], patch_targets=[], summary="必要约定不足", missing_information=["需要业务约定"])
            elif callable(value):
                content = value(task)
            else:
                content = dict(task_id=task["task_id"], base_source_hash=task["source_hash"], goal_ids=ids,
                               decision="propose", patches=value,
                               patch_targets=[{"patch_index": i, "goal_ids": ids} for i in range(len(value))],
                               summary="按需求修复", missing_information=[])
        elif "你负责显式核对" in system:
            context = payload["context"]
            content = {"run_id": context["run_id"], "candidate_hash": payload["candidate"]["source_hash"],
                       "assessments": [{"task_id": t["task_id"], "goal_id": g["goal_id"], "verdict": self.verdict,
                                        "reason": "源码、需求与测试证据相符"} for t in context["tasks"] for g in t["goals"]]}
        else:
            content = {"findings": self.draft}
        return ModelResult(content=json.dumps(content, ensure_ascii=False), tool_calls=[])


def test_result(source, tests="def test_a():\n    assert True\n", passed=("test_solution.py::test_a",),
                failed=(), kind="assertion", status=None, collected=None):
    nodes = list(collected) if collected is not None else list(passed) + list(failed)
    phases = {n: [Phase(phase=p, outcome="failed" if n in failed and p == "call" else "passed", duration_ms=0)
                  for p in ("setup", "call", "teardown")] for n in nodes}
    failures = [Failure(nodeid=n, phase="call", exception_type="AssertionError", traceback="assert result == expected") for n in failed]
    if kind == "source_syntax":
        failures = [Failure(nodeid="test_solution.py", phase="collect", exception_type="SyntaxError",
                            traceback="invalid syntax", location="/input/solution.py")]
    return TestResult(run_id="fixture", source_hash=digest(source), tests_hash=digest(tests), environment_id="fixture",
                      status=status or ("failed" if failed or failures else "passed"), failure_kind=kind if failures else "none",
                      collected=nodes, passed=list(passed), failed=list(failed), results=phases,
                      complete=True, exit_code=1 if failures else 0, failures=failures)


class FakeRunner:
    def __init__(self, outcomes=None, preflight_error=False):
        self.outcomes = list(outcomes or ["passed"])
        self.preflight_error = preflight_error
        self.calls = []

    def preflight(self):
        if self.preflight_error:
            raise RuntimeError("fixture environment unavailable")
        return {"environment": "injected fixture"}

    def run_tests(self, source, tests, directory):
        self.calls.append(source)
        outcome = self.outcomes.pop(0)
        if callable(outcome):
            return outcome(source, tests)
        if outcome == "passed":
            return test_result(source, tests)
        if outcome == "failed":
            return test_result(source, tests, passed=(), failed=("test_solution.py::test_a",))
        return TestResult(run_id="fixture", source_hash=digest(source), tests_hash=digest(tests), environment_id="fixture",
                          status="error", failure_kind=outcome)


@pytest.fixture
def inputs():
    return Inputs(max_attempts=4, source="def add(a, b):\n    return a - b\n", tests="def test_a():\n    assert True\n", requirement="add 返回 a + b。")


@pytest.fixture
def task(inputs):
    report = Reviewer(ScriptedModel()).review(inputs.source, "solution.py", inputs.requirement)
    failure = Failure(nodeid="test_solution.py::test_a", phase="call", exception_type="AssertionError", traceback="assert result == expected")
    return FixTask(task_id="task-01", attempt=1, source=inputs.source, source_hash=digest(inputs.source),
                   requirement=inputs.requirement, requirement_hash=digest(inputs.requirement), tests_hash=digest(inputs.tests),
                   goals=[Goal(goal_id="G1", kind="test_failure", description="修复加法")],
                   targets=[TestTarget(goal_id="G1", source_hash=digest(inputs.source), failures=[failure])],
                   reference_review=report, review_source_hash=report.source_hash)


def patch(line=2, original="    return a - b", replacement="    return a + b"):
    return {"line_start": line, "line_end": line, "original": original, "replacement": replacement, "rationale": "按要求返回两数和"}


ROOT = Path(__file__).resolve().parents[1]

test_result.__test__ = False
