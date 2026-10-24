"""Code-only entry points retain the original complete review and its evidence."""
import json
from pathlib import Path

import pytest

from agents.reviewer import Reviewer
from conftest import FakeRunner, ScriptedModel, ROOT
from review_agent.graph import ReviewAgent
from review_agent.llm import ModelResult
from schemas import Inputs, digest
from test_optional_tests import GenerationModel
from workflow.graph import Workflow


@pytest.mark.parametrize("code,rule", [
    ("def add_item(x, items=[]):\n    items.append(x)\n    return items\n", "AST101"),
    ("def f():\n    try:\n        return 1\n    finally:\n        return 2\n", "AST105"),
    ("def f():\n    return 1 / 0\n", "AST104"),
    ("def f(:\n    return 1\n", "AST001"),
])
def test_code_only_review_identical_to_original_core(code, rule):
    direct_model, wrapped_model = ScriptedModel(), ScriptedModel()
    direct = ReviewAgent(direct_model).review(code, filename="solution.py")
    wrapped = Reviewer(wrapped_model).review(code, "solution.py")
    # Exact prompts, payloads, tools and decisions; no injected requirement or instruction.
    assert direct_model.calls == wrapped_model.calls
    assert direct.findings == wrapped.findings
    assert direct.pending_findings == wrapped.pending_findings
    assert direct.validation_records == wrapped.validation_records
    assert any(f.rule_id == rule for f in wrapped.findings)
    assert direct.focus == wrapped.focus == "全面检查正确性、异常处理、资源和安全"


def test_code_only_generated_tests_receive_confirmed_review(tmp_path):
    code = (ROOT / "examples/mutable_default/solution.py").read_text()
    suite = (ROOT / "examples/mutable_default/test_solution.py").read_text()
    model = GenerationModel({"tests": suite, "rationale": "根据已确认的默认对象共享问题生成回归检查"})
    report = Workflow(model, runner=FakeRunner(["failed"])).run(Inputs(source=code), tmp_path / "run")
    payload = next(p for _, p, _ in model.calls if "original_source" in p)
    assert payload["requirement"] == ""
    assert any(f["rule_id"] == "AST101" for f in payload["confirmed_findings"])
    assert all(f["status"] != "needs_confirmation" for f in payload["confirmed_findings"])
    assert report.initial_review.findings
    assert report.attempts[0].task.requirement == ""
    assert report.attempts[0].task.requirement_hash == digest("")


def test_generation_failure_keeps_code_only_review(tmp_path):
    model = GenerationModel(ValueError("invalid generated suite"))
    report = Workflow(model, runner=FakeRunner()).run(Inputs(source="def f(x, items=[]):\n    return items\n"), tmp_path / "run")
    assert report.status == "error"
    assert report.initial_review and report.initial_review.findings
    assert not report.attempts and not report.accepted_candidate


def test_code_only_cli(monkeypatch, tmp_path, capsys):
    import cli
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key")
    monkeypatch.setattr(cli, "DeepSeekModel", lambda config: GenerationModel())
    monkeypatch.setattr(cli, "Workflow", lambda model, name, progress_callback: Workflow(model, name, runner=FakeRunner(), progress_callback=progress_callback))
    assert cli.main(["run", "--source", str(ROOT / "examples/clean/solution.py"),
                     "--output-dir", str(tmp_path / "run")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert json.loads(Path(output["report"]).read_text())["requirement"] == ""


class CoreOnlyModel(ScriptedModel):
    """No extra contract is assumed; same review protocol as the supplied core."""
    def invoke(self, system, user, tools=None):
        payload = json.loads(user)
        assert "requirement" not in payload or not payload["requirement"]
        if "你负责在修复前" in system:
            self.calls.append((system, payload, tools))
            return ModelResult(json.dumps({"tests": (ROOT / "examples/mutable_default/test_solution.py").read_text(),
                                           "rationale": "固定检查默认对象独立及显式对象身份"}), [])
        return super().invoke(system, user, tools)


@pytest.mark.container
def test_code_only_real_container_full_workflow(tmp_path):
    import os
    from validation.test_runner import TestRunner
    if os.getenv("CODEFLOW_CONTAINER_TESTS") != "1":
        pytest.skip("显式启用真实容器测试")
    code = (ROOT / "examples/mutable_default/solution.py").read_text()
    replacement = "def append_item(item, items=None):\n    if items is None:\n        items = []"
    model = CoreOnlyModel(proposals=[[{"line_start": 1, "line_end": 1,
                                      "original": code.split("\n")[0], "replacement": replacement,
                                      "rationale": "避免默认调用共享状态，保留显式传入列表行为"}]])
    report = Workflow(model, runner=TestRunner()).run(Inputs(source=code), tmp_path / "run")
    assert report.status == "verified", report.reason
    assert report.baseline_test.failed and report.tests_passed and report.goals_verified
    assert report.accepted_candidate and len(report.attempts) == 1
    assert report.requirement == "" and report.tests_origin == "generated"
