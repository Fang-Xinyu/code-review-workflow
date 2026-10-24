import json
import os
import shutil
import subprocess
import sys

from conftest import ROOT, ScriptedModel, FakeRunner
from workflow.graph import Workflow


def test_cli_exit_exports(monkeypatch, tmp_path, capsys):
    import cli
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-private-key")
    monkeypatch.setattr(cli, "DeepSeekModel", lambda config: ScriptedModel())
    def factory(model, name, progress_callback):
        return Workflow(model, name, runner=FakeRunner(), progress_callback=progress_callback)
    monkeypatch.setattr(cli, "Workflow", factory)
    result = cli.main(["run", "--source", str(ROOT / "examples/clean/solution.py"),
                       "--tests", str(ROOT / "examples/clean/test_solution.py"),
                       "--requirement", str(ROOT / "examples/clean/requirement.md"), "--output-dir", str(tmp_path / "run")])
    assert result == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "no_repair_needed"
    assert "initial_review" in output.err
    assert "fixture-private-key" not in (tmp_path / "run/result/report.json").read_text()
    assert cli.main(["review", "--source", str(ROOT / "examples/clean/solution.py"), "--output-dir", str(tmp_path / "run")]) == 2


def test_cli_doctor_no_model(monkeypatch, capsys):
    import cli
    monkeypatch.setattr(cli, "DeepSeekModel", lambda config: (_ for _ in ()).throw(AssertionError("model called")))
    monkeypatch.setattr(cli.TestRunner, "preflight", lambda self: (_ for _ in ()).throw(RuntimeError("no docker")))
    assert cli.main(["doctor"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_cli_missing_key(monkeypatch, capsys):
    import cli
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert cli.main(["review", "--source", str(ROOT / "examples/clean/solution.py")]) == 2
    assert "API_KEY" in capsys.readouterr().out


def test_independent_copy(tmp_path):
    target = tmp_path / "standalone"
    shutil.copytree(ROOT, target, ignore=shutil.ignore_patterns(".venv", "runs", "outputs", "__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info"))
    script = """import pathlib,sys
from workflow.graph import Workflow
from review_agent.graph import ReviewAgent
from schemas import Inputs
assert pathlib.Path(sys.modules['review_agent'].__file__).resolve().is_relative_to(pathlib.Path.cwd())
assert pathlib.Path(sys.modules['workflow'].__file__).resolve().is_relative_to(pathlib.Path.cwd())
assert Inputs(mode='review',source='x=1')
print('independent imports passed')
"""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "DEEPSEEK_API_KEY")}
    result = subprocess.run([sys.executable, "-c", script], cwd=target, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_contracts.py", "tests/test_agents.py", "-q"],
                            cwd=target, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_web_disable_and_examples(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    assert not at.exception
    assert next(b for b in at.button if b.label == "开始任务").disabled
    at.selectbox[0].select("division").run()
    assert "def divide" in at.text_area[0].value
    assert "pytest.raises" in next(t for t in at.text_area if t.key == "tests").value
    assert at.text_input[2].value == "deepseek-flash"


def test_web_stale_and_failure_reset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from streamlit.testing.v1 import AppTest
    import review_agent.llm
    import workflow.graph
    monkeypatch.setattr(review_agent.llm, "DeepSeekModel", lambda config: ScriptedModel())
    class InjectedWorkflow(Workflow):
        def __init__(self, model, model_name, progress_callback=None):
            super().__init__(model, model_name, runner=FakeRunner(), progress_callback=progress_callback)
    monkeypatch.setattr(workflow.graph, "Workflow", InjectedWorkflow)
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    at.text_input[0].set_value("fixture-key").run()
    at.selectbox[0].select("clean").run()
    next(b for b in at.button if b.label == "开始任务").click().run(timeout=15)
    at.session_state["thread"].join(timeout=10)
    at.run(timeout=15)
    assert at.session_state["result"].status == "no_repair_needed"
    at.text_area[0].set_value("x = 2").run()
    assert any("过期" in w.value for w in at.warning)
    next(t for t in at.text_area if t.key == "tests").set_value("x=").run()
    next(b for b in at.button if b.label == "开始任务").click().run()
    assert at.session_state["result"] is None
    assert at.error


def test_symlink_output_rejected(tmp_path, inputs):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    import pytest
    with pytest.raises(ValueError):
        Workflow(ScriptedModel(), runner=FakeRunner()).run(inputs, link / "run")


def test_web_environment_failure_keeps_automatic_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from streamlit.testing.v1 import AppTest
    import review_agent.llm
    import workflow.graph
    monkeypatch.setattr(review_agent.llm, "DeepSeekModel", lambda config: ScriptedModel())
    class UnavailableWorkflow(Workflow):
        def __init__(self, model, model_name, progress_callback=None):
            super().__init__(model, model_name, runner=FakeRunner(preflight_error=True), progress_callback=progress_callback)
    monkeypatch.setattr(workflow.graph, "Workflow", UnavailableWorkflow)
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    at.selectbox[0].select("division").run()
    at.radio[1].set_value("让系统自动生成检查用例").run()
    at.text_input[0].set_value("fixture-key").run()
    next(b for b in at.button if b.label == "开始任务").click().run(timeout=15)
    at.session_state["thread"].join(timeout=10)
    at.run(timeout=15)
    result = at.session_state["result"]
    assert result.mode == "run" and result.status == "error"
    assert not result.model_calls and not result.baseline_test
    assert any("这次仍是自动修复模式" in i.value for i in at.info)
    assert not any("仅审查未执行测试" in i.value for i in at.info)
    assert any("尚未生成检查用例" in w.value for w in at.warning)
    assert all("preflight" not in table.value.to_string() for table in at.dataframe)


def test_web_keeps_current_run_until_next_task(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from streamlit.testing.v1 import AppTest
    import review_agent.llm
    import workflow.graph
    monkeypatch.setattr(review_agent.llm, "DeepSeekModel", lambda config: ScriptedModel())
    class InjectedWorkflow(Workflow):
        def __init__(self, model, model_name, progress_callback=None):
            super().__init__(model, model_name, runner=FakeRunner(), progress_callback=progress_callback)
    monkeypatch.setattr(workflow.graph, "Workflow", InjectedWorkflow)
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    at.text_input[0].set_value("fixture-key").run()
    at.selectbox[0].select("clean").run()
    def start():
        next(b for b in at.button if b.label == "开始任务").click().run(timeout=15)
        at.session_state["thread"].join(timeout=10)
        at.run(timeout=15)
        assert not at.exception
        return tmp_path / "runs" / at.session_state["result"].run_id
    first = start()
    assert (first / "result" / "report.json").exists()
    at.run()
    assert first.exists()
    second = start()
    assert not first.exists()
    assert (second / "result" / "report.json").exists()
