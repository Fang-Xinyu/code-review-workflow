import json
import pytest
from agents.reviewer import Reviewer
from conftest import ScriptedModel, FakeRunner, patch, ROOT
from review_agent.llm import ModelResult
from schemas import Inputs, digest
from workflow.graph import Workflow

SUITE = "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


class GenerationModel(ScriptedModel):
    def __init__(self, response=None, proposals=None):
        super().__init__(proposals=proposals)
        self.response = response if response is not None else {"tests": SUITE, "rationale": "依据加法契约检查公开接口；未覆盖所有数值类型。"}
        self.generations = 0

    def invoke(self, system, user, tools=None):
        if "你负责在修复前" in system:
            self.generations += 1
            self.calls.append((system, json.loads(user), tools))
            if isinstance(self.response, Exception):
                raise self.response
            if isinstance(self.response, ModelResult):
                return self.response
            return ModelResult(json.dumps(self.response), [])
        return super().invoke(system, user, tools)


class TrackingRunner(FakeRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.suites = []

    def run_tests(self, source, tests, directory):
        self.suites.append(tests)
        return super().run_tests(source, tests, directory)


def test_tests_and_requirement_optional():
    assert Inputs(source="x=1", requirement="x 为1").tests == ""
    assert Inputs(source="x=1").requirement == ""
    with pytest.raises(ValueError):
        Inputs(source="x=1", requirement="x 为1", tests=" ")


def test_generated_suite_frozen_and_exported(tmp_path, inputs):
    inputs = Inputs(**(inputs.model_dump() | {"tests": ""}))
    model = GenerationModel(proposals=[[patch()]])
    runner = TrackingRunner(["failed", "passed"])
    wf = Workflow(model, runner=runner)
    report = wf.run(inputs, tmp_path / "run")
    assert report.status == "verified", report.reason
    assert report.tests_origin == "generated" and report.generated_tests.tests == SUITE
    assert model.generations == 2
    assert next(tools for _, payload, tools in model.calls if "original_source" in payload) is None
    generated_call = next(p for _, p, _ in model.calls if "original_source" in p)
    assert generated_call["original_source"] == inputs.source
    consistency_call = [p for _, p, _ in model.calls if "original_source" in p][1]
    assert consistency_call["draft_tests"]["tests"] == SUITE
    assert "consistency_review" in consistency_call
    assert "test_result" not in consistency_call
    assert runner.suites == [SUITE, SUITE]
    assert inputs.tests == ""
    assert report.tests_hash == digest(SUITE) == report.baseline_test.tests_hash == report.attempts[0].tests.tests_hash
    assert report.attempts[0].task.tests_hash == digest(SUITE)
    assert report.attempts[0].task.tests == SUITE
    assert report.attempts[0].task.tests_origin == "generated"
    assert len(report.model_calls) == 6 and report.model_call_limit == 43
    assert (wf.directory / "inputs/test_solution.py").read_text() == SUITE
    assert (wf.directory / "result/generated_test_solution.py").read_text() == SUITE
    text = (wf.directory / "result/report.md").read_text()
    assert "不等同于用户指定验收" in text and SUITE in text
    assert report.limitations
    stages = [e.node for e in report.events if e.phase == "start"]
    assert stages.index("preflight") < stages.index("initial_review") < stages.index("generate_tests") < stages.index("baseline")


def test_user_suite_not_regenerated(tmp_path, inputs):
    model = GenerationModel()
    report = Workflow(model, runner=TrackingRunner()).run(inputs, tmp_path / "run")
    assert model.generations == 0
    assert report.tests_origin == "user" and report.generated_tests is None
    assert report.tests_hash == digest(inputs.tests) and report.model_call_limit == 41


def test_preflight_failure_does_not_generate(tmp_path, inputs):
    inputs = Inputs(**(inputs.model_dump() | {"tests": ""}))
    model = GenerationModel()
    report = Workflow(model, runner=TrackingRunner(preflight_error=True)).run(inputs, tmp_path / "run")
    assert report.status == "error" and model.generations == 0
    assert not report.model_calls and not report.generated_tests and not report.baseline_test


@pytest.mark.parametrize("response", [
    {"tests": "", "rationale": "空"},
    {"tests": "x=", "rationale": "语法无效"},
    {"tests": "pytest_plugins=['evil']", "rationale": "插件"},
    {"tests": SUITE, "rationale": "说明", "unexpected": True},
    ModelResult("bad json", []),
    ModelResult("", [{"name": "read_source_range", "args": {}}]),
    RuntimeError("private-provider-body"),
])
def test_generation_failure_stops_without_free_retry(tmp_path, inputs, response):
    inputs = Inputs(**(inputs.model_dump() | {"tests": ""}))
    model = GenerationModel(response)
    runner = TrackingRunner()
    report = Workflow(model, runner=runner).run(inputs, tmp_path / "run")
    assert report.status == "error" and model.generations == (1 if isinstance(response, RuntimeError) else 2)
    assert len(report.model_calls) == 1 + model.generations
    assert report.initial_review is not None
    assert not runner.suites and not report.attempts and not report.accepted_candidate
    assert "private-provider-body" not in report.model_dump_json()


def test_generation_bounded_format_retry():
    model = GenerationModel(ModelResult("bad json", []))
    reviewer = Reviewer(model)
    with pytest.raises(ValueError):
        reviewer.generate_tests("def add(a,b): return a+b", "返回两数和")
    assert len(reviewer.test_generation_model.calls) == 2


def test_cli_can_omit_tests(monkeypatch, tmp_path, capsys):
    import cli
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-private-key")
    monkeypatch.setattr(cli, "DeepSeekModel", lambda config: GenerationModel())
    monkeypatch.setattr(cli, "Workflow", lambda model, name, progress_callback: Workflow(model, name, runner=FakeRunner(), progress_callback=progress_callback))
    assert cli.main(["run", "--source", str(ROOT / "examples/clean/solution.py"),
                     "--requirement", str(ROOT / "examples/clean/requirement.md"),
                     "--output-dir", str(tmp_path / "run")]) == 0
    assert json.loads(capsys.readouterr().out)["tests_origin"] == "generated"


def test_web_can_start_without_tests(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    at.text_input[0].set_value("fixture-key").run()
    next(t for t in at.text_area if t.key == "source").set_value("def add(a,b): return a+b").run()
    assert at.session_state["tests"] == ""
    assert not next(b for b in at.button if b.label == "开始任务").disabled
    assert any("系统自动生成" in info.value for info in at.info)


@pytest.mark.parametrize("sample,source_text", [("division", "def divide"), ("mutable_default", "def append_item"), ("clean", "def add")])
def test_sample_needs_no_code_upload(monkeypatch, sample, source_text):
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    at.selectbox[0].select(sample).run()
    assert not at.exception
    assert source_text in at.text_area[0].value
    assert at.session_state["tests"].strip()
    assert at.session_state["requirement"] == ""
    assert not any(t.key == "requirement" for t in at.text_area)
    assert any("无需上传任何文件" in message.value for message in at.success)
    assert any("模型访问密钥" in message.value for message in at.warning)
    assert next(b for b in at.button if b.label == "开始任务").disabled
    at.text_input[0].set_value("fixture-key").run()
    assert not next(b for b in at.button if b.label == "开始任务").disabled
    assert all("solution.py" not in t.label for t in at.text_area)
    assert at.radio[1].value == "使用现成的检查用例"


def test_switch_sample_to_own_clears_sample(monkeypatch):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    at.selectbox[0].select("division").run()
    revision = at.session_state["input_revision"]
    at.selectbox[0].select("clean").run()
    assert "def add" in at.text_area[0].value
    assert "divide" not in at.session_state["tests"]
    at.selectbox[0].select("自行输入").run()
    assert not at.exception
    assert at.session_state["input_revision"] > revision
    for field in ("source", "tests", "requirement"):
        assert at.session_state[field] == ""
    assert at.radio[1].value == "让系统自动生成检查用例"


def test_advanced_choice_is_explicit(monkeypatch):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    at.text_input[0].set_value("fixture-key").run()
    next(t for t in at.text_area if t.key == "source").set_value("x=1").run()
    at.radio[1].set_value("使用现成的检查用例").run()
    assert next(b for b in at.button if b.label == "开始任务").disabled
    assert any("现成检查用例为空" in w.value for w in at.warning)
    next(t for t in at.text_area if t.key == "tests").set_value("def test_a(): assert True").run()
    assert not next(b for b in at.button if b.label == "开始任务").disabled
    at.radio[1].set_value("让系统自动生成检查用例").run()
    assert not next(b for b in at.button if b.label == "开始任务").disabled
    assert not any(t.key == "tests" for t in at.text_area)
