import pytest

from agents.model_errors import describe_model_failure
from review_agent.llm import ModelRequestError


@pytest.mark.parametrize("name,expected", [("APITimeoutError", "超时"), ("ReadTimeout", "超时"),
                                         ("APIConnectionError", "无法连接"), ("ConnectError", "无法连接")])
def test_exception_chain_uses_type_not_private_body(name, expected):
    provider = type(name, (RuntimeError,), {})("private-key-or-provider-body")
    error = ModelRequestError("private-outer-body")
    error.__context__ = provider
    message = describe_model_failure(error)
    assert expected in message
    assert "private" not in message


@pytest.mark.parametrize("status,expected", [(400, "参数"), (401, "密钥"), (402, "账户"),
    (403, "权限"), (404, "模型或接口"), (429, "频繁"), (503, "暂时不可用")])
def test_http_status_is_safe(status, expected):
    provider = RuntimeError("private-provider-response")
    provider.status_code = status
    error = ModelRequestError("private")
    error.__cause__ = provider
    message = describe_model_failure(error)
    assert expected in message and str(status) in message
    assert "private" not in message


def test_unknown_request_failure_not_blame_requirements():
    assert "模型请求失败" in describe_model_failure(ModelRequestError("private"))
    error = ModelRequestError("private")
    error.__context__ = error
    assert "private" not in describe_model_failure(error)


def test_generation_request_failure_reported(tmp_path, inputs):
    from schemas import Inputs
    from workflow.graph import Workflow
    from conftest import FakeRunner
    from test_optional_tests import GenerationModel
    error = ModelRequestError("private-body")
    error.__context__ = type("APITimeoutError", (RuntimeError,), {})("private-key")
    model = GenerationModel(error)
    inputs = Inputs(**(inputs.model_dump() | {"tests": ""}))
    report = Workflow(model, runner=FakeRunner()).run(inputs, tmp_path / "run")
    assert report.status == "error" and "超时" in report.reason
    assert "private" not in report.model_dump_json()
    assert len(report.model_calls) == 2 and report.model_calls[-1].failed
    assert report.initial_review is not None


def test_connection_button_uses_bounded_real_adapter_config(monkeypatch):
    import json
    import review_agent.llm
    from review_agent.llm import ModelResult
    from conftest import ROOT
    from streamlit.testing.v1 import AppTest
    captured = []
    class Probe:
        def __init__(self, config):
            captured.append(config)
        def invoke(self, system, user, tools=None):
            return ModelResult(json.dumps({"ok": True}), [])
    monkeypatch.setattr(review_agent.llm, "DeepSeekModel", Probe)
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    at.text_input[0].set_value("fixture-private-key").run()
    next(b for b in at.button if b.label == "测试模型连接").click().run(timeout=15)
    assert captured[-1].timeout == 10 and captured[-1].max_retries == 0
    assert captured[-1].max_tokens == 128
    assert any("模型连接成功" in s.value for s in at.success)


def test_web_request_failure_shows_connection_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    import review_agent.llm
    import workflow.graph
    from workflow.graph import Workflow
    from conftest import ROOT, FakeRunner
    from test_optional_tests import GenerationModel
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(review_agent.llm, "DeepSeekModel", lambda config: GenerationModel(ModelRequestError("private")))
    class Injected(Workflow):
        def __init__(self, model, model_name, progress_callback=None):
            super().__init__(model, model_name, runner=FakeRunner(), progress_callback=progress_callback)
    monkeypatch.setattr(workflow.graph, "Workflow", Injected)
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    at.text_input[0].set_value("fixture-key").run()
    at.selectbox[0].select("division").run()
    at.radio[1].set_value("让系统自动生成检查用例").run()
    next(b for b in at.button if b.label == "开始任务").click().run(timeout=15)
    at.session_state["thread"].join(timeout=10)
    at.run(timeout=15)
    assert any("模型请求失败" in text.value for text in at.markdown)
    assert not any("补充清楚功能描述" in text.value for text in at.markdown)
