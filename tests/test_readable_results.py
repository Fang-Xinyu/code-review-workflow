"""Result pages expose decisions and evidence without protocol JSON trees."""
import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT, ScriptedModel, FakeRunner, patch
from workflow.graph import Workflow


@pytest.mark.parametrize('rejected', [False, True])
def test_readable_repair_results(monkeypatch, tmp_path, rejected):
    import review_agent.llm
    import workflow.graph
    monkeypatch.chdir(tmp_path)
    proposal = patch(original='    return nonexistent') if rejected else patch()
    monkeypatch.setattr(review_agent.llm, 'DeepSeekModel', lambda config: ScriptedModel([[proposal]]))

    class InjectedWorkflow(Workflow):
        def __init__(self, model, model_name, progress_callback=None):
            super().__init__(model, model_name, runner=FakeRunner(['failed', 'passed']), progress_callback=progress_callback)

    monkeypatch.setattr(workflow.graph, 'Workflow', InjectedWorkflow)
    at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=15)
    at.text_input[0].set_value('fixture-key').run()
    next(t for t in at.text_area if t.key == 'source').set_value('def add(a, b):\n    return a - b\n').run()
    at.radio[1].set_value('使用现成的检查用例').run()
    next(t for t in at.text_area if t.key == 'tests').set_value('def test_a():\n    assert True\n').run()
    next(b for b in at.button if b.label == '开始任务').click().run(timeout=15)
    at.session_state['thread'].join(timeout=10)
    at.run(timeout=15)
    assert not at.exception
    assert not at.get('json')
    assert any(m.label == '通过' for m in at.metric)
    assert any('失败原因' in e.label for e in at.expander)
    assert any('修改说明' in m.value for m in at.markdown)
    assert any('下载完整数据报告' in b.label for b in at.get('download_button'))
    table_columns = {str(c) for d in at.dataframe for c in d.value.columns}
    assert '核对项目' in table_columns and '输入用量' in table_columns
    assert not {'source_hash', 'task_id', 'goal_ids'} & table_columns
    if rejected:
        assert at.error and not at.session_state['result'].accepted_candidate
        assert not any('已通过检查' in b.label for b in at.get('download_button'))
    else:
        assert at.session_state['result'].status == 'verified'
        assert any(c.language == 'diff' and '+    return a + b' in c.value for c in at.code)
        assert any('已通过检查' in b.label for b in at.get('download_button'))


def test_readable_review_findings(monkeypatch, tmp_path):
    import review_agent.llm
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(review_agent.llm, 'DeepSeekModel', lambda config: ScriptedModel())
    at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=15)
    at.text_input[0].set_value('fixture-key').run()
    at.radio[0].set_value('仅审查').run()
    next(t for t in at.text_area if t.key == 'source').set_value('def add_item(x, items=[]):\n    items.append(x)\n    return items\n').run()
    next(b for b in at.button if b.label == '开始任务').click().run(timeout=15)
    at.session_state['thread'].join(timeout=10)
    at.run(timeout=15)
    assert not at.exception and not at.get('json')
    assert any('可变默认参数' in e.label and '已确认' in e.label for e in at.expander)
    assert any('触发条件' in m.value for m in at.markdown)
    assert any(c.value == '[]' for c in at.code)
    assert not at.metric
