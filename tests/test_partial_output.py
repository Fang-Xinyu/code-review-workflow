"""Every available modification is exported with an honest verification label."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT, FakeRunner, ScriptedModel, test_result, patch
from reporting import markdown
from schemas import Inputs, WorkflowReport
from test_unattributed_progress import SerializationModel, SOURCE, SUITE, unknown_result
from workflow.graph import Workflow


def partial_run(directory):
    runner = FakeRunner([lambda s, t: unknown_result(s, t),
                         lambda s, t: test_result(s, t, passed=('json',), failed=('total',))])
    return Workflow(SerializationModel(), runner=runner).run(Inputs(source=SOURCE, tests=SUITE, max_attempts=1), directory)


def test_partial_working_version_exported_without_success_claim(tmp_path):
    report = partial_run(tmp_path / 'run')
    assert report.status == 'needs_attention' and report.partial_candidate and not report.accepted_candidate
    assert report.partial_candidate.source_hash == report.working_hash
    assert 'json.dumps(summary)' in report.partial_candidate.code
    assert 'return amount - 1' in report.partial_candidate.code
    assert (tmp_path / 'run/result/partial_solution.py').read_text() == report.partial_candidate.code
    assert not (tmp_path / 'run/result/accepted_solution.py').exists()
    assert not report.tests_passed and report.outstanding_findings
    saved = WorkflowReport.model_validate(json.loads((tmp_path / 'run/result/report.json').read_text()))
    assert saved.partial_candidate == report.partial_candidate
    assert '尚未通过全部最终核验' in markdown(report)


def test_last_rejected_attempt_exported_as_rejected(tmp_path, inputs):
    runner = FakeRunner(['failed', 'failed'])
    report = Workflow(ScriptedModel([[patch()]]), runner=runner).run(
        Inputs(**(inputs.model_dump() | {'max_attempts': 1})), tmp_path / 'run')
    assert not report.partial_candidate and not report.accepted_candidate
    assert not report.attempts[0].retained
    assert 'return a + b' in (tmp_path / 'run/result/last_attempt_solution.py').read_text()
    assert not (tmp_path / 'run/result/partial_solution.py').exists()
    assert '未通过检查、未被采用' in markdown(report)


def test_partial_version_cannot_use_rejected_candidate(tmp_path):
    report = partial_run(tmp_path / 'run')
    data = report.model_dump()
    data['attempts'][0]['retained'] = False
    with pytest.raises(ValueError, match='部分修改版'):
        WorkflowReport.model_validate(data)


def test_partial_output_visible_on_web(monkeypatch, tmp_path):
    import review_agent.llm
    import workflow.graph
    report = partial_run(tmp_path / 'saved')
    monkeypatch.setattr(review_agent.llm, 'DeepSeekModel', lambda config: object())

    class SavedWorkflow:
        directory = tmp_path / 'saved'
        def __init__(self, *args, **kwargs):
            pass
        def run(self, inputs):
            return report

    monkeypatch.setattr(workflow.graph, 'Workflow', SavedWorkflow)
    at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=15)
    at.text_input[0].set_value('fixture-key').run()
    next(t for t in at.text_area if t.key == 'source').set_value(SOURCE).run()
    next(b for b in at.button if b.label == '开始任务').click().run(timeout=15)
    at.session_state['thread'].join(timeout=10)
    at.run(timeout=15)
    assert not at.exception
    assert any('下载部分修改版' in b.label for b in at.get('download_button'))
    assert not any('下载修复后的代码（已通过检查）' == b.label for b in at.get('download_button'))
    assert any(c.language == 'python' and 'json.dumps(summary)' in c.value for c in at.code)
    assert any('剩余' in m.value or '仍有问题' in m.value for m in at.markdown)
