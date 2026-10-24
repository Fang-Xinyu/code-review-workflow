"""Unknown call failures cannot erase independent confirmed repair goals."""
import json
import os

import pytest

from conftest import ScriptedModel, FakeRunner, test_result
from review_agent.llm import ModelResult
from schemas import Inputs
from validation.test_runner import TestRunner
from workflow.graph import Workflow
from workflow.routing import comparable_unattributed, compare_tests

SOURCE = ('import json\n'
          'def save_summary(summary, path):\n'
          '    path.write_text(str(summary))\n'
          'def total(amount):\n'
          '    return amount - 1\n')
SUITE = '''import json
from solution import save_summary, total

def test_json(tmp_path):
    target = tmp_path / "out.json"
    save_summary({"name": "Ada", "average": None}, target)
    assert json.loads(target.read_text()) == {"name": "Ada", "average": None}

def test_total():
    assert total(21) == 42
'''


def unknown_result(source=SOURCE, tests=SUITE, *, passed=(), failed=('json', 'total')):
    result = test_result(source, tests, passed=passed, failed=failed)
    failures = [f.model_copy(update={'exception_type': 'JSONDecodeError', 'location': '/usr/local/lib/python3.12/json/decoder.py'})
                if f.nodeid == 'json' else f for f in result.failures]
    return result.model_copy(update={'status': 'inconclusive', 'failure_kind': 'unknown', 'failures': failures})


class SerializationModel(ScriptedModel):
    def __init__(self):
        super().__init__(proposals=[self.proposal, self.proposal])

    @staticmethod
    def proposal(task):
        serialization = 'str(summary)' in task['source']
        patch = ({'line_start': 3, 'line_end': 3, 'original': '    path.write_text(str(summary))',
                  'replacement': '    path.write_text(json.dumps(summary))', 'rationale': '使用 JSON 序列化保留数据类型'}
                 if serialization else {'line_start': 5, 'line_end': 5, 'original': '    return amount - 1',
                                        'replacement': '    return amount * 2', 'rationale': '按固定测试修复计算'})
        ids = [g['goal_id'] for g in task['goals']]
        addressed = [g['goal_id'] for g in task['goals'] if not serialization or 'JSON' in g['description'] or 'F401' in g['description']]
        return dict(task_id=task['task_id'], base_source_hash=task['source_hash'], goal_ids=ids,
                    decision='propose', patches=[patch], patch_targets=[{'patch_index': 0, 'goal_ids': addressed}],
                    unaddressed={g: '下一轮处理独立计算问题' for g in ids if g not in addressed},
                    summary='局部修复已确认问题', missing_information=[])

    def invoke(self, system, user, tools=None):
        payload = json.loads(user)
        if 'draft_schema' in payload:
            self.calls.append((system, payload, tools))
            findings = []
            for token, title, line, evidence in [('str(summary)', '输出不是合法 JSON', 3, 'path.write_text(str(summary))'),
                                                  ('amount - 1', '计算结果与固定测试不符', 5, 'return amount - 1')]:
                if token in payload['source']:
                    findings.append(dict(title=title, severity='medium', category='bug', line_start=line, line_end=line,
                                         evidence=evidence, trigger='调用公开函数', explanation=title,
                                         suggestion='修复该局部实现', rule_id=None))
            return ModelResult(json.dumps({'findings': findings}), [])
        return super().invoke(system, user, tools)


def test_unattributed_progress_preserves_status_and_limits_authorization(tmp_path):
    model = SerializationModel()
    runner = FakeRunner([lambda s, t: unknown_result(s, t),
                         lambda s, t: test_result(s, t, passed=('json',), failed=('total',)),
                         lambda s, t: test_result(s, t, passed=('json', 'total'))])
    report = Workflow(model, runner=runner).run(Inputs(source=SOURCE, tests=SUITE), tmp_path / 'run')
    assert report.status == 'verified', report.reason
    assert report.baseline_test.status == 'inconclusive' and report.baseline_test.failure_kind == 'unknown'
    assert len(report.attempts) == 2 and all(a.retained for a in report.attempts)
    first = report.attempts[0].task
    assert all(t.kind == 'review_issue' for t in first.targets)
    assert first.feedback[0].test_result.failures[0].exception_type == 'JSONDecodeError'
    assert first.feedback[0].candidate_hash == first.source_hash
    assert report.tests_passed and report.goals_verified and report.accepted_candidate


@pytest.mark.parametrize('change', [
    {'complete': False}, {'truncated': True}, {'exit_code': 3}, {'failure_kind': 'environment'},
    {'skipped': ['skip']}, {'xfailed': ['xfail']}, {'collected': []}, {'results': {}},
])
def test_incomplete_or_environment_failure_still_blocks(change):
    assert not comparable_unattributed(unknown_result().model_copy(update=change))


def test_unknown_failure_without_confirmed_goal_still_stops(tmp_path):
    report = Workflow(ScriptedModel(), runner=FakeRunner([lambda s, t: unknown_result(s, t)])).run(
        Inputs(source='def total(amount):\n    return amount\n', tests=SUITE), tmp_path / 'run')
    assert report.status == 'needs_attention' and not report.attempts


def test_unknown_progress_requires_explicit_authorization_and_no_new_unknown():
    baseline = unknown_result(passed=('safe',), failed=('json', 'total'))
    improved = unknown_result(SOURCE + '# changed\n', passed=('safe', 'total'), failed=('json',))
    assert not compare_tests(baseline, baseline, improved)[0]
    assert compare_tests(baseline, baseline, improved, allow_unattributed=True)[0]
    changed_error = improved.model_copy(update={'failures': [improved.failures[0].model_copy(update={'exception_type': 'ValueError'})]})
    assert compare_tests(baseline, baseline, changed_error, allow_unattributed=True)[3]
    regressed = unknown_result(SOURCE + '# changed\n', passed=('total',), failed=('safe', 'json'))
    retained, regression, _, _ = compare_tests(baseline, baseline, regressed, allow_unattributed=True)
    assert not retained and regression == ['safe']


def test_failed_test_can_reach_existing_later_error_without_losing_progress():
    baseline = unknown_result(passed=('safe',), failed=('json', 'total'))
    candidate = unknown_result(SOURCE + '# changed\n', passed=('safe', 'json'), failed=('total',))
    candidate = candidate.model_copy(update={'failures': [candidate.failures[0].model_copy(update={
        'exception_type': 'JSONDecodeError', 'location': '/usr/local/lib/python3.12/json/decoder.py'})]})
    retained, regression, reason, stop = compare_tests(baseline, baseline, candidate, allow_unattributed=True)
    assert retained and not regression and not stop
    assert candidate.status == 'inconclusive' and candidate.failure_kind == 'unknown'
    assert '改善' in reason


def test_existing_error_on_previously_passing_test_is_still_regression():
    baseline = unknown_result(passed=('safe',), failed=('json', 'total'))
    candidate = unknown_result(SOURCE + '# changed\n', passed=('json', 'total'), failed=('safe',))
    candidate = candidate.model_copy(update={'failures': [candidate.failures[0].model_copy(update={
        'exception_type': 'JSONDecodeError', 'location': '/usr/local/lib/python3.12/json/decoder.py'})]})
    retained, regression, _, _ = compare_tests(baseline, baseline, candidate, allow_unattributed=True)
    assert not retained and regression == ['safe']


def test_partial_unknown_progress_never_becomes_delivery(tmp_path):
    model = SerializationModel()
    original = model.proposal
    # Repair only the confirmed calculation defect; leave the JSON failure unresolved.
    def calculation(task):
        proposal = original(task | {'source': task['source'].replace('str(summary)', 'not_detected')})
        addressed = [g['goal_id'] for g in task['goals'] if '计算' in g['description']]
        proposal['patch_targets'] = [{'patch_index': 0, 'goal_ids': addressed}]
        proposal['unaddressed'] = {g['goal_id']: '本次未处理' for g in task['goals'] if g['goal_id'] not in addressed}
        return proposal
    model.proposals = [calculation]
    runner = FakeRunner([lambda s, t: unknown_result(s, t),
                         lambda s, t: unknown_result(s, t, passed=('total',), failed=('json',))])
    report = Workflow(model, runner=runner).run(Inputs(source=SOURCE, tests=SUITE), tmp_path / 'run')
    assert report.status == 'needs_attention' and not report.accepted_candidate
    assert report.attempts[0].retained and report.working_hash != report.source_hash
    assert report.attempts[0].static.candidate_code and not report.tests_passed
    assert not report.fix_assessment


@pytest.mark.container
@pytest.mark.skipif(os.getenv('CODEFLOW_CONTAINER_TESTS') != '1', reason='显式启用真实 Docker 测试')
def test_serialization_failure_two_rounds_real_container(tmp_path):
    report = Workflow(SerializationModel(), runner=TestRunner()).run(Inputs(source=SOURCE, tests=SUITE), tmp_path / 'run')
    assert report.baseline_test.status == 'inconclusive' and report.baseline_test.failure_kind == 'unknown'
    assert any(f.exception_type == 'JSONDecodeError' for f in report.baseline_test.failures)
    assert report.status == 'verified', report.reason
    assert len(report.attempts) == 2 and all(a.retained for a in report.attempts)
    assert report.tests_passed and report.accepted_candidate


@pytest.mark.container
@pytest.mark.skipif(os.getenv('CODEFLOW_CONTAINER_TESTS') != '1', reason='显式启用真实 Docker 测试')
def test_existing_json_failure_unmasked_after_calculation_fix(tmp_path):
    suite = SUITE + '''
def test_end_to_end(tmp_path):
    assert total(21) == 42
    target = tmp_path / "combined.json"
    save_summary({"amount": total(21)}, target)
    assert json.loads(target.read_text()) == {"amount": 42}
'''
    model = SerializationModel()
    original = model.proposal
    def calculation_first(task):
        result = original(task | {'source': task['source'].replace('str(summary)', 'not_detected')})
        addressed = [g['goal_id'] for g in task['goals'] if '计算' in g['description']]
        result['patch_targets'] = [{'patch_index': 0, 'goal_ids': addressed}]
        result['unaddressed'] = {g['goal_id']: '下一轮处理 JSON 序列化' for g in task['goals'] if g['goal_id'] not in addressed}
        return result
    model.proposals = [calculation_first, original]
    report = Workflow(model, runner=TestRunner()).run(Inputs(source=SOURCE, tests=suite), tmp_path / 'run')
    assert report.status == 'verified', report.reason
    first = report.attempts[0]
    assert first.retained and first.tests.status == 'inconclusive'
    assert len(first.tests.passed) > len(report.baseline_test.passed)
    assert any(f.nodeid.endswith('::test_end_to_end') and f.exception_type == 'JSONDecodeError' for f in first.tests.failures)
    assert len(report.attempts) == 2 and report.accepted_candidate
