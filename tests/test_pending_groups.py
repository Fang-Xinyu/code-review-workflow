from types import SimpleNamespace

from schemas import FixProposal, digest
from validation.patch_checks import separate_pending_conflicts


def proposal_case(links):
    source = 'a = 1\nb = 2\nc = 3\nd = 4\n'
    task = SimpleNamespace(source=source, pending_findings=[SimpleNamespace(evidence='a = 1')])
    proposal = FixProposal(task_id='group-test', base_source_hash=digest(source),
        goal_ids=sorted({g for link in links for g in link}), decision='propose',
        patches=[dict(line_start=i+1, line_end=i+1, original=line,
                      replacement=line.replace(str(i+1), '9'), rationale='修复已确认目标')
                 for i, line in enumerate(source.rstrip().split('\n')[:len(links)])],
        patch_targets=[dict(patch_index=i, goal_ids=link) for i, link in enumerate(links)])
    return task, proposal


def test_conflicting_goal_defers_whole_group_and_reindexes_independent_patch():
    task, original = proposal_case([['cache'], ['cache'], ['mask']])
    filtered = separate_pending_conflicts(task, original)
    assert [p.original for p in filtered.patches] == ['c = 3']
    assert filtered.patch_targets[0].patch_index == 0
    assert filtered.patch_targets[0].goal_ids == ['mask']
    assert set(filtered.unaddressed) == {'cache'}
    assert len(original.patches) == 3 and not original.unaddressed


def test_shared_patch_dependencies_defer_transitively():
    task, original = proposal_case([['cache'], ['cache', 'billing'], ['billing'], ['mask']])
    filtered = separate_pending_conflicts(task, original)
    assert [p.original for p in filtered.patches] == ['d = 4']
    assert set(filtered.unaddressed) == {'cache', 'billing'}


def test_all_blocked_or_invalid_patch_cannot_be_salvaged():
    task, original = proposal_case([['cache'], ['cache']])
    assert separate_pending_conflicts(task, original) is None
    original.patches[0].original = 'wrong source'
    assert separate_pending_conflicts(task, original) is None


def test_no_conflict_does_not_rewrite_proposal():
    task, original = proposal_case([['cache'], ['mask']])
    task.pending_findings = []
    assert separate_pending_conflicts(task, original) is None
