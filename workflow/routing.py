from schemas import Goal, ReviewTarget, TestTarget, digest
from validation.result_parser import classify_failure

CODE_FAILURES = {"assertion", "source_exception", "source_syntax"}


def review_complete(report, allow_syntax=False):
    allowed = {"completed", "syntax_error"} if allow_syntax else {"completed"}
    return (not report.limitations and not any(v.action == "truncated" for v in report.validation_records)
            and {s.name for s in report.tool_statuses} >= {"ast", "ruff"}
            and all(s.status in allowed for s in report.tool_statuses))


def review_goals(report):
    goals, targets = [], []
    groups = {}
    for finding in sorted(report.findings, key=lambda f: ({"high": 0, "medium": 1, "low": 2}[f.severity], f.line_start, f.finding_id)):
        if finding.status == "needs_confirmation":
            continue
        # Conservative mechanism grouping: preserve distinct columns/identities as evidence.
        key = (finding.origin, finding.rule_id, finding.line_start, finding.evidence, finding.explanation)
        groups.setdefault(key, []).append(finding)
    for findings in groups.values():
        f = findings[0]
        gid = "R-" + f.finding_id
        goals.append(Goal(goal_id=gid, kind="review_issue", description=f.title + "；" + f.explanation,
                          constraints=["依据已确认问题进行最小修复；涉及未知业务选择时拒绝猜测", "旧审查行号仅属于 review_source_hash"] ))
        targets.append(ReviewTarget(goal_id=gid, source_hash=report.source_hash, findings=findings))
    return goals, targets


def failure_goals(test):
    if test.status != "failed" or test.failure_kind not in CODE_FAILURES:
        return [], []
    # No root cause assumption: each failing test remains separate evidence.
    goals, targets = [], []
    for failure in test.failures:
        gid = "T-" + digest(failure.nodeid + ":" + failure.phase)[:16]
        goals.append(Goal(goal_id=gid, kind="test_failure", description="依据固定测试及源码证据解决真实失败：" + failure.nodeid,
                          constraints=["根因由 Fixer 核对；不得改变测试或需求", "必要行为依据不足则 cannot_fix"]))
        targets.append(TestTarget(goal_id=gid, source_hash=test.source_hash, failures=[failure]))
    return goals, targets


def comparable_unattributed(test):
    """Complete call failures can measure progress without authorizing unknown fixes."""
    return (test.status == "inconclusive" and test.failure_kind == "unknown"
            and test.complete and not test.truncated and test.exit_code == 1
            and bool(test.collected) and bool(test.failed) and not test.skipped and not test.xfailed
            and set(test.results) == set(test.collected) == set(test.passed) | set(test.failed)
            and not set(test.passed) & set(test.failed)
            and {f.nodeid for f in test.failures} == set(test.failed)
            and any(classify_failure(f) == "unknown" for f in test.failures)
            and all(f.phase == "call" and classify_failure(f) in CODE_FAILURES | {"unknown"} for f in test.failures)
            and all([p.phase for p in phases] == ["setup", "call", "teardown"]
                    and all(not p.wasxfail and p.outcome == ("failed" if phase == "call" and node in test.failed else "passed")
                            for phase, p in zip(("setup", "call", "teardown"), phases))
                    for node, phases in test.results.items()))


def unknown_signatures(test):
    return {(f.nodeid, f.phase, f.exception_type, f.location) for f in test.failures if classify_failure(f) == "unknown"}


def compare_tests(baseline, working, candidate, *, allow_unattributed=False):
    uncertain = allow_unattributed and comparable_unattributed(baseline) and comparable_unattributed(candidate)
    if ((candidate.status not in ("passed", "failed") and not uncertain) or not candidate.complete or candidate.truncated
            or candidate.status == "failed" and candidate.failure_kind not in CODE_FAILURES):
        return False, [], "测试报告不足或执行异常", True
    if candidate.source_hash == working.source_hash or candidate.tests_hash != working.tests_hash:
        return False, [], "测试快照或套件标识不符", True
    if (set(working.collected) != set(candidate.collected)
            and not (working.failure_kind == "source_syntax" and not working.collected)):
        return False, [], "测试身份集合改变", True
    regression = sorted((set(baseline.passed) | set(working.passed)) - set(candidate.passed))
    if regression:
        return False, regression, "保护测试发生回归，回退本轮基准", False
    if uncertain:
        original = unknown_signatures(baseline)
        known_errors = {(phase, exception, location) for _, phase, exception, location in original}
        already_failed = set(baseline.failed) | set(working.failed)
        # A failed test can reach a later failing check after an earlier defect
        # is fixed. An existing error type/location on an already-failing test
        # is not proof of a newly introduced defect, nor proof of its root cause.
        # Keep it unresolved; only measured progress can retain the candidate.
        newly_unexplained = [signature for signature in unknown_signatures(candidate)
                            if signature not in original and not (signature[0] in already_failed
                                                                  and signature[1:] in known_errors)]
        if newly_unexplained:
            return False, [], "出现新的未归因错误类型或位置，停止并保留上一版本", True
    if candidate.status == "passed":
        return True, [], "完整测试通过，保留候选", False
    progress = set(working.failed) & set(candidate.passed)
    if progress:
        return True, [], "无回归且失败测试有改善，保留工作候选", False
    return False, [], "无明确进展，保留本轮基准", False
