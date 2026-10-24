import json
import os
import shutil
from pathlib import Path

from pydantic import BaseModel
from schemas import Candidate


def latest_attempt_candidate(report):
    for attempt in reversed(report.attempts):
        if attempt.static and attempt.static.candidate_code and attempt.static.candidate_hash and attempt.static.candidate_hash != report.source_hash:
            return Candidate(code=attempt.static.candidate_code, source_hash=attempt.static.candidate_hash,
                             diff=attempt.static.diff or "")
    return None


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def new_run_directory(path):
    path = Path(path).absolute()
    # Resolve existing parents; reject symlinks and overwrite of any existing directory.
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError("输出目录不得经过软链接")
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(path, 0o700)
    return path


def remove_previous_run(directory):
    """Only remove a previous task directory belonging to this web session."""
    if not directory:
        return
    path = Path(directory).absolute()
    root = Path("runs").absolute()
    if (path.parent != root or len(path.name) != 32
            or any(c not in "0123456789abcdef" for c in path.name)
            or any(p.is_symlink() for p in [path, *path.parents])):
        raise ValueError("拒绝清理非任务目录")
    if path.exists():
        shutil.rmtree(path)


def markdown(report):
    lines = ["# CodeFlow 任务报告", "", f"任务：`{report.run_id}`", f"模式：`{report.mode}`",
             f"结果：**{report.status}**", report.reason, "", f"原始源码：`{report.source_hash}`",
             f"固定测试：`{report.tests_hash}`", f"测试来源：{report.tests_origin}", f"功能要求：{report.requirement}", "",
             f"测试完整通过：{report.tests_passed}；目标获支持：{report.goals_verified}；审查完整：{report.review_complete}",
             "", "## 基线"]
    if report.tests_origin == "generated":
        lines += ["测试由系统生成并固定；通过仅表示满足这些测试，不等同于用户指定验收或完整正确性。"
                  if report.generated_tests else "未提供用户测试，测试生成尚未完成；没有行为验收结果。"]
        if report.generated_tests:
            lines += [report.generated_tests.rationale, "```python", report.generated_tests.tests, "```"]
    if report.baseline_test:
        t = report.baseline_test
        lines += [f"{t.status} / {t.failure_kind}；收集 {len(t.collected)}；通过 {len(t.passed)}；失败 {len(t.failed)}",
                  "```text", *t.collected, "```"]
    else:
        lines += ["未执行测试。"]
    for i, a in enumerate(report.attempts, 1):
        lines += ["", f"## 尝试 {i}", f"基准：`{a.task.source_hash}`", a.reason,
                  f"保留为工作候选：{a.retained}；回归：{a.regression}"]
        if a.proposal:
            lines += [a.proposal.summary]
            lines += [f"- 未处理：{g.description}；{a.proposal.unaddressed[g.goal_id]}"
                      for g in a.task.goals if g.goal_id in a.proposal.unaddressed]
        if a.static and a.static.diff:
            lines += ["```diff", a.static.diff, "```"]
        if a.tests:
            lines += [f"测试：{a.tests.status} / {a.tests.failure_kind}", *[f"- {f.nodeid}：{f.exception_type}" for f in a.tests.failures]]
    lines += ["", "## 目标核对"]
    if report.fix_assessment:
        lines += [f"- {a.task_id}/{a.goal_id}：{a.verdict}；{a.reason}" for a in report.fix_assessment.assessments]
    else:
        lines += ["未执行目标核对。"]
    remaining_title = "## 初审问题（部分修改尚未完成复审）" if report.partial_candidate and not report.final_review else "## 剩余与人工事项"
    lines += ["", remaining_title, *[f"- {f.title}（{f.status}）" for f in report.outstanding_findings + report.manual_items],
              "", "## 限制", *[f"- {s}" for s in report.limitations], "", "## 调用记录",
              f"逻辑调用 {len(report.model_calls)}/{report.model_call_limit}；HTTP 尝试 {report.http_attempts}；工具 {report.tool_calls}",
              f"已知输入 token：{report.input_tokens}；已知输出 token：{report.output_tokens}"]
    lines += [f"- {c.stage}：{c.elapsed_ms} ms；HTTP {c.http_attempts}；失败 {c.failed}；token {c.input_tokens}/{c.output_tokens}" for c in report.model_calls]
    lines += ["", "## 事件", *[f"- {e.node}/{e.phase}：{e.message}（{e.elapsed_ms} ms，{e.source_hash[:12]}）" for e in report.events]]
    output = ("已通过最终检查的修改见 accepted_solution.py。" if report.accepted_candidate else
              "最新保留的部分修改见 partial_solution.py；尚未通过全部最终核验，剩余问题与测试失败见上文。" if report.partial_candidate else
              "最后尝试的代码见 last_attempt_solution.py；该版本未通过检查、未被采用，不是已验证修复。" if latest_attempt_candidate(report) else
              "未生成修改代码；原始源码保留在 inputs/solution.py。")
    lines += ["", "## 输出代码", output]
    return "\n\n".join(lines)


def export_report(directory, report):
    result = Path(directory) / "result"
    result.mkdir(exist_ok=True)
    save_json(result / "report.json", report)
    (result / "report.md").write_text(markdown(report), encoding="utf-8")
    if report.generated_tests:
        (result / "generated_test_solution.py").write_text(report.generated_tests.tests, encoding="utf-8")
    if report.accepted_candidate:
        (result / "accepted_solution.py").write_text(report.accepted_candidate.code, encoding="utf-8")
    elif report.partial_candidate:
        (result / "partial_solution.py").write_text(report.partial_candidate.code, encoding="utf-8")
    else:
        last = latest_attempt_candidate(report)
        if last:
            (result / "last_attempt_solution.py").write_text(last.code, encoding="utf-8")
