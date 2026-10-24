from __future__ import annotations

import ast
import hashlib
from typing import Annotated, Literal

from pydantic import Field, model_validator
from review_agent.models import (StrictModel, PatchEdit, ReviewedFinding, ReviewReport,
                                 StaticFinding, ToolStatus, ModelCallStats)
from review_agent.verification import normalize_and_validate

Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Text = Annotated[str, Field(min_length=1, max_length=4000)]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Inputs(StrictModel):
    mode: Literal["review", "run"] = "run"
    filename: Literal["solution.py"] = "solution.py"
    source: str
    tests: str = ""
    requirement: str = Field(default="", max_length=1000)
    max_attempts: int = Field(default=8, ge=1, le=8)

    @model_validator(mode="after")
    def validate_text(self):
        self.source, _ = normalize_and_validate(self.source, self.filename)
        if self.mode == "run":
            if self.tests:
                self.tests, _ = normalize_and_validate(self.tests, "test_solution.py")
            try:
                ast.parse(self.tests, filename="test_solution.py")
            except SyntaxError:
                raise ValueError("验收测试语法无效，不能由 Fixer 修改") from None
            tree = ast.parse(self.tests)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("pytest_"):
                    raise ValueError("不支持用户 pytest 插件钩子")
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(isinstance(t, ast.Name) and t.id == "pytest_plugins" for t in targets):
                        raise ValueError("不支持用户 pytest_plugins")
        return self


class GeneratedTests(StrictModel):
    tests: str = Field(min_length=1, max_length=102400)
    rationale: Text

    @model_validator(mode="after")
    def valid_suite(self):
        validated = Inputs(source="pass\n", tests=self.tests, requirement="校验测试文件")
        self.tests = validated.tests
        tree = ast.parse(self.tests)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr in {"skip", "skipif", "xfail", "importorskip"}
                    or isinstance(node, ast.ImportFrom) and node.module == "pytest"
                    and any(alias.name in {"skip", "skipif", "xfail", "importorskip"} for alias in node.names)):
                raise ValueError("系统生成的固定测试不能包含跳过或预期失败；不确定的契约应写入覆盖限制")
        return self


class Failure(StrictModel):
    nodeid: Text
    phase: Literal["setup", "call", "teardown", "collect"]
    exception_type: str = ""
    traceback: str = Field(max_length=32000)
    location: str = ""
    source_frames: list[str] = Field(default_factory=list)
    expected: str | None = None
    actual: str | None = None


class Phase(StrictModel):
    phase: Literal["setup", "call", "teardown"]
    outcome: Literal["passed", "failed", "skipped"]
    duration_ms: int = Field(ge=0)
    wasxfail: bool = False


class TestResult(StrictModel):
    __test__ = False
    run_id: Text
    source_hash: Hash
    tests_hash: Hash
    environment_id: str
    status: Literal["passed", "failed", "error", "inconclusive"]
    failure_kind: Literal["none", "assertion", "source_exception", "source_syntax", "test_input", "environment", "timeout", "unknown"]
    collected: list[str] = Field(default_factory=list)
    passed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    xfailed: list[str] = Field(default_factory=list)
    results: dict[str, list[Phase]] = Field(default_factory=dict)
    exit_code: int | None = None
    duration_ms: int = Field(default=0, ge=0)
    stdout: str = Field(default="", max_length=65536)
    stderr: str = Field(default="", max_length=65536)
    report_path: str = ""
    failures: list[Failure] = Field(default_factory=list)
    complete: bool = False
    truncated: bool = False

    @model_validator(mode="after")
    def honest_pass(self):
        if self.status == "passed" and (not self.complete or self.truncated or not self.collected
                or set(self.passed) != set(self.collected) or self.failed or self.skipped or self.xfailed
                or self.exit_code != 0 or set(self.results) != set(self.collected)
                or any([p.phase for p in self.results[n]] != ["setup", "call", "teardown"]
                       or any(p.outcome != "passed" or p.wasxfail for p in self.results[n]) for n in self.collected)):
            raise ValueError("完整通过需要所有测试的三个阶段和完整报告")
        return self


class Goal(StrictModel):
    goal_id: Text
    kind: Literal["review_issue", "test_failure"]
    description: Text
    constraints: list[str] = Field(default_factory=list, max_length=10)


class ReviewTarget(StrictModel):
    kind: Literal["review_issue"] = "review_issue"
    goal_id: Text
    source_hash: Hash
    findings: list[ReviewedFinding] = Field(min_length=1)

    @model_validator(mode="after")
    def confirmed(self):
        if any(f.status == "needs_confirmation" for f in self.findings):
            raise ValueError("待确认问题不能成为自动目标")
        return self


class TestTarget(StrictModel):
    __test__ = False
    kind: Literal["test_failure"] = "test_failure"
    goal_id: Text
    source_hash: Hash
    failures: list[Failure] = Field(min_length=1)


Target = Annotated[ReviewTarget | TestTarget, Field(discriminator="kind")]


class Feedback(StrictModel):
    candidate_hash: Hash
    current_snapshot: Hash
    diff: str = ""
    rolled_back: bool
    errors: list[str] = Field(default_factory=list)
    rejected_candidate: str | None = None
    test_result: TestResult | None = None

    @model_validator(mode="after")
    def bound(self):
        if self.test_result and self.test_result.source_hash != self.candidate_hash:
            raise ValueError("反馈测试快照不一致")
        return self


class FixTask(StrictModel):
    task_id: Text
    attempt: int = Field(ge=1, le=8)
    filename: Literal["solution.py"] = "solution.py"
    source: str = Field(min_length=1, max_length=102400)
    source_hash: Hash
    requirement: str = Field(default="", max_length=1000)
    requirement_hash: Hash
    tests_hash: Hash
    tests: str | None = Field(default=None, max_length=102400)
    tests_origin: Literal["user", "generated", "none"] = "none"
    goals: list[Goal] = Field(min_length=1, max_length=5)
    targets: list[Target] = Field(min_length=1, max_length=5)
    reference_review: ReviewReport
    review_source_hash: Hash
    pending_findings: list[ReviewedFinding] = Field(default_factory=list)
    feedback: list[Feedback] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def matching(self):
        ids = [g.goal_id for g in self.goals]
        if len(set(ids)) != len(ids) or sorted(ids) != sorted(t.goal_id for t in self.targets):
            raise ValueError("目标必须唯一且与证据一一关联")
        if digest(self.source) != self.source_hash or digest(self.requirement) != self.requirement_hash:
            raise ValueError("任务哈希不一致")
        if self.tests is not None and digest(self.tests) != self.tests_hash:
            raise ValueError("固定测试内容与哈希不一致")
        if self.reference_review.source_hash != self.review_source_hash:
            raise ValueError("审查背景哈希不一致")
        for goal, target in zip(sorted(self.goals, key=lambda x: x.goal_id), sorted(self.targets, key=lambda x: x.goal_id)):
            if goal.kind != target.kind:
                raise ValueError("目标来源不一致")
            if target.kind == "test_failure" and target.source_hash != self.source_hash:
                raise ValueError("测试失败证据必须属于当前快照")
        if any(f.current_snapshot != self.source_hash for f in self.feedback):
            raise ValueError("反馈 current_snapshot 与任务不符")
        return self


class PatchTarget(StrictModel):
    patch_index: int = Field(ge=0, le=9)
    goal_ids: list[str] = Field(min_length=1, max_length=5)


class FixProposal(StrictModel):
    task_id: Text
    base_source_hash: Hash
    goal_ids: list[str] = Field(min_length=1, max_length=5)
    decision: Literal["propose", "cannot_fix"]
    patches: list[PatchEdit] = Field(default_factory=list, max_length=10)
    patch_targets: list[PatchTarget] = Field(default_factory=list, max_length=10)
    unaddressed: dict[str, Text] = Field(default_factory=dict)
    summary: Text = "模型未提供修改摘要；请查看具体补丁。"
    missing_information: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_decision(self):
        ids = set(self.goal_ids)
        if len(ids) != len(self.goal_ids):
            raise ValueError("重复目标")
        if self.decision == "cannot_fix":
            if self.patches or self.patch_targets or not self.missing_information:
                raise ValueError("拒绝修复必须无补丁并说明缺失信息")
        elif not self.patches:
            raise ValueError("propose 必须包含补丁")
        if self.decision == "propose":
            if sorted(p.patch_index for p in self.patch_targets) != list(range(len(self.patches))):
                raise ValueError("每处补丁必须恰好关联一次目标")
            associated = {g for p in self.patch_targets for g in p.goal_ids}
            if not associated <= ids or not set(self.unaddressed) <= ids or associated | set(self.unaddressed) != ids:
                raise ValueError("补丁目标不完整或包含未知目标")
        return self


class StaticResult(StrictModel):
    status: Literal["passed", "failed", "error"]
    base_source_hash: Hash
    candidate_hash: Hash | None = None
    candidate_code: str | None = None
    diff: str | None = None
    syntax: Literal["passed", "failed", "not_checked"] = "not_checked"
    tool_statuses: list[ToolStatus] = Field(default_factory=list)
    before: list[StaticFinding] = Field(default_factory=list)
    after: list[StaticFinding] = Field(default_factory=list)
    new_diagnostics: list[StaticFinding] = Field(default_factory=list)
    target_checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AssessmentItem(StrictModel):
    task_id: Text
    goal_id: Text
    verdict: Literal["resolved", "unresolved", "uncertain"]
    reason: Text


class FixAssessment(StrictModel):
    run_id: Text
    candidate_hash: Hash
    assessments: list[AssessmentItem] = Field(min_length=1, max_length=40)


class Candidate(StrictModel):
    source_hash: Hash
    code: str
    diff: str


class Attempt(StrictModel):
    task: FixTask
    proposal: FixProposal | None = None
    unfiltered_proposal: FixProposal | None = None
    tool_observations: list[dict] = Field(default_factory=list)
    static: StaticResult | None = None
    tests: TestResult | None = None
    regression: list[str] = Field(default_factory=list)
    retained: bool = False
    next_source_hash: Hash
    reason: str = ""


class Event(StrictModel):
    node: str
    phase: Literal["start", "complete", "failed"]
    role: str
    attempt: int
    source_hash: Hash
    elapsed_ms: int = Field(default=0, ge=0)
    message: str


class WorkflowReport(StrictModel):
    run_id: Text
    mode: Literal["run", "review"]
    status: Literal["verified", "no_repair_needed", "needs_attention", "error", "review_completed"]
    reason: str
    source_hash: Hash
    tests_hash: Hash
    requirement_hash: Hash
    requirement: str
    tests_origin: Literal["user", "generated", "none"] = "user"
    generated_tests: GeneratedTests | None = None
    working_hash: Hash
    baseline_test: TestResult | None = None
    initial_review: ReviewReport | None = None
    attempts: list[Attempt] = Field(default_factory=list, max_length=8)
    final_review: ReviewReport | None = None
    fix_assessment: FixAssessment | None = None
    accepted_candidate: Candidate | None = None
    partial_candidate: Candidate | None = None
    outstanding_findings: list[ReviewedFinding] = Field(default_factory=list)
    manual_items: list[ReviewedFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    goals_verified: bool = False
    tests_passed: bool = False
    review_complete: bool = False
    events: list[Event] = Field(default_factory=list)
    model_calls: list[ModelCallStats] = Field(default_factory=list)
    model_call_limit: int
    http_attempts: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: int | None = 0
    environment: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def truthful_conclusion(self):
        if self.partial_candidate:
            retained = [a for a in self.attempts if a.retained]
            if (self.accepted_candidate or not retained or not retained[-1].static
                    or retained[-1].static.status != "passed" or not retained[-1].tests
                    or retained[-1].tests.source_hash != self.partial_candidate.source_hash
                    or self.partial_candidate.source_hash != self.working_hash
                    or self.partial_candidate.source_hash == self.source_hash
                    or digest(self.partial_candidate.code) != self.partial_candidate.source_hash
                    or retained[-1].static.candidate_code != self.partial_candidate.code
                    or retained[-1].next_source_hash != self.partial_candidate.source_hash):
                raise ValueError("部分修改版必须来自实际保留的工作版本，不能用被回退的候选代替")
        if self.accepted_candidate:
            if not (self.tests_passed and self.goals_verified and self.review_complete
                    and self.final_review and self.fix_assessment
                    and self.accepted_candidate.source_hash == self.working_hash == self.final_review.source_hash
                    and digest(self.accepted_candidate.code) == self.accepted_candidate.source_hash):
                raise ValueError("已接受代码缺少最终验证证据")
        if self.status == "verified" and (not self.accepted_candidate or self.manual_items or self.outstanding_findings):
            raise ValueError("verified 不允许缺少候选或存在剩余事项")
        if self.status == "no_repair_needed" and (self.mode != "run" or not self.tests_passed or not self.review_complete
                or self.manual_items or self.outstanding_findings):
            raise ValueError("无需修复必须完整通过基线与审查")
        if self.status == "review_completed" and (self.mode != "review" or self.accepted_candidate or self.baseline_test):
            raise ValueError("仅审查不能声称测试或修复通过")
        return self
