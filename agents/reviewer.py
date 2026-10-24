import json
from pydantic import ValidationError

from review_agent.graph import ReviewAgent
from review_agent.llm import RecordingModel, parse_json_object
from schemas import FixAssessment, GeneratedTests, digest
from .prompts import ASSESS, GENERATE_TESTS, REVIEW_TESTS


class AssessmentProtocolError(ValueError):
    """Program-generated explanation; contains no raw provider response."""


class RequirementContextModel:
    """Pass the fixed user contract through every full-review model stage."""

    def __init__(self, model, requirement):
        self.model = model
        self.requirement = requirement

    @property
    def attempts(self):
        return getattr(self.model, "attempts", 0)

    @property
    def request_stats(self):
        return getattr(self.model, "request_stats", [])

    def invoke(self, system, user, tools=None):
        payload = parse_json_object(user)
        payload["requirement"] = self.requirement
        return self.model.invoke(system, json.dumps(payload, ensure_ascii=False), tools)


class Reviewer:
    def __init__(self, model, model_name="unknown"):
        self.model = model
        self.core = ReviewAgent(model, model_name=model_name)
        self.assessment_model = None
        self.test_generation_model = None

    def generate_tests(self, code, requirement="", review=None):
        self.test_generation_model = RecordingModel(self.model, max_calls=2)
        self.test_generation_model.stage = "generate_tests"
        payload = {"original_source": code, "source_hash": digest(code), "requirement": requirement,
                   "confirmed_findings": [f.model_dump() for f in review.findings] if review else [],
                   "schema": GeneratedTests.model_json_schema()}
        for attempt in range(2):
            system = REVIEW_TESTS if "draft_tests" in payload else GENERATE_TESTS
            result = self.test_generation_model.invoke(system, json.dumps(payload, ensure_ascii=False))
            try:
                if result.tool_calls:
                    raise ValueError("生成验收测试禁止工具调用")
                generated = GeneratedTests.model_validate(parse_json_object(result.content))
                if attempt == 0:
                    payload["draft_tests"] = generated.model_dump()
                    payload["consistency_review"] = (
                        "在运行任何测试之前，独立复核这份草稿并返回最终完整 tests/rationale。"
                        "逐条对照原始源码及 confirmed_findings：保护测试不能把已确认的错误行为固定为正确；"
                        "检查测试间是否要求相互矛盾的行为，尤其边界值、资源预算、异常策略、共享状态和返回顺序。"
                        "依据不足的业务选择只记入 rationale，不擅自增加断言。"
                        "保持有依据的缺陷检查，不通过删除缺陷检查来消除冲突；修正错误的保护断言。"
                        "未发现问题时原样返回草稿。本阶段没有执行结果，禁止宣称测试已运行。")
                    continue
                return generated
            except ValueError:
                if attempt:
                    raise ValueError("生成测试两次均未通过格式或可执行性检查") from None
                payload["format_feedback"] = (
                    "上次输出无效。请重新提交完整 JSON：只含 tests 和 rationale。"
                    "控制测试长度，禁止 skip、skipif、xfail、importorskip、插件或钩子；"
                    "无法确定的规则仅记录在 rationale 中。不读取执行结果，不改变验收标准。")

    def review(self, code, filename, requirement=""):
        # Use the public constructor; the copied review core remains unchanged.
        model = RequirementContextModel(self.model, requirement) if requirement.strip() else self.model
        self.core = ReviewAgent(model,
                                model_name=self.core.model_name, ruff_runner=self.core.ruff_runner,
                                progress_callback=self.core.progress_callback)
        offset = len(self.model.calls) if isinstance(self.model, RecordingModel) else 0
        stage = self.model.stage if isinstance(self.model, RecordingModel) else "review"
        try:
            if requirement.strip():
                report = self.core.review(code, filename=filename, focus=requirement)
            else:
                report = self.core.review(code, filename=filename)
        finally:
            if isinstance(self.model, RecordingModel):
                for global_call, local_call in zip(self.model.calls[offset:], self.core.model.calls):
                    global_call.stage = stage + "/" + local_call.stage
        if (report.source_hash, report.filename) != (digest(code), filename) or (requirement.strip() and report.focus != requirement):
            raise ValueError("完整审查快照或输入标识不符")
        return report

    def assess_fix(self, task, candidate, static_result, test_result, final_review):
        self.assessment_model = RecordingModel(self.model, max_calls=1)
        self.assessment_model.stage = "assess_fix"
        payload = {"context": task, "candidate": candidate.model_dump(),
                   "static_result": static_result.model_dump(), "test_result": test_result.model_dump(),
                   "final_review": final_review.model_dump(), "schema": FixAssessment.model_json_schema()}
        payload["required_assessment_ids"] = [{"task_id": x["task_id"], "goal_id": g["goal_id"]}
                                              for x in task["tasks"] for g in x["goals"]]
        result = self.assessment_model.invoke(ASSESS, json.dumps(payload, ensure_ascii=False))
        if result.tool_calls:
            raise AssessmentProtocolError("最终核对返回了不允许的工具调用；保留代码及已有检查结果")
        try:
            assessment = FixAssessment.model_validate(parse_json_object(result.content))
        except ValidationError as exc:
            raise AssessmentProtocolError(f"最终核对返回格式不符合要求（{exc.error_count()} 处）；保留代码及已有检查结果") from exc
        except ValueError as exc:
            raise AssessmentProtocolError("最终核对未返回有效结构化结果；保留代码及已有检查结果") from exc
        expected = {(x["task_id"], g["goal_id"]) for x in task["tasks"] for g in x["goals"]}
        actual = [(x.task_id, x.goal_id) for x in assessment.assessments]
        if (assessment.run_id != task["run_id"] or assessment.candidate_hash != candidate.source_hash
                or len(actual) != len(set(actual)) or set(actual) != expected):
            raise AssessmentProtocolError("目标核对 ID 或快照不完整："
                             f"应核对 {len(expected)} 项，返回 {len(actual)} 项，"
                             f"遗漏 {len(expected - set(actual))} 项，多出 {len(set(actual) - expected)} 项")
        return assessment
