import queue
import threading
from pathlib import Path

import streamlit as st

from reporting import markdown, latest_attempt_candidate, remove_previous_run
from review_agent.config import ModelConfig
from review_agent.llm import DeepSeekModel, ModelRequestError, test_connection as check_model_connection
from agents.model_errors import describe_model_failure
from schemas import Inputs, digest
from validation.test_runner import TestRunner
from workflow.graph import Workflow


STAGES = {"initialize": "准备输入", "preflight": "检查运行环境", "generate_tests": "生成检查用例",
          "baseline": "检查原始代码", "initial_review": "审查代码问题", "select": "选择修复问题",
          "fixer": "提出修改", "static": "检查修改是否有效", "tests": "运行检查用例",
          "final_review": "复审修改后的代码", "assessment": "核对问题是否解决", "finish": "汇总结果"}
TEST_STATUS = {"passed": "全部通过", "failed": "存在失败", "error": "执行异常", "inconclusive": "无法完整验证"}


def show_findings(findings, empty_message):
    if not findings:
        st.caption(empty_message)
        return
    for finding in findings:
        level = {"high": "高", "medium": "中", "low": "低"}[finding.severity]
        location = f"第 {finding.line_start} 行" if finding.line_start == finding.line_end else f"第 {finding.line_start}–{finding.line_end} 行"
        confirmation = "待确认" if finding.status == "needs_confirmation" else "已确认"
        with st.expander(f"{level}风险 · {finding.title} · {location} · {confirmation}"):
            st.write(finding.explanation)
            st.markdown("**触发条件**")
            st.write(finding.trigger)
            st.markdown("**相关代码**")
            st.code(finding.evidence, language="python")
            st.markdown("**建议处理**")
            st.write(finding.suggestion)
            if finding.review_note:
                st.caption("复核说明：" + finding.review_note)


def show_test_result(test):
    st.write("检查结果：" + TEST_STATUS[test.status])
    columns = st.columns(4)
    for column, label, value in zip(columns, ("检查项", "通过", "失败", "跳过或预期失败"),
                                    (len(test.collected), len(test.passed), len(test.failed), len(test.skipped) + len(test.xfailed))):
        column.metric(label, value)
    st.caption(f"耗时 {test.duration_ms / 1000:.2f} 秒")
    if not test.complete or test.truncated:
        st.warning("检查记录不完整，不能据此判定全部通过。")
    if test.collected:
        rows = []
        for node in test.collected:
            status = "通过" if node in test.passed else "失败" if node in test.failed else "跳过" if node in test.skipped else "预期失败" if node in test.xfailed else "未完成"
            rows.append({"检查项": node.split("::", 1)[-1], "结果": status})
        st.dataframe(rows, hide_index=True, width="stretch")
    for failure in test.failures:
        with st.expander("失败原因：" + failure.nodeid.split("::", 1)[-1]):
            st.write("异常类型：" + (failure.exception_type or "未确定"))
            if failure.expected is not None:
                st.text("预期：" + failure.expected)
            if failure.actual is not None:
                st.text("实际：" + failure.actual)
            st.code(failure.traceback, language="text")
    if test.status in ("error", "inconclusive"):
        kinds = {"environment": "运行环境异常", "timeout": "运行超时", "test_input": "检查用例或依赖不受支持",
                 "unknown": "失败原因尚不能确定", "none": "未取得完整检查结果"}
        st.warning(kinds.get(test.failure_kind, "检查未完成，请查看失败原因。"))
        if test.stderr or test.stdout:
            with st.expander("查看运行错误信息"):
                st.code(test.stderr or test.stdout, language="text")


def show_attempt(attempt):
    st.markdown("**本次要解决的问题**")
    for goal in attempt.task.goals:
        st.write("• " + goal.description)
    if attempt.proposal:
        if attempt.unfiltered_proposal:
            st.info("本轮部分修改涉及待人工确认的事项，已按目标整组暂缓；下面展示实际进入检查的独立修改。原始方案保存在任务记录中。")
        st.markdown("**修改说明**")
        st.write(attempt.proposal.summary)
        for missing in attempt.proposal.missing_information:
            st.warning(missing)
        for goal in attempt.task.goals:
            if goal.goal_id in attempt.proposal.unaddressed:
                st.write("未处理：" + goal.description + "；" + attempt.proposal.unaddressed[goal.goal_id])
    if attempt.static:
        st.write("代码检查：" + {"passed": "通过", "failed": "未通过", "error": "工具执行异常"}[attempt.static.status])
        for error in attempt.static.errors:
            st.error(error)
        for limitation in attempt.static.limitations:
            st.warning(limitation)
        if attempt.static.diff:
            st.markdown("**具体改动（红色删除、绿色新增）**")
            st.code(attempt.static.diff, language="diff")
    if attempt.tests:
        show_test_result(attempt.tests)
    else:
        st.caption("本次修改未进入测试执行。")
    if attempt.regression:
        st.warning("修改导致原本通过的检查失败，已回退：" + "、".join(n.split("::", 1)[-1] for n in attempt.regression))
    st.write("处理结果：" + ("保留为后续工作版本，最终是否接受见结束原因" if attempt.retained else "未采用本次修改"))


def worker(inputs, config, messages):
    workflow = None
    try:
        workflow = Workflow(DeepSeekModel(config), config.model, progress_callback=lambda e: messages.put(("event", e)))
        workflow.retain_run_files = True
        report = workflow.run(inputs)
        messages.put(("result", (report, str(workflow.directory))))
    except Exception as exc:
        messages.put(("error", f"任务失败（{type(exc).__name__}）；请检查输入、配置和环境"))
    finally:
        if workflow is not None and workflow.directory is not None:
            messages.put(("directory", str(workflow.directory)))


st.set_page_config(page_title="CodeFlow", layout="wide")
st.title("CodeFlow")
st.caption("上传或粘贴一个 Python 代码文件即可。系统先完整审查，再尝试修复并检查结果。")
config = ModelConfig.from_env()
with st.sidebar:
    key = st.text_input("模型访问密钥", value=config.api_key, type="password")
    base_url = st.text_input("模型服务地址", value=config.base_url)
    model_name = st.text_input("模型名称", value=config.model)
    if st.button("测试模型连接", disabled=not key.strip()):
        with st.spinner("正在测试模型连接（10 秒超时，不自动重试）…"):
            try:
                probe = DeepSeekModel(ModelConfig(api_key=key, base_url=base_url, model=model_name,
                                                 timeout=10, max_retries=0, max_tokens=128))
                check_model_connection(probe)
                st.success("模型连接成功。请求已设置关闭深度思考。")
            except ModelRequestError as exc:
                st.error(describe_model_failure(exc))
            except ValueError:
                st.error("模型已返回响应，但格式不符合要求，请检查服务接口是否兼容。")
            except Exception as exc:
                st.error(describe_model_failure(exc))
    max_attempts = st.slider("最多修复尝试", 1, 8, 8)
    st.caption("最多完整复审 3 次；确认问题可继续修复，固定测试不变。")
    if st.button("检查运行环境"):
        try:
            environment = TestRunner().preflight()
            st.success("运行环境检查通过，可以执行隔离测试。")
            st.caption(f"Python {environment.get('python', '已核对')} · pytest {environment.get('pytest', '已核对')} · 断网隔离运行")
        except Exception:
            st.error("运行环境尚未准备好。请安装并启动 Docker、准备项目测试镜像，再重新检查；系统不会直接在你的电脑上运行上传的程序。")
mode_label = st.radio("运行模式", ["自动修复", "仅审查"], horizontal=True)
mode = "run" if mode_label == "自动修复" else "review"
EXAMPLES = {"自行输入": "使用自己的代码", "division": "除法程序：修复分母为零的处理",
            "mutable_default": "列表程序：修复多次调用互相影响的问题", "clean": "加法程序：检查无需修复的代码"}
AUTO_TESTS = "让系统自动生成检查用例"
PROVIDED_TESTS = "使用现成的检查用例"
example = st.selectbox("选择演示样例或使用自己的代码", list(EXAMPLES), format_func=EXAMPLES.get)
if st.session_state.get("loaded_example") != example:
    st.session_state.loaded_example = example
    st.session_state.input_revision = st.session_state.get("input_revision", 0) + 1
    for field in ("source", "tests", "requirement"):
        st.session_state[field] = ""
        st.session_state.pop("upload_token_" + field, None)
    if example != "自行输入":
        folder = Path(__file__).parent / "examples" / example
        for field, filename in (("source", "solution.py"), ("tests", "test_solution.py")):
            st.session_state[field] = (folder / filename).read_text(encoding="utf-8")
    st.session_state.test_policy = AUTO_TESTS if example == "自行输入" else PROVIDED_TESTS
if example != "自行输入":
    st.success("样例的代码和检查用例已全部填好，无需上传任何文件。")
    st.caption("可以直接使用下面的样例。填写左侧模型访问密钥后，点击“开始任务”；高级选项保持默认即可。")
else:
    st.caption("上传代码或直接粘贴即可开始，无需填写功能描述。检查用例可以由系统生成。")
values = {"requirement": ""}
invalid_upload = False


def input_field(field):
    global invalid_upload
    if field == "tests" or field == "source" and example == "自行输入":
        label = "上传自己的代码（也可在下方粘贴）" if field == "source" else "上传检查用例（也可在下方粘贴）"
        upload = st.file_uploader(label, type=["py"], key=f"upload_{field}_{st.session_state.input_revision}")
        if upload is not None:
            token = digest(upload.getvalue().hex())
            if st.session_state.get("upload_token_" + field) != token:
                try:
                    data = upload.getvalue()
                    if len(data) > 102400:
                        raise ValueError("文件超限")
                    st.session_state[field] = data.decode("utf-8")
                    st.session_state["upload_token_" + field] = token
                except (ValueError, UnicodeError):
                    invalid_upload = True
                    st.error("文件必须为受支持大小的 UTF-8 文本")
    label = {"source": "待检查的代码", "tests": "检查用例代码"}[field]
    values[field] = st.text_area(label, key=field, height=300)


input_field("source")
values["tests"] = ""
if mode == "run":
    with st.expander("高级选项：如何检查修复结果"):
        st.caption("检查用例是一段自动检查程序，例如检查分母为零时是否正确报错。它不是你要修复的业务代码。")
        policy = st.radio("选择检查方式", [AUTO_TESTS, PROVIDED_TESTS], key="test_policy")
        if policy == PROVIDED_TESTS:
            if example != "自行输入":
                st.caption("下方已经填好这个样例的检查用例，无需再上传。")
            else:
                st.caption("只有已有检查用例时才选这一项；没有的话，请选择让系统自动生成。")
            input_field("tests")
        else:
            st.caption("不用填写或上传检查用例。系统先完整审查代码，再根据源码和已确认的问题生成检查；无法判断的业务规则会保留待确认。")
    if policy == AUTO_TESTS:
        st.info("检查方式：系统自动生成。生成的检查可能不完整，结果页会提供查看和下载。")
    elif values["tests"].strip():
        st.info("检查方式：使用样例自带或你提供的检查用例。高级选项无需再操作。")
# Fingerprint contains inputs/configuration, never the key.
fingerprint = digest(repr((mode, values, base_url, model_name, max_attempts, policy if mode == "run" else "仅审查")))
busy = bool(st.session_state.get("thread") and st.session_state.thread.is_alive())
missing = []
if busy:
    missing.append("上一项任务还在运行，请等待完成")
if not key.strip():
    missing.append("请在左侧填写模型访问密钥（演示样例也需要真实模型）")
if not values["source"].strip():
    missing.append("请上传或粘贴代码，或者选择一个演示样例")
if mode == "run" and policy == PROVIDED_TESTS and not values["tests"].strip():
    missing.append("现成检查用例为空；请填写，或在高级选项中选择让系统自动生成")
if invalid_upload:
    missing.append("上传文件无效，请换成有效的代码文件")
can_start = not missing
if missing:
    st.warning("暂时不能开始：" + "；".join(missing))
if st.button("开始任务", disabled=busy or not can_start, type="primary"):
    st.session_state.result = None
    st.session_state.error = ""
    st.session_state.events = []
    st.session_state.task_fingerprint = fingerprint
    try:
        inputs = Inputs(mode=mode, max_attempts=max_attempts, **values)
        remove_previous_run(st.session_state.get("directory"))
        st.session_state.pop("directory", None)
        messages = queue.Queue()
        st.session_state.messages = messages
        thread = threading.Thread(target=worker, args=(inputs, ModelConfig(api_key=key, base_url=base_url, model=model_name), messages), daemon=True)
        st.session_state.thread = thread
        thread.start()
    except Exception as exc:
        st.session_state.error = f"输入校验失败（{type(exc).__name__}）；请检查源码和检查用例限制"


@st.fragment(run_every=.5)
def render_progress(current_fingerprint):
    messages = st.session_state.get("messages")
    if messages:
        while True:
            try:
                kind, value = messages.get_nowait()
            except queue.Empty:
                break
            if kind == "event":
                st.session_state.events = [*st.session_state.get("events", []), value]
            elif kind == "directory":
                st.session_state.directory = value
            elif kind == "result":
                st.session_state.result, st.session_state.directory = value
            else:
                st.session_state.result = None
                st.session_state.error = value
    events = st.session_state.get("events", [])
    if events:
        st.write("任务进度")
        st.dataframe([{"阶段": STAGES.get(e["node"], e["node"]),
                       "状态": {"start": "开始", "complete": "完成", "failed": "失败"}[e["phase"]],
                       "修复次数": e["attempt"], "说明": "本阶段未完成，具体原因见任务结论" if e["phase"] == "failed" else "任务未完成，具体原因见下方结论" if "阶段失败" in e["message"] else STAGES.get(e["message"], e["message"]),
                       "耗时（毫秒）": e["elapsed_ms"]} for e in events], width="stretch")
    if st.session_state.get("error"):
        st.error(st.session_state.error)
    result = st.session_state.get("result")
    if result is None:
        if st.session_state.get("thread") and st.session_state.thread.is_alive():
            st.info("任务运行中；页面按真实节点事件刷新。任务完成后可开始下一次。")
        return
    if current_fingerprint != st.session_state.get("task_fingerprint"):
        st.warning("输入或配置已改变，以下任务结果已过期；请启动新任务。")
        return
    st.subheader({"verified": "修复已通过检查", "no_repair_needed": "检查通过，无需修复",
                  "needs_attention": "仍有问题需要处理", "error": "任务未完成", "review_completed": "代码审查已完成"}[result.status])
    reason = result.reason
    if result.status == "error":
        stage = next((e.node for e in reversed(result.events) if e.phase == "failed"), "")
        if "ModelRequestError" in result.reason:
            reason = result.reason.split("；", 1)[1] if "；" in result.reason else "模型请求失败，未取得有效响应。请先点击左侧“测试模型连接”，检查服务地址、模型名称和网络。"
        else:
            reason = {"preflight": "检查程序的运行环境未准备好，请先点击左侧“检查运行环境”。",
                  "generate_tests": "检查用例生成阶段未完成，请查看模型调用记录，或先点击左侧“测试模型连接”。",
                  "initial_review": "代码审查未能完成，请检查模型访问密钥、服务地址及网络连接。",
                  "fixer": "模型未能提交有效修改。具体失败阶段已记录在报告中。"}.get(stage, "任务未能完成，请查看下方详细报告。")
    st.write(reason)
    if result.tests_origin == "generated":
        st.warning("本次使用系统生成的测试；通过表示满足这些测试，不等同于用户指定验收或完整正确性。"
                   if result.generated_tests else "尚未生成检查用例，也未开始修复。已完成的代码审查会保留在“修复前的问题”中。")
    tabs = st.tabs(["修复前的问题", "代码", "修改过程", "修复后的情况", "结束原因", "模型调用记录"])
    output_code = result.accepted_candidate or result.partial_candidate or latest_attempt_candidate(result)
    with tabs[0]:
        if result.initial_review:
            st.markdown("**原始代码发现的问题**")
            show_findings(result.initial_review.findings, "完整初审未发现已确认的问题。")
            if result.initial_review.pending_findings:
                st.markdown("**需要人工确认**")
                show_findings(result.initial_review.pending_findings, "")
            for limitation in result.initial_review.limitations:
                st.warning(limitation)
        else:
            st.info("代码审查尚未完成，暂时没有审查结论。")
        if result.baseline_test:
            show_test_result(result.baseline_test)
        elif result.mode == "review":
            st.info("本次选择了仅审查，不运行检查用例。")
        else:
            st.info("尚未执行修复前的检查：任务在准备阶段已停止。这次仍是自动修复模式。")
    with tabs[1]:
        if output_code:
            st.markdown("**当前保留的代码**" if result.accepted_candidate or result.partial_candidate else "**最后尝试代码（检查未通过，未采用）**")
            st.code(output_code.code, language="python")
        else:
            st.info("没有生成修改版本。")
        with st.expander("查看原始代码", expanded=not bool(output_code)):
            st.code(values["source"], language="python")
    with tabs[2]:
        if result.accepted_candidate:
            st.markdown("**最终采用的修改**")
            st.code(result.accepted_candidate.diff, language="diff")
            with st.expander("查看修复后的完整代码"):
                st.code(result.accepted_candidate.code, language="python")
        elif output_code:
            st.markdown("**最新保留的部分修改**" if result.partial_candidate else "**最后尝试的修改（未被采用）**")
            st.warning("该版本尚未通过全部最终核验，请结合剩余问题和测试结果使用。" if result.partial_candidate
                       else "该版本被检查拒绝或已回退，仅供查看修改；不能当作通过检查的版本。")
            st.code(output_code.diff, language="diff")
            with st.expander("查看当前输出的完整代码"):
                st.code(output_code.code, language="python")
        elif not result.attempts:
            st.info("本次没有提出修改。")
        else:
            st.info("本次尚无通过最终检查的修改，原始代码未被替换。")
        for index, attempt in enumerate(result.attempts, 1):
            with st.expander(f"尝试 {index}：{attempt.reason}"):
                show_attempt(attempt)
                if attempt.static and attempt.static.candidate_code is not None:
                    st.download_button("下载本次尝试代码（过程记录）", attempt.static.candidate_code,
                                       file_name=f"attempt_{index}.py", key=f"candidate_{index}")
    with tabs[3]:
        if result.final_review:
            st.markdown("**当前保留代码的复审结果**")
            show_findings(result.final_review.findings, "本次复审未发现已确认的问题。")
            if result.final_review.pending_findings:
                st.markdown("**需要人工确认，未自动修改**")
                show_findings(result.final_review.pending_findings, "")
        elif result.partial_candidate:
            st.warning("部分修改版尚未完成复审，不能把原始问题直接当作修改后仍存在的问题。")
        st.write("测试来源：" + {"user": "用户提供", "generated": "系统生成并固定", "none": "未执行测试"}[result.tests_origin])
        if result.generated_tests:
            st.write("检查范围：" + result.generated_tests.rationale)
            with st.expander("查看系统生成的检查用例"):
                st.code(result.generated_tests.tests, language="python")
                st.download_button("下载本次生成的固定测试", result.generated_tests.tests, "test_solution.py")
        executed = [(i, a) for i, a in enumerate(result.attempts, 1) if a.tests]
        if executed:
            for index, attempt in executed:
                with st.expander(f"第 {index} 次修改后的检查：{TEST_STATUS[attempt.tests.status]}", expanded=index == executed[-1][0]):
                    show_test_result(attempt.tests)
        else:
            st.info("尚未执行修改后的检查。")
    with tabs[4]:
        st.markdown("**任务结束原因**")
        st.write(result.reason)
        if result.attempts:
            st.caption(f"已尝试修复 {len(result.attempts)} 次，其中 {sum(a.retained for a in result.attempts)} 次修改被保留。")
        else:
            st.caption("本次未进入修改阶段；没有修复后的代码。")
        unresolved_choices = list(dict.fromkeys(info for a in result.attempts if a.proposal
                                                for info in a.proposal.missing_information))
        if unresolved_choices:
            st.markdown("**缺少约定，未自动修改的事项**")
            for info in unresolved_choices:
                st.write("• " + info)
        st.write("输出结果：" + ("修复后的代码已通过最终接受检查，可下载。" if result.accepted_candidate else
                 "已输出最新保留的部分修改版；仍有问题或核验未完成，详情见下方。" if result.partial_candidate else
                 "已输出最后尝试版；该版本未通过检查、未被采用，供查看修改。" if output_code else "本次没有生成修改代码。"))
        st.dataframe([{"核对项目": "修复目标", "结论": "全部获支持" if result.goals_verified else "尚未全部核实" if result.fix_assessment else "未执行目标核对"},
                      {"核对项目": "固定测试", "结论": "完整通过" if result.tests_passed else "尚未完整通过"},
                      {"核对项目": "代码审查", "结论": "覆盖完整" if result.review_complete else "尚未完成或覆盖受限"}], hide_index=True, width="stretch")
        if result.fix_assessment:
            descriptions = {(a.task.task_id, g.goal_id): g.description for a in result.attempts for g in a.task.goals}
            st.markdown("**逐项核对**")
            for item in result.fix_assessment.assessments:
                title = descriptions.get((item.task_id, item.goal_id), "修复目标")
                st.write(title + "：" + {"resolved": "已解决", "unresolved": "未解决", "uncertain": "仍需确认"}[item.verdict])
                st.caption(item.reason)
        if result.outstanding_findings:
            st.markdown("**仍未解决的问题**" if result.final_review or not result.partial_candidate
                        else "**初审发现的问题（修改后尚未复核）**")
            show_findings(result.outstanding_findings, "")
        if result.manual_items:
            st.markdown("**需要人工确认的事项**")
            show_findings(result.manual_items, "")
        if result.limitations:
            st.markdown("**本次检查的限制**")
            for limitation in dict.fromkeys(result.limitations):
                st.write("• " + limitation)
    with tabs[5]:
        if result.model_calls:
            st.caption(f"共 {len(result.model_calls)} 次模型调用，{result.http_attempts} 次网络请求（含重试）；未知用量显示为“未返回”。")
            rows = []
            for index, call in enumerate(result.model_calls, 1):
                stage, _, substage = call.stage.partition("/")
                detail = {"initial": "初步检查", "reflection": "独立复核", "reflection_repair": "修正复核格式", "format_repair": "修正输出格式"}.get(substage, "")
                rows.append({"次数": index, "阶段": STAGES.get(stage, stage) + (" · " + detail if detail else ""),
                             "结果": "请求失败" if call.failed else "已返回", "耗时（秒）": round(call.elapsed_ms / 1000, 2),
                             "输入用量": str(call.input_tokens) if call.input_tokens is not None else "未返回",
                             "输出用量": str(call.output_tokens) if call.output_tokens is not None else "未返回"})
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.info("本次尚未调用模型。")
    if result.accepted_candidate:
        st.download_button("下载修复后的代码（已通过检查）", result.accepted_candidate.code, "accepted_solution.py", type="primary")
    elif result.partial_candidate:
        st.download_button("下载部分修改版（尚未全部通过）", result.partial_candidate.code, "partial_solution.py", type="primary")
    elif output_code:
        st.download_button("下载最后尝试版（未通过检查、未采用）", output_code.code, "last_attempt_solution.py")
    with st.expander("下载报告与原始代码"):
        st.caption("完整报告包含测试证据、内部编号和快照标识，供进一步核对。")
        st.download_button("下载可读报告", markdown(result), "report.md", mime="text/markdown")
        st.download_button("下载完整数据报告（JSON）", result.model_dump_json(indent=2), "report.json", mime="application/json")
        st.download_button("下载原始代码", values["source"], "original_solution.py")


render_progress(fingerprint)
