import argparse
import importlib.metadata
import json
import sys

from review_agent.config import ModelConfig
from review_agent.llm import DeepSeekModel
from schemas import Inputs
from validation.test_runner import TestRunner
from workflow.graph import Workflow


def read_input(path, limit=102400):
    with open(path, "rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("输入超过字节限制")
    return data.decode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="CodeFlow：审查、局部修复及隔离验证")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="检查依赖及固定隔离环境；不调用模型")
    for name in ("review", "run"):
        p = sub.add_parser(name)
        p.add_argument("--source", required=True)
        p.add_argument("--requirement", help="可选：额外行为约定；省略时按原审查 Agent 全面检查")
        p.add_argument("--output-dir", help="显式保存运行记录；默认运行结束后清理")
        if name == "run":
            p.add_argument("--tests", help="可选：固定检查用例；省略时在完整审查后生成一次并固定")
            p.add_argument("--max-attempts", type=int, default=8, choices=range(1, 9))
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            versions = {n: importlib.metadata.version(n) for n in
                        ("pydantic", "langgraph", "langchain-openai", "streamlit", "ruff", "pytest")}
            try:
                environment = TestRunner().preflight()
            except Exception as exc:
                print(json.dumps({"status": "error", "dependencies": versions,
                                  "reason": "隔离环境预检失败", "error_type": type(exc).__name__,
                                  "action": "安装并启动 Docker，显式构建 Dockerfile.test，检查 cgroup v2 与资源限制"}, ensure_ascii=False))
                return 2
            print(json.dumps({"status": "passed", "dependencies": versions, "environment": environment}, ensure_ascii=False))
            return 0
        inputs = Inputs(mode=args.command, source=read_input(args.source),
                        tests=read_input(args.tests) if args.command == "run" and args.tests else "",
                        requirement=read_input(args.requirement, 4000) if args.requirement else "",
                        max_attempts=getattr(args, "max_attempts", 8))
        config = ModelConfig.from_env()
        if not config.api_key.strip():
            print(json.dumps({"status": "error", "reason": "需要 DEEPSEEK_API_KEY；正常入口使用真实模型"}, ensure_ascii=False))
            return 2
        workflow = Workflow(DeepSeekModel(config), config.model,
                            progress_callback=lambda e: print(f'{e["node"]}/{e["phase"]}: {e["message"]}', file=sys.stderr))
        report = workflow.run(inputs, args.output_dir)
        print(json.dumps({"run_id": report.run_id, "status": report.status, "reason": report.reason,
                          "tests_origin": report.tests_origin,
                          "report": str(workflow.directory / "result" / "report.json") if args.output_dir else None,
                          **({"result": report.model_dump(mode="json")} if not args.output_dir else {})}, ensure_ascii=False))
        return 0 if report.status in ("verified", "no_repair_needed", "review_completed") else 1 if report.status == "needs_attention" else 2
    except Exception as exc:
        print(json.dumps({"status": "error", "reason": "输入、配置或任务创建失败", "error_type": type(exc).__name__}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
