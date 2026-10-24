from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import MAX_SOURCE_BYTES, ModelConfig
from .graph import ReviewAgent
from .llm import DeepSeekModel
from .reporting import report_json, report_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m review_agent", description="TraceReview Python 代码审查 Agent")
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("review", help="审查一个 Python 文件")
    review.add_argument("path")
    review.add_argument("--focus", default="全面检查正确性、异常处理、资源和安全")
    review.add_argument("--format", choices=("markdown", "json"), default="markdown")
    review.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ModelConfig.from_env()
    if not config.api_key:
        print("错误：请设置 DEEPSEEK_API_KEY。", file=sys.stderr)
        return 2
    path = Path(args.path)
    try:
        with path.open("rb") as source_file:
            raw = source_file.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES:
            raise ValueError("源码 UTF-8 大小超过 100 KB（102400 字节）")
        code = raw.decode("utf-8")
        report = ReviewAgent(DeepSeekModel(config), config.model).review(code, path.name, args.focus)
        output = report_json(report) if args.format == "json" else report_markdown(report)
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
        else:
            print(output)
        return 0
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
