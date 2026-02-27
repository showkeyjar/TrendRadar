# coding=utf-8
"""
行业情报 CLI
"""

import argparse
from pathlib import Path

from .feedback import record_feedback
from .pipeline import build_intel_digest, dump_results_json, run_intel_pipeline


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def cmd_run(args: argparse.Namespace) -> int:
    results = run_intel_pipeline(
        profile_file=args.config,
        feedback_file=args.feedback_file,
        top_k=args.top,
    )
    md_content = build_intel_digest(results, output_file=args.output)
    if args.json_output:
        dump_results_json(results, args.json_output)
    print(md_content)
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    state = record_feedback(
        state_file=args.feedback_file,
        rating=args.rating,
        keywords=_split_csv(args.keywords),
        source_names=_split_csv(args.sources),
        add_keywords=_split_csv(args.add_keywords),
        block_keywords=_split_csv(args.block_keywords),
        note=args.note,
    )
    print(f"反馈已记录，更新时间: {state.get('updated_at', '')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TrendRadar 行业情报工具")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="运行情报打分并输出简报")
    run_parser.add_argument("--config", default="config/intelligence.yaml", help="行业情报配置文件")
    run_parser.add_argument("--feedback-file", default="output/intel/feedback_state.json", help="反馈状态文件")
    run_parser.add_argument("--top", type=int, default=30, help="输出前 N 条")
    run_parser.add_argument("--output", default="", help="Markdown 输出文件")
    run_parser.add_argument("--json-output", default="", help="JSON 输出文件")
    run_parser.set_defaults(func=cmd_run)

    fb_parser = sub.add_parser("feedback", help="记录用户反馈并更新学习状态")
    fb_parser.add_argument("--feedback-file", default="output/intel/feedback_state.json", help="反馈状态文件")
    fb_parser.add_argument("--rating", type=int, required=True, help="反馈评分 (1-5)")
    fb_parser.add_argument("--keywords", default="", help="命中的关键词，逗号分隔")
    fb_parser.add_argument("--sources", default="", help="来源名称，逗号分隔")
    fb_parser.add_argument("--add-keywords", default="", help="新增关键词，逗号分隔")
    fb_parser.add_argument("--block-keywords", default="", help="屏蔽关键词，逗号分隔")
    fb_parser.add_argument("--note", default="", help="备注")
    fb_parser.set_defaults(func=cmd_feedback)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    if args.command == "run" and not args.output:
        default_file = Path("output/intel") / "latest_digest.md"
        args.output = str(default_file)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

