"""离线评估入口（DeepEval + Langfuse 回流）。

用法：
    # 全量评估（含 LLM 裁判，会真调 DeepSeek）
    python -m scripts.run_eval

    # 只跑确定性指标：零 LLM 成本、零方差，适合 CI 门禁
    python -m scripts.run_eval --no-judge

    # 只评检索链路（不生成，最快）
    python -m scripts.run_eval --tasks retrieval

    # 每类只跑前 2 条，快速冒烟
    python -m scripts.run_eval --limit 2

    # CI 门禁：通过率低于 0.8 时以非零码退出
    python -m scripts.run_eval --no-judge --min-pass-rate 0.8

产物：
    data/reports/eval_{ts}/report.md
    data/reports/eval_{ts}/report.json

若配置了 Langfuse，每条用例会生成独立 trace，且指标分数按 trace_id 回流，
可在看板中把"质量分 / 耗时 / token / 检索命中"放在同一视图对比。
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shader_agent.config.settings import settings
from shader_agent.eval import EvalRunner, save_report, summary
from shader_agent.observability import is_enabled as lf_enabled

console = Console()


def section(t: str) -> None:
    console.rule(f"[bold cyan]{t}[/bold cyan]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shader Agent 离线评估")
    p.add_argument("--tasks", default="retrieval,generation,analysis",
                   help="逗号分隔：retrieval / generation / analysis")
    p.add_argument("--no-judge", action="store_true",
                   help="关闭 LLM-as-a-judge，只跑确定性指标（零 LLM 成本）")
    p.add_argument("--limit", type=int, default=0,
                   help="每类任务只跑前 N 条 golden（0=全部）")
    p.add_argument("--render", default="auto",
                   choices=["auto", "mock", "real"],
                   help="渲染/编译后端；real 用于验证真实 GLSL 编译")
    p.add_argument("--no-vector-store", action="store_true",
                   help="关闭向量库（用于对比『有无 RAG』的质量差异）")
    p.add_argument("--max-fix-loops", type=int, default=1)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--min-pass-rate", type=float, default=-1.0,
                   help="CI 门禁：通过率低于该值则退出码为 1（默认不设门禁）")
    p.add_argument("--name", default="eval", help="报告目录前缀")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())

    section("环境检查")
    ds = summary()
    console.print(f"  数据集         = {ds['total']} 条 "
                  f"(检索 {ds['retrieval']} / 生成 {ds['generation']} / 分析 {ds['analysis']})")
    console.print(f"  任务           = {', '.join(tasks)}")
    console.print(f"  LLM 裁判       = {'OFF（仅确定性指标）' if args.no_judge else 'ON'}")
    console.print(f"  渲染后端       = {args.render}")
    console.print(f"  向量库         = {'off' if args.no_vector_store else 'auto'}")
    console.print(f"  Langfuse       = {'已启用' if lf_enabled() else '未启用（分数不回流）'}")
    console.print(f"  chat_model     = {settings.llm.chat_model}")

    has_key = bool(settings.deepseek_api_key) and not settings.deepseek_api_key.startswith("sk-your")
    needs_llm = ("generation" in tasks or "analysis" in tasks)
    if needs_llm and not has_key:
        console.print("[red]FAIL[/red] generation/analysis 任务需要 DEEPSEEK_API_KEY")
        console.print("[yellow]提示：可先跑 `--tasks retrieval --no-judge` 验证检索链路[/yellow]")
        return 1

    section("开始评估")
    try:
        runner = EvalRunner(
            with_llm_judge=not args.no_judge,
            render_backend=args.render,
            use_vector_store="off" if args.no_vector_store else "auto",
            max_fix_loops=args.max_fix_loops,
            top_k=args.top_k,
        )
    except Exception as e:
        console.print(f"[red]FAIL[/red] 评估装配失败: {type(e).__name__}: {e}")
        return 1

    report = runner.run(tasks=tasks, limit=args.limit)

    section("指标汇总")
    agg = report.aggregate()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("指标"); table.add_column("均值", justify="right")
    table.add_column("最小", justify="right"); table.add_column("最大", justify="right")
    table.add_column("样本", justify="right")
    for k, v in agg["by_metric"].items():
        table.add_row(k, f"{v['mean']:.3f}", f"{v['min']:.3f}",
                      f"{v['max']:.3f}", str(v["n"]))
    console.print(table)

    section("用例结果")
    for c in report.cases:
        flag = "[green]PASS[/green]" if c.passed else "[red]FAIL[/red]"
        console.print(f"  {flag}  {c.case_id:<24} {c.task:<11} "
                      f"score={c.mean_score:.3f}  {c.elapsed_ms:.0f}ms"
                      + (f"  err={c.error[:60]}" if c.error else ""))

    section("落盘")
    out_dir = save_report(report, name=args.name)
    console.print(f"报告: {out_dir / 'report.md'}")
    console.print(f"原始: {out_dir / 'report.json'}")

    section("结论")
    console.print(f"  通过率     = {agg['pass_rate']:.1%} "
                  f"({agg['n_passed']}/{agg['n_cases']})")
    console.print(f"  平均分     = {agg['mean_score']:.3f}")
    console.print(f"  平均耗时   = {agg['avg_latency_ms']:.0f} ms")

    if args.min_pass_rate >= 0:
        if agg["pass_rate"] < args.min_pass_rate:
            console.print(Panel.fit(
                f"[bold red]GATE FAILED[/bold red] "
                f"通过率 {agg['pass_rate']:.1%} < 门槛 {args.min_pass_rate:.1%}",
                border_style="red"))
            return 1
        console.print(Panel.fit("[bold green]GATE PASSED[/bold green]",
                                border_style="green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
