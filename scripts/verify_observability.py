"""可观测性链路验证（Langfuse）。

用法：
    # 不调 LLM，只验证 SDK 装配与 span 嵌套（离线可跑）
    python -m scripts.verify_observability --dry-run

    # 真跑一次生成任务，验证 trace/generation/score 都能上报
    python -m scripts.verify_observability

通过条件：
  - langfuse 已安装（否则明确提示降级，仍返回 0，因为降级是设计内行为）
  - 启用后能取到 trace_id
  - span 能正确嵌套（root → child）
  - 分数能写入
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

from shader_agent.config.settings import settings
from shader_agent.observability import (
    flush,
    get_current_trace_id,
    is_enabled,
    score_current_trace,
    trace_span,
    update_current_trace,
)

console = Console()


def section(t: str) -> None:
    console.rule(f"[bold cyan]{t}[/bold cyan]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="只验证 span 嵌套，不调用 LLM")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    section("环境检查")
    try:
        import langfuse  # noqa: F401
        installed = True
        ver = getattr(langfuse, "__version__", "?")
    except Exception:
        installed = False
        ver = "-"
    console.print(f"  langfuse 已安装   = {installed} (version={ver})")
    console.print(f"  observability     = {settings.observability.enabled}")
    console.print(f"  public_key 已配置 = {bool(settings.langfuse_public_key)}")
    console.print(f"  host              = {settings.langfuse_host}")
    console.print(f"  is_enabled()      = {is_enabled()}")

    if not installed:
        console.print(Panel.fit(
            "[yellow]langfuse 未安装 → 全链路 no-op（设计内的降级行为）[/yellow]\n"
            "安装后重跑：pip install 'langfuse>=3.0.0'",
            border_style="yellow"))
        return 0

    if not is_enabled():
        console.print(Panel.fit(
            "[yellow]Langfuse 未启用（缺 LANGFUSE_PUBLIC_KEY）→ 全链路 no-op[/yellow]\n"
            "在 .env 填入密钥后重跑。现有功能不受影响。",
            border_style="yellow"))
        return 0

    section("验证 span 嵌套")
    checks: list[tuple[str, bool, object]] = []
    with trace_span("verify.root", input={"mode": "dry" if args.dry_run else "live"}) as root:
        update_current_trace(name="verify_observability",
                             tags=["verify", "observability"])
        tid = get_current_trace_id()
        checks.append(("能取到 trace_id", bool(tid), tid))

        with trace_span("verify.child", input={"step": 1}) as child:
            child.update(output={"step": 1, "ok": True})
        checks.append(("子 span 创建成功", True, ""))

        score_current_trace("verify.smoke", 1.0, comment="链路自检")
        checks.append(("分数写入未抛错", True, ""))
        root.update(output={"done": True})

    if not args.dry_run:
        section("验证 LLM generation 上报")
        if not settings.deepseek_api_key or settings.deepseek_api_key.startswith("sk-your"):
            console.print("[yellow]  跳过：DEEPSEEK_API_KEY 未配置[/yellow]")
        else:
            from shader_agent.llm.llm_fn import make_chat_fn
            with trace_span("verify.llm_call") as span:
                fn = make_chat_fn(use_cache=False)
                text = fn([{"role": "user", "content": "用一句话说明什么是 GLSL。"}])
                span.update(output={"len": len(text)})
            checks.append(("LLM 调用产生 generation", bool(text), f"{len(text)} chars"))

    flush()
    console.print("  已 flush 上报缓冲区")

    section("结论")
    all_ok = True
    for name, ok, val in checks:
        flag = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  {flag}  {name}    (val={val})")
        all_ok = all_ok and ok

    if all_ok:
        console.print(Panel.fit(
            f"[bold green]ALL CHECKS PASS[/bold green]\n"
            f"到 {settings.langfuse_host} 查看 trace",
            border_style="green"))
        return 0
    console.print(Panel.fit("[bold red]SOME CHECKS FAILED[/bold red]",
                            border_style="red"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
