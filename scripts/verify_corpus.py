"""验收脚本：检查语料库与向量索引是否就绪。

用法：
    python -m scripts.verify_corpus

通过条件：
  - clean 目录非空
  - 向量库 count > 0
  - 三个验证 query 均能在 top-1 命中合理主题（按 tags_topic 弱校验）
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from shader_agent.config.settings import settings
from shader_agent.corpus.vector_store import ShaderVectorStore

console = Console()


def section(t: str) -> None:
    console.rule(f"[bold cyan]{t}[/bold cyan]")


VERIFY_CASES = [
    {
        "query": "raymarching a signed distance sphere",
        "expect_any_tag": {"raymarching", "sdf"},
    },
    {
        "query": "value noise 2d pattern with hash",
        "expect_any_tag": {"noise", "2d-pattern"},
    },
    {
        "query": "mandelbrot fractal escape time coloring",
        "expect_any_tag": {"fractal"},
    },
]


def main() -> int:
    section("1/3  Clean dir")
    clean_dir = settings.corpus_clean_dir
    files = list(clean_dir.glob("*.json")) if clean_dir.exists() else []
    console.print(f"clean_dir = {clean_dir}")
    console.print(f"clean records on disk = [bold]{len(files)}[/bold]")
    if not files:
        console.print("[red]FAIL[/red] clean 目录为空，先跑 build_corpus")
        return 1

    section("2/3  Vector store")
    vstore = ShaderVectorStore()
    total = vstore.count()
    console.print(f"vector collection count = [bold]{total}[/bold]")
    if total == 0:
        console.print("[red]FAIL[/red] 向量库为空")
        return 1

    section("3/3  Smoke retrieval")
    passes = 0
    for case in VERIFY_CASES:
        q = case["query"]
        expect = case["expect_any_tag"]
        results = vstore.query_by_text(q, top_k=3)
        console.print(f"\n[bold]Query:[/bold] {q}")
        if not results:
            console.print("  [red]no hits[/red]")
            continue
        top = results[0]
        md = top["metadata"]
        top_tags = set((md.get("tags_topic") or "").split(","))
        ok = bool(top_tags & expect)
        tag_str = ",".join(sorted(top_tags))
        marker = "[green]PASS[/green]" if ok else "[red]MISS[/red]"
        console.print(
            f"  top1: {md.get('name','?')} "
            f"[dim]tags=({tag_str}) dist={top['distance']:.4f}[/dim] {marker}"
        )
        for i, r in enumerate(results[1:], 2):
            mdi = r["metadata"]
            console.print(
                f"  top{i}: {mdi.get('name','?')} "
                f"[dim]tags=({mdi.get('tags_topic','')}) dist={r['distance']:.4f}[/dim]"
            )
        if ok:
            passes += 1

    console.print()
    if passes == len(VERIFY_CASES):
        console.print(Panel.fit("[bold green]ALL PASS[/bold green]", border_style="green"))
        return 0
    console.print(
        Panel.fit(
            f"[bold yellow]{passes}/{len(VERIFY_CASES)} passed[/bold yellow]\n"
            "提示：若大部分查询命中相邻主题（例如查 'raymarching' 命中了 sdf-only），"
            "说明语料覆盖偏窄，建议补充 SHADERTOY_API_KEY 后重跑 build_corpus。",
            border_style="yellow",
        )
    )
    return 1 if passes == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
