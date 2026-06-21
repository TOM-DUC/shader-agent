"""Analyzer 端到端验证（真调 DeepSeek）。

用法：
    python -m scripts.verify_analyzer                 # 用 seed03 Raymarched Sphere
    python -m scripts.verify_analyzer --seed seed04   # 指定 seed id
    python -m scripts.verify_analyzer --no-cache      # 禁用缓存
    python -m scripts.verify_analyzer --strategy single  # 用单 prompt 对照

通过条件（与 dry-run 不同，这是质量门槛）：
  - 报告 algorithm_summary 长度 ≥ 80 字符
  - techniques 非空且全部在受控词表内
  - section_walkthrough ≥ 1 条
  - 命中 ≥ 1 个 similar shader（若向量库非空）
  - visual_effect 非空
  - 没有 "(占位)" "fallback" 字样（说明 LLM 真生效）

输出：
  - 控制台打印 Markdown 报告
  - 保存到 data/reports/analyzer_{seed_id}_{timestamp}.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from shader_agent.agents.analyzer import ShaderAnalyzer
from shader_agent.agents.actions.analyzer_actions_v2 import TECHNIQUE_VOCAB
from shader_agent.agents.orchestrator import Orchestrator
from shader_agent.agents.schemas import Message
from shader_agent.config.settings import settings
from shader_agent.corpus.seed_shaders import get_seed_shaders
from shader_agent.llm.llm_fn import make_chat_fn, make_json_fn, stats

console = Console()


def section(t: str) -> None:
    console.rule(f"[bold cyan]{t}[/bold cyan]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default="seed03",
                   help="seed shader id（默认 seed03 = Raymarched Sphere）")
    p.add_argument("--no-cache", action="store_true", help="禁用 LLM 缓存")
    p.add_argument("--strategy", choices=["single", "fourstage"], default="fourstage")
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def _try_get_vector_store():
    try:
        from shader_agent.corpus.vector_store import ShaderVectorStore
        vs = ShaderVectorStore()
        if vs.count() == 0:
            return None
        return vs
    except Exception as e:
        console.print(f"[yellow]向量库不可用: {e}[/yellow]")
        return None


def main() -> int:
    args = parse_args()

    section("环境检查")
    if not settings.deepseek_api_key or settings.deepseek_api_key == "sk-your-key-here":
        console.print("[red]FAIL[/red] DEEPSEEK_API_KEY 未配置")
        return 1
    console.print(f"  base_url    = {settings.deepseek_base_url}")
    console.print(f"  chat_model  = {settings.llm.chat_model}")
    console.print(f"  cache       = {'OFF' if args.no_cache else 'ON'}")
    console.print(f"  strategy    = {args.strategy}")

    vstore = _try_get_vector_store()
    console.print(f"  vector_db   = {'OK ({} docs)'.format(vstore.count()) if vstore else 'empty / unavailable'}")

    # 构造 LLM 函数：walkthrough/summary 用 JSON 模式，effect/compare 用普通 chat
    use_cache = not args.no_cache
    json_fn = make_json_fn(use_cache=use_cache)
    chat_fn = make_chat_fn(use_cache=use_cache)

    analyzer = ShaderAnalyzer(
        vector_store=vstore,
        walkthrough_llm=json_fn,
        summary_llm=json_fn,
        effect_llm=chat_fn,
        compare_llm=chat_fn,
        llm_fn=chat_fn,                # fallback
        model_name=settings.llm.chat_model,
        top_k=args.top_k,
        strategy=args.strategy,
    )
    orch = Orchestrator(analyzer=analyzer)

    # 取目标 shader
    seeds = get_seed_shaders()
    target = next((s for s in seeds if s.shader_id == args.seed), None)
    if target is None:
        console.print(f"[red]FAIL[/red] seed '{args.seed}' 不存在，可选: {[s.shader_id for s in seeds]}")
        return 1
    console.print(f"  target      = {target.shader_id} - {target.name}")

    section("调用 Analyzer（真 DeepSeek）")
    t0 = time.perf_counter()
    result = orch.analyze_only(target.code_image)
    elapsed = time.perf_counter() - t0
    rep = result["report"]

    console.print(f"耗时: {elapsed:.2f}s")
    console.print(f"LLM 调用统计: {stats.snapshot()}")

    if rep is None:
        console.print("[red]FAIL[/red] 没产出报告")
        return 1

    section("质量门槛检查")
    checks = []

    c1 = len((rep.algorithm_summary or "").strip()) >= 80
    checks.append(("algorithm_summary ≥ 80 chars", c1, len(rep.algorithm_summary or "")))

    c2 = bool(rep.techniques) and all(t in TECHNIQUE_VOCAB for t in rep.techniques)
    checks.append(("techniques 非空且在词表内", c2, rep.techniques))

    c3 = bool(rep.section_walkthrough) and len(rep.section_walkthrough) >= 1
    checks.append(("section_walkthrough ≥ 1 条", c3, len(rep.section_walkthrough or {})))

    if vstore is not None:
        c4 = len(rep.similar_shaders) >= 1
        checks.append(("similar_shaders ≥ 1", c4, len(rep.similar_shaders)))

    c5 = bool((rep.visual_effect or "").strip())
    checks.append(("visual_effect 非空", c5, len(rep.visual_effect or "")))

    text_blob = (rep.algorithm_summary or "") + " " + (rep.visual_effect or "") + " " + \
                " ".join(rep.section_walkthrough.values())
    c6 = "占位" not in text_blob and "fallback" not in text_blob.lower()
    checks.append(("无 fallback/占位 字样（LLM 真生效）", c6, ""))

    all_ok = True
    for name, ok, val in checks:
        flag = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  {flag}  {name}    (val={val})")
        all_ok = all_ok and ok

    # 落盘
    section("落盘 Markdown 报告")
    out_dir = settings.project_root / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"analyzer_{target.shader_id}_{ts}.md"
    out_path.write_text(rep.to_markdown(), encoding="utf-8")
    console.print(f"saved: {out_path}")

    # 摘要打印
    section("报告摘要（前 800 字）")
    md = rep.to_markdown()
    console.print(md[:800] + ("\n…(略)…" if len(md) > 800 else ""))

    section("结论")
    if all_ok:
        console.print(Panel.fit("[bold green]ALL CHECKS PASS[/bold green]",
                                border_style="green"))
        return 0
    console.print(Panel.fit("[bold red]SOME CHECKS FAILED[/bold red]",
                            border_style="red"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
