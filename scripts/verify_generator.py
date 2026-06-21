"""Generator 端到端验证（真调 DeepSeek）。

用法：
    python -m scripts.verify_generator
    python -m scripts.verify_generator --no-cache
    python -m scripts.verify_generator --critique
    python -m scripts.verify_generator --case "画一个程序生成的霓虹蓝紫万花筒，带时间动画"
    python -m scripts.verify_generator --combined  # 走 analyze_then_generate

通过条件：
  - draft 的代码含 mainImage 签名
  - validate_code 报告 ok=True 或经修正后变为 True
  - explanation 非空
  - 若启用 self_critique，score >= 0.5

产物：
  - data/reports/generator_{slug}_{ts}.md
"""
from __future__ import annotations

import argparse
import re
import sys
import time

from rich.console import Console
from rich.panel import Panel

from shader_agent.agents.analyzer import ShaderAnalyzer
from shader_agent.agents.generator import ShaderGenerator
from shader_agent.agents.orchestrator import Orchestrator
from shader_agent.agents.schemas import Message
from shader_agent.config.settings import settings
from shader_agent.corpus.seed_shaders import get_seed_shaders
from shader_agent.llm.llm_fn import (
    make_chat_fn,
    make_code_fn,
    make_json_fn,
    stats,
)

console = Console()


DEFAULT_CASES: dict[str, str] = {
    "neon_kaleido": "画一个程序生成的霓虹蓝紫万花筒，带时间动画，6 折对称",
    "raymarch_blob": "raymarching 一个软融合的两个球（smin），中等复杂度，冷色调",
    "noise_water": "用 fbm 噪声画一个水波纹效果，简单，蓝绿色",
}


def section(t: str) -> None:
    console.rule(f"[bold cyan]{t}[/bold cyan]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="neon_kaleido",
                   help=f"预设 case 名 {list(DEFAULT_CASES)} 或自定义中文 prompt")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--critique", action="store_true",
                   help="启用文本层自评")
    p.add_argument("--combined", action="store_true",
                   help="跑 analyze_then_generate 组合任务（先解读 seed03 再改写）")
    p.add_argument("--max-fix-loops", type=int, default=2)
    p.add_argument("--top-k", type=int, default=3)
    # headless 渲染
    p.add_argument("--render", action="store_true",
                   help="启用真 GL 编译器（ValidateCodeAction 走真 compiler）")
    p.add_argument("--vision-critique", action="store_true",
                   help="启用图像自评（隐含 --render + --critique）")
    p.add_argument("--save-png", action="store_true",
                   help="渲染成功后把 PNG 落盘到 data/reports/")
    return p.parse_args()


def _slugify(text: str) -> str:
    s = re.sub(r"\s+", "_", text.strip())[:30]
    s = re.sub(r"[^\w\u4e00-\u9fa5]+", "", s)
    return s or "case"


def _try_get_vector_store():
    try:
        from shader_agent.corpus.vector_store import ShaderVectorStore
        vs = ShaderVectorStore()
        return vs if vs.count() > 0 else None
    except Exception as e:
        console.print(f"[yellow]vector store unavailable: {e}[/yellow]")
        return None


def main() -> int:
    args = parse_args()
    prompt = DEFAULT_CASES.get(args.case, args.case)

    section("环境检查")
    if not settings.deepseek_api_key or settings.deepseek_api_key.startswith("sk-your"):
        console.print("[red]FAIL[/red] DEEPSEEK_API_KEY 未配置")
        return 1
    console.print(f"  base_url      = {settings.deepseek_base_url}")
    console.print(f"  chat_model    = {settings.llm.chat_model}")
    console.print(f"  coder_model   = {settings.llm.coder_model}")
    console.print(f"  cache         = {'OFF' if args.no_cache else 'ON'}")
    console.print(f"  case          = {args.case}: {prompt}")
    console.print(f"  combined      = {args.combined}")
    console.print(f"  self_critique = {args.critique}")
    console.print(f"  max_fix_loops = {args.max_fix_loops}")

    vstore = _try_get_vector_store()
    console.print(
        f"  vector_db     = "
        + (f"OK ({vstore.count()} docs)" if vstore else "empty / unavailable")
    )

    use_cache = not args.no_cache
    code_fn = make_code_fn(use_cache=use_cache)
    chat_fn = make_chat_fn(use_cache=use_cache)
    json_fn = make_json_fn(use_cache=use_cache)

    # 根据 flag 决定是否启用真渲染
    enable_render = args.render or args.vision_critique
    enable_critique = args.critique or args.vision_critique
    compiler = None
    renderer = None
    critique_fn = None

    if enable_render:
        from shader_agent.rendering import GLSLCompiler, GLSLRenderer
        compiler, reason = GLSLCompiler.try_create()
        if compiler is None:
            console.print(f"[yellow]  compiler 不可用: {reason.splitlines()[0]}[/yellow]")
            console.print("[yellow]  → 回退到静态 validate[/yellow]")
        else:
            console.print("  compiler      = real GL (moderngl)")
        renderer, rreason = GLSLRenderer.try_create()
        if renderer is None and compiler is not None:
            # compiler 在但 renderer 不在的情况几乎不会出现（共享 ctx），但稳一点
            console.print(f"[yellow]  renderer 不可用: {rreason}[/yellow]")
        elif renderer is not None:
            console.print("  renderer      = real GL")

    if args.vision_critique:
        from shader_agent.llm.llm_fn import make_vision_critique_fn
        critique_fn = make_vision_critique_fn(use_cache=use_cache)
        console.print(f"  critique_fn   = vision (with text fallback)")

    generator = ShaderGenerator(
        vector_store=vstore,
        llm_fn=code_fn,
        compiler=compiler,                    # 可注入真编译器
        renderer=renderer,                     # 可注入真渲染器
        critique_fn=critique_fn,               # 可注入多模态自评
        enable_self_critique=enable_critique,
        model_name=settings.llm.coder_model,
        max_fix_loops=args.max_fix_loops,
        top_k=args.top_k,
    )

    if args.combined:
        # 组合：先 analyze seed03 (Raymarched Sphere)，再让 Generator 在此基础上改写
        analyzer = ShaderAnalyzer(
            vector_store=vstore,
            walkthrough_llm=json_fn, summary_llm=json_fn,
            effect_llm=chat_fn, compare_llm=chat_fn, llm_fn=chat_fn,
            model_name=settings.llm.chat_model,
            top_k=args.top_k, strategy="fourstage",
        )
        orch = Orchestrator(analyzer=analyzer, generator=generator)
        seeds = get_seed_shaders()
        ref = next(s for s in seeds if s.shader_id == "seed03")
        section(f"调用 analyze_then_generate（参考: {ref.name}）")
        t0 = time.perf_counter()
        result = orch.analyze_then_generate(code=ref.code_image, ask=prompt)
        elapsed = time.perf_counter() - t0
        gen = result["generated"]
        report = result["report"]
    else:
        orch = Orchestrator(generator=generator)
        section("调用 generate_only")
        t0 = time.perf_counter()
        result = orch.generate_only(prompt)
        elapsed = time.perf_counter() - t0
        gen = result["generated"]
        report = None

    console.print(f"耗时: {elapsed:.2f}s")
    console.print(f"LLM 调用统计: {stats.snapshot()}")

    if gen is None:
        console.print("[red]FAIL[/red] 没产出 GeneratedShader")
        return 1

    section("质量门槛检查")
    checks: list[tuple[str, bool, object]] = []

    sig_re = re.compile(
        r"\bvoid\s+mainImage\s*\(\s*out\s+vec4\s+\w+\s*,\s*in\s+vec2\s+\w+\s*\)"
    )
    c1 = bool(sig_re.search(gen.code))
    checks.append(("mainImage 签名正确", c1, ""))

    c2 = gen.compile_result.ok
    checks.append(("validate ok", c2, gen.compile_result.errors[:80] if not c2 else ""))

    c3 = bool((gen.explanation or "").strip())
    checks.append(("explanation 非空", c3, len(gen.explanation or "")))

    c4 = gen.iterations >= 1
    checks.append(("iterations >= 1", c4, gen.iterations))

    if args.critique:
        c5 = gen.self_critique_score >= 0.5
        checks.append(("self_critique score >= 0.5", c5, gen.self_critique_score))

    if args.combined:
        c6 = (gen.spec is not None and gen.spec.reference_report is not None)
        checks.append(("reference_report 穿透", c6, ""))

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
    out_path = out_dir / f"generator_{_slugify(args.case)}_{ts}.md"
    png_path = None

    # 如果 renderer 可用，单独再 render 一次落盘 PNG
    if args.save_png and renderer is not None and gen.compile_result.ok:
        try:
            png_bytes = renderer.render(gen.code, width=512, height=384, time=1.5)
            png_path = out_dir / f"generator_{_slugify(args.case)}_{ts}.png"
            png_path.write_bytes(png_bytes)
            console.print(f"saved PNG: {png_path}")
        except Exception as e:
            console.print(f"[yellow]render 失败: {e}[/yellow]")

    md = "# Generation Trace\n\n"
    md += f"**prompt**: {prompt}\n\n"
    md += f"**combined**: {args.combined}\n\n"
    md += f"**render**: {args.render or args.vision_critique}\n\n"
    md += f"**elapsed**: {elapsed:.1f}s\n\n"
    if png_path is not None:
        md += f"![rendered]({png_path.name})\n\n"
    if report is not None:
        md += "---\n## Analyzer Report (前置)\n\n"
        md += report.to_markdown() + "\n"
    md += "---\n## Generated\n\n"
    md += gen.to_markdown()
    out_path.write_text(md, encoding="utf-8")
    console.print(f"saved: {out_path}")

    section("代码摘录")
    console.print(gen.code[:900] + ("\n…(略)…" if len(gen.code) > 900 else ""))

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
