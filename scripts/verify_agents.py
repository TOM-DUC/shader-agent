"""阶段三验收脚本（dry-run，不真调 LLM 也不真编译）。

跑通：
  1) 仅 Analyzer：用 seed03 (Raymarched Sphere) 走完 parse → retrieve → explain(fallback) → synthesize
  2) 仅 Generator：用一句中文需求走完 parse_spec → retrieve_examples → draft_code(stub) → validate
  3) Analyze-then-Generate：拼接两端，验证 AnalysisReport 作为 reference_report 被消费

用法（项目根）：
    python -m scripts.verify_agents

通过条件：
  - 三个任务均返回非 None 的 report / generated；
  - generated.code 含 mainImage；
  - generated.spec.reference_report 在 task3 中非空。
"""
from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from shader_agent.agents.analyzer import ShaderAnalyzer
from shader_agent.agents.generator import ShaderGenerator
from shader_agent.agents.orchestrator import Orchestrator
from shader_agent.config.settings import settings
from shader_agent.corpus.seed_shaders import get_seed_shaders

console = Console()


def section(t: str) -> None:
    console.rule(f"[bold cyan]{t}[/bold cyan]")


def _try_get_vector_store():
    """尝试加载已有的向量库；若不可用则返回 None（走无检索分支）。"""
    try:
        from shader_agent.corpus.vector_store import ShaderVectorStore
        vs = ShaderVectorStore()
        if vs.count() == 0:
            console.print("[yellow]向量库为空，跳过 retrieve 步骤。[/yellow]")
            return None
        return vs
    except Exception as e:
        console.print(f"[yellow]无法加载向量库（{e}），跳过 retrieve 步骤。[/yellow]")
        return None


def main() -> int:
    vstore = _try_get_vector_store()

    analyzer = ShaderAnalyzer(vector_store=vstore, llm_fn=None, top_k=3)
    generator = ShaderGenerator(vector_store=vstore, llm_fn=None, max_fix_loops=1)
    orch = Orchestrator(analyzer=analyzer, generator=generator)

    seeds = get_seed_shaders()
    # 找一个 raymarching seed
    sphere = next((s for s in seeds if s.name == "Raymarched Sphere"), seeds[0])

    # ---------- task 1 ----------
    section("1/3  analyze_only (Raymarched Sphere)")
    r1 = orch.analyze_only(sphere.code_image)
    rep = r1["report"]
    assert rep is not None, "no report produced"
    console.print(f"[green]OK[/green] elapsed={r1['elapsed_ms']:.1f}ms")
    console.print(f"  techniques: {rep.techniques}")
    console.print(f"  custom_funcs (from parse): inferred from explain={list(rep.key_variables)[:5]}")
    console.print(f"  similar count: {len(rep.similar_shaders)}")

    # ---------- task 2 ----------
    section("2/3  generate_only (中文 prompt)")
    r2 = orch.generate_only("画一个带光照的 raymarching 球，使用冷色调")
    gen = r2["generated"]
    assert gen is not None, "no generated"
    assert "mainImage" in gen.code, "generated code missing mainImage"
    console.print(f"[green]OK[/green] elapsed={r2['elapsed_ms']:.1f}ms")
    console.print(f"  spec.effect_type = {gen.spec.effect_type if gen.spec else '?'}")
    console.print(f"  spec.palette     = {gen.spec.palette if gen.spec else '?'}")
    console.print(f"  iterations       = {gen.iterations}")
    console.print(f"  compile_ok       = {gen.compile_result.ok}")
    console.print(f"  code excerpt: {gen.code[:120].replace(chr(10),' ')}")

    # ---------- task 3 ----------
    section("3/3  analyze_then_generate（关键路径）")
    r3 = orch.analyze_then_generate(
        code=sphere.code_image,
        ask="保持算法不变，把球的颜色改成霓虹紫，加一点呼吸动画",
    )
    rep3 = r3["report"]
    gen3 = r3["generated"]
    assert rep3 is not None and gen3 is not None
    assert gen3.spec is not None and gen3.spec.reference_report is not None, \
        "reference_report should be carried into Generator"
    assert "mainImage" in gen3.code
    console.print(f"[green]OK[/green] elapsed={r3['elapsed_ms']:.1f}ms")
    console.print(f"  reference techniques: {gen3.spec.reference_report.techniques}")
    console.print(f"  generator iterations: {gen3.iterations}")

    section("Summary")
    console.print(Panel.fit("[bold green]ALL 3 TASKS PASS (dry-run)[/bold green]",
                            border_style="green"))
    console.print(
        "[dim]说明：本验证使用 fallback / stub，未真调 LLM；阶段四、五接入 DeepSeek 后会有质量提升。[/dim]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
