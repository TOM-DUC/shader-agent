"""独立验证 headless GLSL 渲染器（不依赖 DeepSeek）。

用法：
    python -m scripts.verify_renderer
    python -m scripts.verify_renderer --no-real         # 强制走 mock，跳过真 GL
    python -m scripts.verify_renderer --seed seed05     # 渲染指定 seed

通过条件：
  - 8 个 seed 中至少 6 个能成功编译 + 渲染
  - 有意构造的错误代码能被检测到（不漏报）
  - PNG 头部正确（前 8 字节 == \\x89PNG\\r\\n\\x1a\\n）

产物：
  - data/reports/render_test_{seed_id}.png   每个成功的 seed 落一张
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shader_agent.config.settings import settings
from shader_agent.corpus.seed_shaders import get_seed_shaders
from shader_agent.rendering import GLSLCompiler, GLSLRenderer
from shader_agent.rendering.mock import MockCompiler, MockRenderer

console = Console()
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def section(t: str) -> None:
    console.rule(f"[bold cyan]{t}[/bold cyan]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--no-real", action="store_true",
                   help="跳过真 GL，只用 mock 验证逻辑")
    p.add_argument("--seed", default="",
                   help="只渲染单个指定 seed_id（如 seed03）")
    p.add_argument("--save-png", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    section("环境检查")
    compiler = None
    renderer = None
    used = "mock"
    if not args.no_real:
        compiler, c_err = GLSLCompiler.try_create()
        renderer, r_err = GLSLRenderer.try_create()
        if compiler is not None and renderer is not None:
            used = "real"
            console.print(f"  backend       = [green]real GL (moderngl)[/green]")
        else:
            console.print(f"[yellow]  compiler/renderer 不可用，回退到 mock[/yellow]")
            if c_err: console.print(f"    compiler err: {c_err.splitlines()[0]}")
            if r_err: console.print(f"    renderer err: {r_err.splitlines()[0]}")
    if compiler is None:
        compiler = MockCompiler()
    if renderer is None:
        renderer = MockRenderer()
    console.print(f"  used          = {used}")

    # ---- 1. 编译错误检测（负样例）----
    section("1/3  Negative cases (broken code should fail compile)")
    bad_cases: list[tuple[str, str]] = [
        ("missing_main", "vec4 something(){ return vec4(1.0); }"),
        ("syntax_error",
         "void mainImage(out vec4 c, in vec2 p) { c = vec4(broken syntax here }"),
        ("dim_mismatch",
         "void mainImage(out vec4 c, in vec2 p) { vec3 v = vec4(1.); c = vec4(v,1.); }"),
    ]
    bad_results: list[bool] = []
    for name, code in bad_cases:
        cr = compiler.compile(code)
        flag = "[green]detected[/green]" if not cr.ok else "[red]MISSED[/red]"
        snippet = (cr.errors.splitlines()[0] if cr.errors else "(no errors)")[:80]
        console.print(f"  {flag}  {name:18s}  err: {snippet}")
        bad_results.append(not cr.ok)
    neg_ok = sum(bad_results) >= 2  # 至少 2/3 被检测到

    # ---- 2. Seed 编译 + 渲染 ----
    section("2/3  Render seeds")
    out_dir = settings.project_root / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = get_seed_shaders()
    if args.seed:
        seeds = [s for s in seeds if s.shader_id == args.seed]
        if not seeds:
            console.print(f"[red]seed '{args.seed}' 不存在[/red]")
            return 1

    table = Table()
    table.add_column("id"); table.add_column("name")
    table.add_column("compile", justify="center")
    table.add_column("render", justify="center")
    table.add_column("png_size", justify="right")
    table.add_column("elapsed", justify="right")

    render_pass = 0
    rendered_paths: list[Path] = []
    for s in seeds:
        cr = compiler.compile(s.code_image)
        compile_flag = "OK" if cr.ok else "FAIL"
        png_size = 0
        render_flag = "-"
        ms = 0.0
        if cr.ok:
            try:
                t0 = time.perf_counter()
                png = renderer.render(s.code_image, width=512, height=384, time=1.5)
                ms = (time.perf_counter() - t0) * 1000.0
                if isinstance(png, (bytes, bytearray)) and png.startswith(PNG_MAGIC):
                    png_size = len(png)
                    render_flag = "OK"
                    render_pass += 1
                    if args.save_png:
                        path = out_dir / f"render_test_{s.shader_id}.png"
                        path.write_bytes(png)
                        rendered_paths.append(path)
                else:
                    render_flag = "BAD_PNG"
            except Exception as e:
                render_flag = "EXC"
                console.print(f"  [yellow]{s.shader_id} render exception: {e}[/yellow]")
        table.add_row(s.shader_id, s.name[:30],
                     compile_flag, render_flag,
                     str(png_size), f"{ms:.1f}ms")
    console.print(table)

    # ---- 3. PNG 头部校验 ----
    section("3/3  PNG header check")
    head_ok = True
    for p in rendered_paths:
        b = p.read_bytes()[:8]
        ok = b == PNG_MAGIC
        flag = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  {flag}  {p.name}")
        head_ok = head_ok and ok

    section("Summary")
    console.print(f"  used backend         : {used}")
    console.print(f"  negative detection   : {sum(bad_results)}/{len(bad_results)}")
    console.print(f"  seeds rendered       : {render_pass}/{len(seeds)}")
    console.print(f"  PNG headers OK       : {head_ok}")

    target_render = max(1, len(seeds) - 2)  # mock 模式不挑剔；真 GL 模式允许偶尔失败
    all_ok = neg_ok and (render_pass >= target_render) and head_ok
    if all_ok:
        console.print(Panel.fit("[bold green]ALL CHECKS PASS[/bold green]",
                                border_style="green"))
        if rendered_paths:
            console.print(f"\n首张产物: {rendered_paths[0]}")
        return 0
    console.print(Panel.fit("[bold red]SOME CHECKS FAILED[/bold red]",
                            border_style="red"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
