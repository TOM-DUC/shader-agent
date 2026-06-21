"""Gradio 主应用（重设计版 · 修订 v2）。

设计依据 taste-skill（redesign + soft）：编辑式 / 柔和结构主义。

本次（v2）改动聚焦于布局稳定性与文案直陈：
- Hero：从横向 tag 条改为左文右标二栏。
- Section 标题：细线 + 居中标签，缓解通篇标题左对齐。
- 锁定列高：右列预览容器最小高度统一为 380px，状态条预填"待命"芯片，
  分析前 / 分析后骨架一致，不再出现"分析前右列空荡 + 分析后右列暴涨"。
- 锁定按钮尺寸：所有 ▶ 按钮加 `ts-action-btn` class，不随列宽缩放。
- Dropdown overflow / z-index 修复（详见 styling.py）。
- 圆角文本框 padding 修复（详见 styling.py）。
- 去 emoji：tab 名、accordion 标题等装饰 emoji 全部删除。
- 文案去 AI 化：副标题、footer 改为直陈式。

三个标签页
==========
  ① Analyzer · 解读现成 shader
  ② Generator · 按中文需求写新 shader
  ③ Remixer  · 基于现有代码做最小化改写

运行
=====
    python -m scripts.run_ui                 # 默认 127.0.0.1:7860
    python -m scripts.run_ui --share         # Gradio share 公网（慎用）
    python -m scripts.run_ui --port 8800
"""
from __future__ import annotations

import os
import warnings
from typing import Any

# Gradio 6.0 把 theme/css 从 Blocks() 迁到 launch()，多版本兼容：保留在 Blocks()，
# 仅屏蔽这条无害告警，保持日志干净。
warnings.filterwarnings(
    "ignore",
    message=r".*parameters have been moved from the Blocks constructor.*",
    category=UserWarning,
)

try:
    import gradio as gr
except ImportError as e:
    raise RuntimeError("需要 gradio：请 `pip install gradio>=4.40` 后重试。") from e

from shader_agent.ui import examples as ex
from shader_agent.ui import runners as rn
from shader_agent.ui.styling import (
    CUSTOM_CSS,
    GLOBAL_JS,
    build_theme,
    diagnostics_html,
    error_block,
    hero_html,
    idle_status_html,
    preview_placeholder,
    references_html,
    running_html,
    section_title,
    status_html,
)
from shader_agent.ui.webgl_preview import webgl_preview_html


# 预览高度常量：与 CSS 变量 --ts-preview-h 保持一致，避免散落魔法数
_PREVIEW_H = 380


# ============================================================
# 选项 → AssemblyOptions
# ============================================================

def _opts_from_ui(render_backend, use_vector_store, use_llm_cache,
                  enable_self_critique, max_fix_loops, top_k) -> rn.AssemblyOptions:
    return rn.AssemblyOptions(
        render_backend=render_backend,
        use_vector_store=use_vector_store,
        use_llm_cache=bool(use_llm_cache),
        enable_self_critique=bool(enable_self_critique),
        max_fix_loops=int(max_fix_loops),
        top_k=int(top_k),
    )


# ============================================================
# 实时预览 HTML
# ============================================================

def _webgl_enabled() -> bool:
    """是否启用 WebGL 实时预览。设 SHADER_AGENT_DISABLE_WEBGL=1 可关闭，回退静态图。"""
    return os.environ.get("SHADER_AGENT_DISABLE_WEBGL", "") != "1"


def _wrap_preview_empty(msg: str, hint: str) -> str:
    """统一预览占位包装，三处分支结构完全一致，配合 styling 锁定高度。"""
    return (
        '<div class="ts-preview"><div class="ts-preview-empty">'
        '<span class="ts-ph-mark">◇</span>'
        f'<span class="ts-ph-msg">{msg}</span>'
        f'<span class="ts-ph-hint">{hint}</span>'
        '</div></div>'
    )


def _preview_html(code: str, height: int = _PREVIEW_H) -> str:
    """三种分支都包在统一的 `.ts-preview` 外壳里，运行前后骨架一致。"""
    if not _webgl_enabled():
        return _wrap_preview_empty(
            "已禁用动态预览（SHADER_AGENT_DISABLE_WEBGL=1）· 展开下方静态图查看",
            "PREVIEW · DISABLED",
        )
    html = webgl_preview_html(code or "", height=height)
    if html:
        return f'<div class="ts-preview">{html}</div>'
    return _wrap_preview_empty(
        "无法实时预览",
        "PREVIEW · UNSUPPORTED",
    )


# ============================================================
# Tab 1 · Analyzer 回调
# ============================================================

def on_analyze(code, render_backend, use_vstore, use_cache, critique, max_fix, top_k):
    yield (gr.update(), gr.update(), gr.update(), gr.update(),
           running_html("正在分析 shader（检索 + 四段式讲解）…"), "", gr.update())

    opts = _opts_from_ui(render_backend, use_vstore, use_cache, critique, max_fix, top_k)
    res = rn.run_analyze(code, opts)
    asm = rn.get_assembly(opts)

    if not res["ok"]:
        status = status_html(backend_label=asm.backend_label, vstore_label=asm.vstore_label,
                             elapsed_ms=res["elapsed_ms"], compile_ok=None)
        yield (preview_placeholder("分析失败 · 请检查代码或下方诊断信息"), None,
               "", references_html([]),
               status, error_block(res["error"]),
               diagnostics_html(res.get("diagnostics") or []))
        return

    status = status_html(backend_label=asm.backend_label, vstore_label=asm.vstore_label,
                         elapsed_ms=res["elapsed_ms"], compile_ok=None)
    yield (_preview_html(code), res["image"],
           res["report_md"],
           references_html(res.get("references") or []),
           status, "", diagnostics_html(res.get("diagnostics") or []))


def on_load_analyzer_example(label: str):
    for lbl, code in ex.analyzer_examples():
        if lbl == label:
            return code
    return ""


# ============================================================
# Tab 2 · Generator 回调
# ============================================================

def on_generate(user_text, palette, complexity, dynamic,
                render_backend, use_vstore, use_cache, critique, max_fix, top_k):
    yield (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
           running_html("正在生成 GLSL（检索 → 起草 → 编译校验循环）…"),
           "", gr.update(), gr.update())

    opts = _opts_from_ui(render_backend, use_vstore, use_cache, critique, max_fix, top_k)
    res = rn.run_generate(user_text, opts, palette=palette,
                          complexity=complexity, dynamic=dynamic)
    asm = rn.get_assembly(opts)
    status = status_html(backend_label=asm.backend_label, vstore_label=asm.vstore_label,
                         elapsed_ms=res["elapsed_ms"], compile_ok=res.get("compile_ok"),
                         iterations=res.get("iterations", 0))
    if not res["ok"]:
        yield (preview_placeholder("生成失败 · 请检查需求或下方诊断信息"), None, "", "", "",
               status, error_block(res["error"]),
               diagnostics_html(res.get("diagnostics") or []), [])
        return

    compile_block = res["compile_errors"] if not res["compile_ok"] and res["compile_errors"] else ""
    crit = res["critique"] or "（未启用 / 评估失败）"
    yield (_preview_html(res["code"]), res["image"], res["code"],
           f"**解释**：{res['explanation'] or '(无)'}\n\n**自评**：{crit}",
           compile_block, status, "",
           diagnostics_html(res.get("diagnostics") or []),
           res.get("references") or [])





# ============================================================
# Tab 3 · Remixer 回调
# ============================================================

def on_remix(code, ask,
             render_backend, use_vstore, use_cache, critique, max_fix, top_k):
    yield (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
           running_html("正在基于原代码改写（最小化改写 → 编译校验）…"), "", gr.update())

    opts = _opts_from_ui(render_backend, use_vstore, use_cache, critique, max_fix, top_k)
    res = rn.run_collaborate(code, ask, opts)
    asm = rn.get_assembly(opts)
    status = status_html(backend_label=asm.backend_label, vstore_label=asm.vstore_label,
                         elapsed_ms=res["elapsed_ms"], compile_ok=res.get("compile_ok"),
                         iterations=res.get("iterations", 0))
    if not res["ok"]:
        yield (preview_placeholder("改写失败 · 请检查原代码与指令"), None, None, "", "", "",
               status, error_block(res["error"]),
               diagnostics_html(res.get("diagnostics") or []))
        return

    compile_block = res["compile_errors"] if not res["compile_ok"] and res["compile_errors"] else ""
    crit = res["critique"] or "（未启用）"
    rewrite_md = (f"**改写说明**：{res['new_explanation'] or '(无)'}\n\n**自评**：{crit}\n"
                  + (f"\n```\n{compile_block}\n```" if compile_block else ""))
    yield (_preview_html(res["new_code"]),
           res["image_before"], res["image_after"],
           res.get("report_md", ""), res["new_code"], rewrite_md,
           status, "", diagnostics_html(res.get("diagnostics") or []))


def on_load_remix_example(label: str):
    for lbl, code, ask in ex.collaborate_examples():
        if lbl == label:
            return code, ask
    return "", ""


# ============================================================
# 控件选项（中文显示 + 英文值）
# ============================================================

_PALETTE_CHOICES = [
    ("（不指定）", ""),
    ("霓虹（高饱和发光）", "neon glowing"),
    ("暖色夕阳（橙红金）", "warm sunset orange red"),
    ("冷色蓝调（蓝青）", "cool blue cyan"),
    ("单色灰阶", "monochrome grayscale"),
    ("柔和粉彩", "soft pastel"),
    ("鲜艳明快", "vibrant saturated"),
    ("赛博朋克（紫粉青）", "cyberpunk purple pink cyan"),
    ("自然大地色（绿棕）", "earthy green brown nature"),
    ("黑金高对比", "black and gold high contrast"),
]
_COMPLEXITY_CHOICES = [
    ("极简（几行，最快）", "minimal"),
    ("简单（单一效果）", "simple"),
    ("适中（多技巧组合）", "moderate"),
    ("复杂（炫技，较慢）", "complex"),
]
_CODE_LANG = "javascript"  # gr.Code 无 glsl，用 js 近似高亮


def build_app() -> "gr.Blocks":
    analyzer_ex = ex.analyzer_examples()
    analyzer_labels = [r[0] for r in analyzer_ex]
    remix_ex = ex.collaborate_examples()
    remix_labels = [r[0] for r in remix_ex]

    with gr.Blocks(title="Shader Agent", theme=build_theme(), css=CUSTOM_CSS) as app:
        # ---------- Hero ----------
        gr.HTML(hero_html())

        # ---------- 运行选项 ----------
        # 去掉装饰 emoji。Accordion 自身已经有视觉提示，不需要齿轮 icon。
        with gr.Accordion("运行选项（通用）", open=False,
                          elem_classes="ts-run-opts"):
            with gr.Row():
                render_backend = gr.Dropdown(
                    label="渲染后端", choices=["auto", "real", "mock"], value="auto",
                    info="auto = 优先真 GL，失败回退 mock")
                use_vstore = gr.Dropdown(
                    label="向量库", choices=["auto", "off"], value="auto",
                    info="auto = 若 vector_db 存在则连接")
                use_cache = gr.Checkbox(label="LLM 缓存", value=True,
                    info="相同请求不重复调用")
                critique = gr.Checkbox(label="自评", value=False,
                    info="对结果做质量自评（含编译错误分析）")
            with gr.Row():
                max_fix = gr.Slider(label="修正循环最大轮数",
                                    minimum=0, maximum=4, step=1, value=2)
                top_k = gr.Slider(label="检索 top-k",
                                  minimum=1, maximum=8, step=1, value=3)

        common_inputs = [render_backend, use_vstore, use_cache, critique, max_fix, top_k]

        # ============================================================
        # Tab 1 · Analyzer
        # ============================================================
        with gr.Tab("Analyzer · 解读现成 shader"):
            with gr.Row(elem_classes="ts-tab-row"):
                with gr.Column(scale=5, elem_classes="ts-col-input"):
                    gr.HTML(section_title("输入"))
                    a_ex = gr.Dropdown(label="加载示例 shader",
                                       choices=[""] + analyzer_labels, value="")
                    a_code = gr.Code(label="GLSL 代码（Shadertoy 风格 mainImage）",
                                     language=_CODE_LANG, lines=18, value="",
                                     elem_classes="ts-tight")
                    a_btn = gr.Button("▶ 分析", variant="primary",
                                      size="lg", elem_classes="ts-action-btn")
                with gr.Column(scale=4, elem_classes="ts-col-output"):
                    gr.HTML(section_title("实时预览"))
                    a_preview = gr.HTML(_preview_html(""))
                    a_status = gr.HTML(idle_status_html("待命 · 点击「▶ 分析」开始"))
                    a_err = gr.HTML()
                    with gr.Accordion("后端单帧渲染（静态）", open=False):
                        a_img = gr.Image(label="渲染预览", height=280)
                    a_diag = gr.HTML()

            gr.HTML(section_title("分析报告"))
            a_report = gr.Markdown(label="分析报告")

            gr.HTML(section_title("对照参考样本（检索源码）"))
            a_refs = gr.HTML(references_html([]))

            a_ex.change(on_load_analyzer_example, inputs=[a_ex], outputs=[a_code])
            a_btn.click(on_analyze, inputs=[a_code] + common_inputs,
                        outputs=[a_preview, a_img, a_report, a_refs, a_status, a_err, a_diag])

        # ============================================================
        # Tab 2 · Generator
        # ============================================================
        with gr.Tab("Generator · 按需求写新 shader"):
            with gr.Row(elem_classes="ts-tab-row"):
                with gr.Column(scale=5, elem_classes="ts-col-input"):
                    gr.HTML(section_title("需求"))
                    g_prompt = gr.Textbox(
                        label="自然语言需求",
                        placeholder="例：画一个霓虹蓝紫万花筒，6 折对称，带时间动画",
                        lines=4)
                    with gr.Row():
                        g_palette = gr.Dropdown(
                            label="调色板", choices=_PALETTE_CHOICES,
                            value="",
                            info="主色调倾向；不指定则模型自由发挥")
                        g_complexity = gr.Dropdown(
                            label="复杂度", choices=_COMPLEXITY_CHOICES,
                            value="simple",
                            info="越复杂越炫，但更慢、更易出错")
                        g_dynamic = gr.Checkbox(
                            label="动态（iTime）", value=True,
                            info="勾选要求时间动画")
                    g_btn = gr.Button("▶ 生成", variant="primary",
                                      size="lg", elem_classes="ts-action-btn")
                    gr.Examples(label="一键填入预置需求",
                                examples=ex.generator_examples(),
                                inputs=[g_prompt, g_palette, g_complexity, g_dynamic])
                with gr.Column(scale=4, elem_classes="ts-col-output"):
                    gr.HTML(section_title("实时预览"))
                    g_preview = gr.HTML(_preview_html(""))
                    g_status = gr.HTML(idle_status_html("待命 · 点击「▶ 生成」开始"))
                    g_err = gr.HTML()
                    with gr.Accordion("后端单帧渲染（静态）", open=False):
                        g_img = gr.Image(label="渲染预览", height=280)
                    g_diag = gr.HTML()

            gr.HTML(section_title("生成结果"))
            with gr.Row():
                with gr.Column(scale=3):
                    g_code = gr.Code(label="生成的 GLSL", language=_CODE_LANG,
                                     lines=18, elem_classes="ts-tight")
                    g_explain = gr.Markdown()
                    g_compile = gr.Code(label="编译错误（若有）", language=_CODE_LANG,
                                        lines=3, interactive=False,
                                        elem_classes="ts-tight")
                with gr.Column(scale=1):
                    with gr.Accordion("使用的参考样本（检索）", open=False):
                        g_refs = gr.JSON()

            g_btn.click(on_generate,
                        inputs=[g_prompt, g_palette, g_complexity, g_dynamic] + common_inputs,
                        outputs=[g_preview, g_img, g_code, g_explain, g_compile,
                                 g_status, g_err, g_diag, g_refs])

        # ============================================================
        # Tab 3 · Remixer
        # ============================================================
        with gr.Tab("Remixer · 基于现有代码改写"):
            with gr.Row(elem_classes="ts-tab-row"):
                with gr.Column(scale=5, elem_classes="ts-col-input"):
                    gr.HTML(section_title("原始代码 + 指令"))
                    r_ex = gr.Dropdown(label="加载示例代码 + 改写指令",
                                       choices=[""] + remix_labels, value="")
                    r_code = gr.Code(label="原始 GLSL 代码（将在此基础上改写）",
                                     language=_CODE_LANG, lines=16, value="",
                                     elem_classes="ts-tight")
                    r_ask = gr.Textbox(label="改写指令（中文）", lines=3,
                        placeholder="例：把主色调换成霓虹紫；或：让旋转速度加快一倍。"
                                    "指令越具体，改动越精准。")
                    r_btn = gr.Button("▶ 基于原代码改写", variant="primary",
                                      size="lg", elem_classes="ts-action-btn")
                with gr.Column(scale=4, elem_classes="ts-col-output"):
                    gr.HTML(section_title("改写后实时预览"))
                    r_preview = gr.HTML(_preview_html(""))
                    r_status = gr.HTML(idle_status_html("待命 · 点击「▶ 基于原代码改写」开始"))
                    r_err = gr.HTML()
                    with gr.Accordion("后端单帧渲染（静态对比）", open=False):
                        with gr.Row():
                            r_before = gr.Image(label="原始", height=200)
                            r_after = gr.Image(label="改写后", height=200)
                    r_diag = gr.HTML()

            gr.HTML(section_title("改写结果"))
            with gr.Row():
                with gr.Column(scale=3):
                    r_new_code = gr.Code(label="改写后 GLSL", language=_CODE_LANG,
                                         lines=16, elem_classes="ts-tight")
                    r_rewrite = gr.Markdown()
                with gr.Column(scale=1):
                    with gr.Accordion("原代码简析（辅助参考）", open=False):
                        r_report = gr.Markdown()

            r_ex.change(on_load_remix_example, inputs=[r_ex],
                        outputs=[r_code, r_ask])
            r_btn.click(on_remix, inputs=[r_code, r_ask] + common_inputs,
                        outputs=[r_preview, r_before, r_after, r_report,
                                 r_new_code, r_rewrite, r_status, r_err, r_diag])

        # ---------- Footer ----------
        gr.HTML(
            '<div class="ts-footer">'
            '渲染后端 auto 失败时回退 mock'
            '<span class="ts-foot-sep">·</span>'
            '向量库为空时检索安静跳过'
            '<span class="ts-foot-sep">·</span>'
            '产物落在 <code>data/reports/</code>'
            '</div>'
        )

        # ---------- 全局 JS（复制按钮 + 下拉框层叠兜底，注入一次） ----------
        gr.HTML(GLOBAL_JS)

    return app


def _warmup_background() -> None:
    """后台预热：提前加载嵌入模型 + 建好 GL context，避免首次请求被冷启动拖慢。"""
    import threading

    def _job():
        try:
            from shader_agent.embeddings.bge_embedder import get_embedder
            get_embedder().embed_one("warmup: a shader that draws a red circle")
        except Exception:
            pass
        try:
            from shader_agent.rendering import GLSLCompiler
            GLSLCompiler.try_create()
        except Exception:
            pass

    threading.Thread(target=_job, name="warmup", daemon=True).start()


def launch(*, server_name: str = "127.0.0.1", server_port: int = 7860,
           share: bool = False, inbrowser: bool = True, warmup: bool = True,
           **launch_kwargs: Any) -> None:
    app = build_app()
    if warmup:
        _warmup_background()
    app.queue(max_size=16).launch(
        server_name=server_name, server_port=server_port,
        share=share, inbrowser=inbrowser, **launch_kwargs)


if __name__ == "__main__":
    launch()
