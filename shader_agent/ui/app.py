"""阶段七：Gradio 三标签页主应用。

布局总览
=========

  顶部：运行选项行（渲染后端 / 向量库 / 缓存 / 自评 / max_fix_loops / top_k）+ 状态徽章

  Tab 1 「Shader Analyzer」
  ------------------------
   左： [Examples 下拉] + GLSL 代码输入框 + 「分析」按钮
   右上：渲染缩略图（512×384）
   右下：分析报告 Markdown（折叠）+ 原始 JSON

  Tab 2 「Shader Generator」
  ------------------------
   左： 中文自然语言输入框 + 调色板/复杂度/动态 控件 + 「生成」按钮
   右上：渲染缩略图
   右下：生成的代码（GLSL 高亮）+ 编译徽章 + 自评

  Tab 3 「Collaborative Rewrite」
  -------------------------------
   左上：参考 GLSL 代码
   左下：改写指令 + 「先分析后改写」按钮
   右上：原始 vs 新版 两张缩略图并排
   右下：分析报告 + 新版代码 + 编译徽章

运行
=====
    python -m scripts.run_ui                  # 默认 127.0.0.1:7860
    python -m scripts.run_ui --share          # Gradio share 公网（慎用）
    python -m scripts.run_ui --port 8800
"""
from __future__ import annotations

import os
import warnings
from typing import Any

# Gradio 6.0 把 theme/css 从 Blocks() 迁到 launch()，但旧版仍需在 Blocks() 传入。
# 为兼容多版本仍保留在 Blocks()，仅屏蔽这条无害的 UserWarning，保持日志干净。
warnings.filterwarnings(
    "ignore",
    message=r".*parameters have been moved from the Blocks constructor.*",
    category=UserWarning,
)

try:
    import gradio as gr
except ImportError as e:  # 给出友好的错误信息而不是 ImportError 直接崩
    raise RuntimeError(
        "阶段七需要 gradio：请 `pip install gradio>=4.40` 后重试。"
    ) from e

from shader_agent.ui import examples as ex
from shader_agent.ui import runners as rn
from shader_agent.ui.styling import (
    CUSTOM_CSS,
    diagnostics_html,
    error_block,
    running_html,
    status_html,
)
from shader_agent.ui.webgl_preview import webgl_preview_html


# ============================================================
# 选项面板控件 → AssemblyOptions
# ============================================================

def _opts_from_ui(
    render_backend: str,
    use_vector_store: str,
    use_llm_cache: bool,
    enable_self_critique: bool,
    max_fix_loops: int,
    top_k: int,
) -> rn.AssemblyOptions:
    return rn.AssemblyOptions(
        render_backend=render_backend,
        use_vector_store=use_vector_store,
        use_llm_cache=bool(use_llm_cache),
        enable_self_critique=bool(enable_self_critique),
        max_fix_loops=int(max_fix_loops),
        top_k=int(top_k),
    )


# ============================================================
# Tab 1 回调
# ============================================================

def _webgl_enabled() -> bool:
    """是否启用 WebGL 实时预览（方案一）。

    默认开启。若用户实测方案一仍黑屏，想彻底关闭动态预览改回纯静态图
    （方案二），无需改代码——设置环境变量即可：
        SHADER_AGENT_DISABLE_WEBGL=1
    关闭后，预览区会直接提示"已禁用动态预览"，只看后端单帧静态图。
    """
    return os.environ.get("SHADER_AGENT_DISABLE_WEBGL", "") != "1"


def _preview_html(code: str, height: int = 300) -> str:
    """构造实时 WebGL 预览 HTML；不支持/已禁用时返回提示，让用户看静态图。"""
    if not _webgl_enabled():
        return (
            '<div style="padding:10px;font:12px sans-serif;color:#555;'
            'background:#f3f3f5;border-radius:8px;">'
            "已禁用动态预览（SHADER_AGENT_DISABLE_WEBGL=1）；"
            "请展开下方「后端单帧渲染」查看静态效果图。</div>"
        )
    html = webgl_preview_html(code or "", height=height)
    if html:
        return html
    return (
        '<div style="padding:10px;font:12px sans-serif;color:#876500;'
        'background:#fff7e0;border-radius:8px;">'
        "此 shader 含多通道/纹理（iChannel 等）或缺少 mainImage，无法实时预览；"
        "下方静态图为后端单帧渲染结果。</div>"
    )


def on_analyze(code: str, render_backend, use_vstore, use_cache,
               critique, max_fix, top_k):
    # 先给即时反馈：避免界面像"卡死"。第一次 yield 只更新状态条，
    # 其余输出用 gr.update() 保持不变。
    yield (
        gr.update(),                       # 实时预览 HTML
        gr.update(),                       # 图像
        gr.update(),                       # report_md
        gr.update(),                       # json
        running_html("正在分析 shader（检索 + 四段式讲解）…"),
        "",                                # 清空旧错误
        gr.update(),                       # diagnostics
    )

    opts = _opts_from_ui(render_backend, use_vstore, use_cache,
                         critique, max_fix, top_k)
    res = rn.run_analyze(code, opts)
    asm = rn.get_assembly(opts)

    if not res["ok"]:
        status = status_html(
            backend_label=asm.backend_label, vstore_label=asm.vstore_label,
            elapsed_ms=res["elapsed_ms"], compile_ok=None,
        )
        yield (
            "",    # 预览
            None,  # 图像
            "",    # report_md
            {},    # json
            status,
            error_block(res["error"]),
            diagnostics_html(res.get("diagnostics") or []),
        )
        return
    status = status_html(
        backend_label=asm.backend_label, vstore_label=asm.vstore_label,
        elapsed_ms=res["elapsed_ms"], compile_ok=None,
    )
    yield (
        _preview_html(code),
        res["image"],
        res["report_md"],
        res["report_json"],
        status,
        "",  # 无错误
        diagnostics_html(res.get("diagnostics") or []),
    )


def on_load_analyzer_example(label: str):
    """从下拉框 label 取出对应的 seed code。"""
    for lbl, code in ex.analyzer_examples():
        if lbl == label:
            return code
    return ""


# ============================================================
# Tab 2 回调
# ============================================================

def on_generate(user_text, palette, complexity, dynamic,
                render_backend, use_vstore, use_cache,
                critique, max_fix, top_k):
    yield (
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        running_html("正在生成 GLSL（检索 → 起草 → 编译校验循环）…"),
        "", gr.update(), gr.update(),
    )
    opts = _opts_from_ui(render_backend, use_vstore, use_cache,
                         critique, max_fix, top_k)
    res = rn.run_generate(user_text, opts, palette=palette,
                          complexity=complexity, dynamic=dynamic)
    asm = rn.get_assembly(opts)
    status = status_html(
        backend_label=asm.backend_label, vstore_label=asm.vstore_label,
        elapsed_ms=res["elapsed_ms"], compile_ok=res.get("compile_ok"),
        iterations=res.get("iterations", 0),
    )
    if not res["ok"]:
        yield (
            "", None, "", "", "",
            status,
            error_block(res["error"]),
            diagnostics_html(res.get("diagnostics") or []),
            [],
        )
        return
    compile_block = res["compile_errors"] if not res["compile_ok"] and res["compile_errors"] else ""
    crit = res["critique"] or "（未启用 / 评估失败）"
    yield (
        _preview_html(res["code"]),
        res["image"],
        res["code"],
        f"**解释**：{res['explanation'] or '(无)'}\n\n**自评**：{crit}",
        compile_block,
        status,
        "",  # 无错误
        diagnostics_html(res.get("diagnostics") or []),
        res.get("references") or [],
    )


def on_load_generator_example(prompt, palette, complexity, dynamic):
    """Examples 组件直接把行作为参数填到对应控件。这里仅作为 placeholder
       保证类型，gr.Examples 会自动完成填充。"""
    return prompt, palette, complexity, dynamic


# ============================================================
# Tab 3 回调
# ============================================================

def on_collaborate(code, ask, render_backend, use_vstore, use_cache,
                   critique, max_fix, top_k):
    yield (
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        running_html("正在基于原代码改写（最小化改写 → 编译校验）…"),
        "", gr.update(),
    )
    opts = _opts_from_ui(render_backend, use_vstore, use_cache,
                         critique, max_fix, top_k)
    res = rn.run_collaborate(code, ask, opts)
    asm = rn.get_assembly(opts)
    status = status_html(
        backend_label=asm.backend_label, vstore_label=asm.vstore_label,
        elapsed_ms=res["elapsed_ms"], compile_ok=res.get("compile_ok"),
        iterations=res.get("iterations", 0),
    )
    if not res["ok"]:
        yield (
            "", None, None, "", "", "",
            status,
            error_block(res["error"]),
            diagnostics_html(res.get("diagnostics") or []),
        )
        return
    compile_block = res["compile_errors"] if not res["compile_ok"] and res["compile_errors"] else ""
    crit = res["critique"] or "（未启用）"
    rewrite_md = (
        f"**改写解释**：{res['new_explanation'] or '(无)'}\n\n"
        f"**自评**：{crit}\n"
    )
    yield (
        _preview_html(res["new_code"], height=220),
        res["image_before"],
        res["image_after"],
        res["report_md"],
        res["new_code"],
        rewrite_md + (f"\n```\n{compile_block}\n```" if compile_block else ""),
        status,
        "",  # 无错误
        diagnostics_html(res.get("diagnostics") or []),
    )


def on_load_collab_example(label: str):
    for lbl, code, ask in ex.collaborate_examples():
        if lbl == label:
            return code, ask
    return "", ""


# ============================================================
# 装配 Gradio Blocks
# ============================================================

# 调色板：UI 显示中文，实际传给 LLM 的是英文/描述值（label, value）。
# Gradio Dropdown 的 choices 支持 (显示文本, 实际值) 二元组。
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
# 复杂度：中文显示 + 英文枚举值（与 GenerationSpec.complexity 的 Literal 对齐）
_COMPLEXITY_CHOICES = [
    ("极简（几行，最快）", "minimal"),
    ("简单（单一效果）", "simple"),
    ("适中（多技巧组合）", "moderate"),
    ("复杂（炫技，较慢）", "complex"),
]


def build_app() -> "gr.Blocks":
    analyzer_ex = ex.analyzer_examples()
    analyzer_labels = [r[0] for r in analyzer_ex]
    collab_ex = ex.collaborate_examples()
    collab_labels = [r[0] for r in collab_ex]

    with gr.Blocks(
        title="Shader Agent",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
        css=CUSTOM_CSS,
    ) as app:
        gr.Markdown(
            "# Shader Agent\n"
            "基于 DeepSeek + bge-m3 + moderngl 的 Shadertoy 智能教学助手。"
            "三个标签页：Analyzer 解读现成 shader；Generator 按中文需求写新 shader；"
            "Remixer 在你给的代码基础上按指令做**最小化改写**。"
        )

        # ----- 顶部：运行选项 -----
        with gr.Accordion("⚙ 运行选项（一次设置，三个标签页通用）", open=False):
            with gr.Row():
                render_backend = gr.Dropdown(
                    label="渲染后端",
                    choices=["auto", "real", "mock"], value="auto",
                    info="auto = 优先真 GL，失败回退 mock"
                )
                use_vstore = gr.Dropdown(
                    label="向量库",
                    choices=["auto", "off"], value="auto",
                    info="auto = 若 vector_db 存在则连接"
                )
                use_cache = gr.Checkbox(
                    label="LLM 缓存", value=True,
                    info="同 messages 不重复请求"
                )
                critique = gr.Checkbox(
                    label="自评 (vision critique)", value=False,
                    info="启用渲染图 + 多模态评估"
                )
            with gr.Row():
                max_fix = gr.Slider(
                    label="修正循环最大轮数",
                    minimum=0, maximum=4, step=1, value=2,
                )
                top_k = gr.Slider(
                    label="检索 top_k",
                    minimum=1, maximum=8, step=1, value=3,
                )

        common_inputs = [render_backend, use_vstore, use_cache,
                         critique, max_fix, top_k]

        # ============================================================
        # Tab 1: Analyzer
        # ============================================================
        with gr.Tab("① Analyzer · 解读现成 shader"):
            with gr.Row():
                with gr.Column(scale=1):
                    ex_dd = gr.Dropdown(
                        label="加载示例 shader",
                        choices=[""] + analyzer_labels,
                        value="",
                    )
                    code_in = gr.Code(
                        label="GLSL 代码（Shadertoy 风格 mainImage）",
                        language="javascript",  # gr.Code 不支持 glsl，用 js 近似高亮
                        lines=22,
                        value="",
                    )
                    btn_analyze = gr.Button("▶ 分析", variant="primary")
                with gr.Column(scale=1):
                    preview_out = gr.HTML(label="实时预览")
                    with gr.Accordion("后端单帧渲染（静态）", open=False):
                        img_out = gr.Image(label="渲染预览", height=300)
                    status_out = gr.HTML()
                    err_out = gr.HTML()
                    diag_out = gr.HTML()
            with gr.Row():
                with gr.Column(scale=2):
                    report_md = gr.Markdown(label="分析报告")
                with gr.Column(scale=1):
                    with gr.Accordion("Report JSON", open=False):
                        report_json = gr.JSON(label="报告原始字段")

            ex_dd.change(on_load_analyzer_example, inputs=[ex_dd], outputs=[code_in])
            btn_analyze.click(
                on_analyze,
                inputs=[code_in] + common_inputs,
                outputs=[preview_out, img_out, report_md, report_json,
                         status_out, err_out, diag_out],
            )

        # ============================================================
        # Tab 2: Generator
        # ============================================================
        with gr.Tab("② Generator · 按中文需求写新 shader"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_in = gr.Textbox(
                        label="自然语言需求（中文 OK）",
                        placeholder="例：画一个霓虹蓝紫万花筒，6 折对称，带时间动画",
                        lines=4,
                    )
                    with gr.Row():
                        palette_in = gr.Dropdown(
                            label="调色板", choices=_PALETTE_CHOICES, value="",
                            info="主色调倾向。会写进生成提示词，影响配色；选「不指定」则由模型自由发挥。",
                        )
                        complexity_in = gr.Dropdown(
                            label="复杂度", choices=_COMPLEXITY_CHOICES, value="simple",
                            info="期望的算法繁复程度。越复杂越炫但生成/编译越慢、越易出错。",
                        )
                        dynamic_in = gr.Checkbox(
                            label="动态（带 iTime）", value=True,
                            info="勾选则要求使用 iTime 做时间动画；取消则倾向静态画面。",
                        )
                    btn_gen = gr.Button("▶ 生成", variant="primary")
                    gr.Examples(
                        label="一键填入预置 prompt",
                        examples=ex.generator_examples(),
                        inputs=[prompt_in, palette_in, complexity_in, dynamic_in],
                    )
                with gr.Column(scale=1):
                    gen_preview_out = gr.HTML(label="实时预览")
                    with gr.Accordion("后端单帧渲染（静态）", open=False):
                        gen_img_out = gr.Image(label="渲染预览", height=300)
                    gen_status_out = gr.HTML()
                    gen_err_out = gr.HTML()
                    gen_diag_out = gr.HTML()
            with gr.Row():
                with gr.Column(scale=2):
                    gen_code_out = gr.Code(
                        label="生成的 GLSL",
                        language="javascript", lines=18,
                    )
                    gen_explain_md = gr.Markdown(label="解释 / 自评")
                    gen_compile_out = gr.Code(
                        label="编译错误（若有）",
                        language="javascript", lines=4, interactive=False,
                    )
                with gr.Column(scale=1):
                    gen_refs_out = gr.JSON(label="使用的参考样本（向量检索）")

            btn_gen.click(
                on_generate,
                inputs=[prompt_in, palette_in, complexity_in, dynamic_in]
                       + common_inputs,
                outputs=[gen_preview_out, gen_img_out, gen_code_out, gen_explain_md,
                         gen_compile_out, gen_status_out, gen_err_out,
                         gen_diag_out, gen_refs_out],
            )

        # ============================================================
        # Tab 3: Remixer
        # ============================================================
        with gr.Tab("③ Remixer · 基于现有代码改写"):
            with gr.Row():
                with gr.Column(scale=1):
                    collab_ex_dd = gr.Dropdown(
                        label="加载示例代码 + 改写指令",
                        choices=[""] + collab_labels,
                        value="",
                    )
                    collab_code_in = gr.Code(
                        label="原始 GLSL 代码（将在此基础上改写）",
                        language="javascript", lines=18, value="",
                    )
                    collab_ask_in = gr.Textbox(
                        label="改写指令（中文）",
                        placeholder=(
                            "例：把主色调换成霓虹紫；或：在中心加一个柔和光斑；"
                            "或：让旋转速度加快一倍。指令越具体，改动越精准。"
                        ),
                        lines=3,
                    )
                    btn_collab = gr.Button(
                        "▶ 基于原代码改写", variant="primary",
                    )
                with gr.Column(scale=1):
                    collab_preview_out = gr.HTML(label="改写后实时预览")
                    with gr.Accordion("后端单帧渲染（静态对比）", open=False):
                        with gr.Row():
                            before_img = gr.Image(label="原始", height=220)
                            after_img = gr.Image(label="改写后", height=220)
                    collab_status_out = gr.HTML()
                    collab_err_out = gr.HTML()
                    collab_diag_out = gr.HTML()
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("原代码简析（辅助参考，可折叠）", open=False):
                        collab_report_md = gr.Markdown()
                with gr.Column(scale=1):
                    collab_new_code = gr.Code(
                        label="改写后 GLSL",
                        language="javascript", lines=16,
                    )
                    collab_rewrite_md = gr.Markdown()

            collab_ex_dd.change(
                on_load_collab_example,
                inputs=[collab_ex_dd],
                outputs=[collab_code_in, collab_ask_in],
            )
            btn_collab.click(
                on_collaborate,
                inputs=[collab_code_in, collab_ask_in] + common_inputs,
                outputs=[
                    collab_preview_out,
                    before_img, after_img,
                    collab_report_md, collab_new_code, collab_rewrite_md,
                    collab_status_out, collab_err_out, collab_diag_out,
                ],
            )

        gr.Markdown(
            "---\n"
            "**Tips**：⚠ 首次启动时若选 `auto` 渲染后端，会探测 moderngl；"
            "若失败会自动回退到 mock（图像始终是 1×1 红色像素）。"
            "向量库为空时检索分支会安静跳过，不报错。\n"
            "运行产物会落在 `data/reports/ui_session_*/`，可定期清理。"
        )

    return app


def _warmup_background() -> None:
    """后台预热：提前 import torch/sentence-transformers 并加载嵌入模型 +
    建好 GL context。把首次请求才会触发的 ~100s 冷启动（torch 导入 + 模型加载）
    挪到启动阶段后台完成，用户点击时基本已就绪，首条请求不再被拖慢。
    """
    import threading

    def _job():
        try:
            from shader_agent.embeddings.bge_embedder import get_embedder
            emb = get_embedder()
            emb.embed_one("warmup: a shader that draws a red circle")
        except Exception:
            pass
        try:
            from shader_agent.rendering import GLSLCompiler
            GLSLCompiler.try_create()  # 在 GL 专用线程上建好 context
        except Exception:
            pass

    threading.Thread(target=_job, name="warmup", daemon=True).start()


def launch(
    *,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
    warmup: bool = True,
    **launch_kwargs: Any,
) -> None:
    app = build_app()
    if warmup:
        _warmup_background()
    app.queue(max_size=16).launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        inbrowser=inbrowser,
        **launch_kwargs,
    )


if __name__ == "__main__":
    launch()
