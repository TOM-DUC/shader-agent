"""把 Analyzer/Generator/Orchestrator 包装成 UI 友好的同步函数。

Gradio 回调期望：
  - 输入 / 输出都是基础类型（str / dict / bytes / PIL.Image）；
  - 失败不抛异常，而是返回错误字符串放在状态栏；
  - 长任务可以多次 yield 给前端做进度提示。

本模块同时承担"启动期依赖装配"：决定 llm_fn / compiler / renderer / vector_store
是真还是 mock，并把装配后的单例缓存住，避免每次回调都重建。
"""
from __future__ import annotations

import io
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from shader_agent.agents.analyzer import ShaderAnalyzer
from shader_agent.agents.generator import ShaderGenerator
from shader_agent.agents.orchestrator import Orchestrator
from shader_agent.agents.schemas import (
    AnalysisReport,
    GeneratedShader,
    GenerationSpec,
    Message,
)
from shader_agent.config.settings import settings
from shader_agent.utils.logger import logger


# ============================================================
# 装配选项 + 单例缓存
# ============================================================

@dataclass
class AssemblyOptions:
    """UI 顶部"运行选项"控件的状态。"""
    render_backend: str = "auto"        # "auto" | "mock" | "real"
    use_vector_store: str = "auto"      # "auto" | "off"
    use_llm_cache: bool = True
    enable_self_critique: bool = False
    max_fix_loops: int = 2
    top_k: int = 3


@dataclass
class _Assembly:
    """缓存一次装配后的产物。"""
    analyzer: ShaderAnalyzer
    generator: ShaderGenerator
    orchestrator: Orchestrator
    backend_label: str
    vstore_label: str
    diagnostics: list[str] = field(default_factory=list)


_ASSEMBLY_CACHE: dict[str, _Assembly] = {}


def _cache_key(opts: AssemblyOptions) -> str:
    return (
        f"{opts.render_backend}|{opts.use_vector_store}|"
        f"{opts.use_llm_cache}|{opts.enable_self_critique}|"
        f"{opts.max_fix_loops}|{opts.top_k}"
    )


def _build_llm_fns(use_cache: bool):
    """构造 chat / json / code / vision / text_critique 五个 llm_fn。
    若 DEEPSEEK_API_KEY 缺失，返回 (None, None, None, None, None, "缺 key")，
    Analyzer/Generator 会自动走 fallback 路径。"""
    if not os.environ.get("DEEPSEEK_API_KEY") and not settings.deepseek_api_key:
        return None, None, None, None, None, "缺 DEEPSEEK_API_KEY；Analyzer/Generator 走 fallback"
    try:
        from shader_agent.llm.llm_fn import (
            make_chat_fn, make_code_fn, make_json_fn,
            make_text_critique_fn, make_vision_critique_fn,
        )
        chat_fn = make_chat_fn(use_cache=use_cache)
        json_fn = make_json_fn(use_cache=use_cache)
        code_fn = make_code_fn(use_cache=use_cache)
        vision_fn = make_vision_critique_fn(use_cache=use_cache)
        text_critique_fn = make_text_critique_fn(use_cache=use_cache)
        return chat_fn, json_fn, code_fn, vision_fn, text_critique_fn, ""
    except Exception as e:
        return None, None, None, None, None, f"llm_fn 装配失败: {e}"


def _build_vector_store(use_vstore: str):
    """按需懒加载 vector store。失败返回 (None, reason)。"""
    if use_vstore == "off":
        return None, "用户在 UI 中关闭了向量库"
    try:
        from shader_agent.corpus.vector_store import ShaderVectorStore
        vs = ShaderVectorStore()
        if vs.count() == 0:
            return None, "向量库为空。运行 `python -m scripts.build_corpus` 先建库。"
        return vs, f"已连接（{vs.count()} 条）"
    except Exception as e:
        return None, f"向量库不可用: {e}"


def _build_render_backend(prefer: str):
    """决定 compiler/renderer 用真 GL 还是 mock。"""
    from shader_agent.rendering import GLSLCompiler, GLSLRenderer
    from shader_agent.rendering.mock import MockCompiler, MockRenderer

    if prefer == "mock":
        return MockCompiler(), MockRenderer(), "mock（用户强制）"

    if prefer in ("auto", "real"):
        c, c_err = GLSLCompiler.try_create()
        r, r_err = GLSLRenderer.try_create()
        if c is not None and r is not None:
            return c, r, "moderngl 真 GL"
        # auto 时回退
        if prefer == "auto":
            return MockCompiler(), MockRenderer(), (
                f"mock（真 GL 不可用：{(c_err or r_err or '').splitlines()[0][:80]}）"
            )
        # real 时拒绝降级，让 UI 显示错误
        raise RuntimeError(f"真 GL 不可用：{c_err or r_err}")

    return MockCompiler(), MockRenderer(), "mock（未知 backend 选项）"


def get_assembly(opts: AssemblyOptions) -> _Assembly:
    """按 opts 装配 Analyzer/Generator/Orchestrator；命中缓存则直接复用。"""
    key = _cache_key(opts)
    if key in _ASSEMBLY_CACHE:
        return _ASSEMBLY_CACHE[key]

    diag: list[str] = []

    # 1. LLM
    chat_fn, json_fn, code_fn, vision_fn, text_critique_fn, llm_msg = _build_llm_fns(opts.use_llm_cache)
    if llm_msg:
        diag.append("LLM: " + llm_msg)
    else:
        diag.append(f"LLM: chat_model={settings.llm.chat_model}, coder={settings.llm.coder_model}")

    # 2. 向量库
    vstore, vs_msg = _build_vector_store(opts.use_vector_store)
    diag.append("向量库: " + vs_msg)

    # 3. 渲染后端
    compiler, renderer, backend_label = _build_render_backend(opts.render_backend)
    diag.append("渲染后端: " + backend_label)

    # 4. 装配 Analyzer
    analyzer = ShaderAnalyzer(
        vector_store=vstore,
        llm_fn=chat_fn,
        walkthrough_llm=json_fn,
        summary_llm=json_fn,
        effect_llm=chat_fn,
        compare_llm=chat_fn,
        model_name=settings.llm.chat_model,
        top_k=opts.top_k,
        strategy="fourstage",
    )

    # 5. 装配 Generator
    generator = ShaderGenerator(
        vector_store=vstore,
        llm_fn=code_fn,
        compiler=compiler,
        renderer=renderer,
        critique_fn=vision_fn,
        text_critique_fn=text_critique_fn,
        # 自评只要有"文本自评"或"多模态自评"任一可用即可开启；
        # 没有多模态模型时，文本自评仍能分析编译错误与 spec 吻合度。
        enable_self_critique=opts.enable_self_critique and (
            text_critique_fn is not None or vision_fn is not None
        ),
        model_name=settings.llm.coder_model,
        max_fix_loops=opts.max_fix_loops,
        top_k=opts.top_k,
    )

    asm = _Assembly(
        analyzer=analyzer,
        generator=generator,
        orchestrator=Orchestrator(analyzer=analyzer, generator=generator),
        backend_label=backend_label,
        vstore_label=vs_msg,
        diagnostics=diag,
    )
    _ASSEMBLY_CACHE[key] = asm
    return asm


def clear_assembly_cache() -> int:
    n = len(_ASSEMBLY_CACHE)
    _ASSEMBLY_CACHE.clear()
    return n


# ============================================================
# 工具：把 GLSL 代码渲染成 PNG bytes
# ============================================================

def render_code_to_png(code: str, opts: AssemblyOptions,
                       *, width: int = 768, height: int = 576,
                       time_s: float = 1.5) -> tuple[Optional[bytes], str]:
    """同步渲染一帧。返回 (png_bytes 或 None, 错误说明)。

    分辨率说明：预览默认 768×576（4:3）。之前 512×384 在 Gradio 里被放大显示
    会偏糊；提到 768×576 后清晰度明显改善，单帧渲染仍是几十毫秒级别。
    """
    if not code or not code.strip():
        return None, "代码为空"
    asm = get_assembly(opts)
    renderer = asm.generator._renderer  # type: ignore[attr-defined]
    if renderer is None:
        return None, "渲染器未装配"
    try:
        png = renderer.render(code, width=width, height=height, time=time_s)
        if isinstance(png, (bytes, bytearray)):
            return bytes(png), ""
        return None, f"渲染器返回非 bytes: {type(png).__name__}"
    except Exception as e:
        return None, f"渲染失败: {type(e).__name__}: {e}"


def png_to_pil(png_bytes: bytes | None):
    """Gradio Image 组件最稳的输入是 PIL.Image；这里做一层转换。"""
    if not png_bytes:
        return None
    try:
        from PIL import Image
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception as e:
        logger.warning(f"[ui] PNG → PIL 失败: {e}")
        return None


# ============================================================
# 任务 1：仅分析
# ============================================================

_SEED_CODE_CACHE: dict[str, str] | None = None


def _seed_code_index() -> dict[str, str]:
    """{ shader_id: 完整 Image-pass 源码 }，懒加载并缓存。

    seed shaders 全在内存里，带 code_image；用作"对照参考样本"的源码来源。
    取不到（比如来自外部语料且未落盘）的样本，UI 会回退到 code_excerpt。
    """
    global _SEED_CODE_CACHE
    if _SEED_CODE_CACHE is None:
        try:
            from shader_agent.corpus.seed_shaders import get_seed_shaders
            _SEED_CODE_CACHE = {
                s.shader_id: (s.code_image or "") for s in get_seed_shaders()
            }
        except Exception:
            _SEED_CODE_CACHE = {}
    return _SEED_CODE_CACHE


def _references_with_source(similar: list[Any]) -> list[dict[str, Any]]:
    """把 SimilarShader 列表转成带完整源码的轻量 dict，供 UI 源码对照展示。

    每项：{ shader_id, name, distance, tags, code, is_excerpt }
    code 取源优先级：seed 内存缓存 → 本地语料库磁盘 → code_excerpt 片段。
    """
    seed_index = _seed_code_index()
    out: list[dict[str, Any]] = []
    for s in similar or []:
        sid = getattr(s, "shader_id", "") or ""
        excerpt = getattr(s, "code_excerpt", "") or ""

        # 1) seed 内存缓存
        full = seed_index.get(sid, "")
        # 2) 本地语料库磁盘兜底
        if not full and sid:
            full = _load_clean_code(sid)
        out.append({
            "shader_id": sid,
            "name": getattr(s, "name", "") or sid or "(未命名)",
            "distance": round(float(getattr(s, "distance", 0.0) or 0.0), 4),
            "tags": list(getattr(s, "tags_topic", []) or []),
            "code": full or excerpt,
            "is_excerpt": (not full) and bool(excerpt),
        })
    return out


_CORPUS_CACHE: dict[str, str] = {}


def _load_clean_code(shader_id: str) -> str:
    """从 data/shadertoy_corpus/clean/{id}.json 读取完整代码（带缓存）。"""
    if shader_id in _CORPUS_CACHE:
        return _CORPUS_CACHE[shader_id]
    from pathlib import Path
    import json
    path = Path("data") / "shadertoy_corpus" / "clean" / f"{shader_id}.json"
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                code = json.load(f).get("code_image", "") or ""
            _CORPUS_CACHE[shader_id] = code
            return code
    except Exception:
        pass
    _CORPUS_CACHE[shader_id] = ""
    return ""


def run_analyze(code: str, opts: AssemblyOptions) -> dict[str, Any]:
    """分析 + 渲染原始 shader。返回 dict 给 UI 取字段：
       { ok, error, report_md, report_json, image, elapsed_ms,
         diagnostics, references }"""
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "ok": False, "error": "",
        "report_md": "", "report_json": {},
        "image": None, "elapsed_ms": 0.0,
        "diagnostics": [],
        "references": [],   # 检索到的对照参考样本（含完整源码，供 UI 展示）
    }
    if not code or not code.strip():
        out["error"] = "请输入 GLSL 代码"
        return out
    try:
        asm = get_assembly(opts)
        out["diagnostics"] = list(asm.diagnostics)
        result = asm.orchestrator.analyze_only(code)
        report: AnalysisReport | None = result.get("report")
        if report is None:
            out["error"] = "Analyzer 未产出 report"
            return out
        out["report_md"] = report.to_markdown()
        out["report_json"] = report.model_dump()
        # 把检索到的相似样本补成"带完整源码"的引用，供 UI 做源码对照展示。
        # SimilarShader 自身只带 code_excerpt；完整代码按 shader_id 从 seed 取，
        # 取不到则回退 excerpt。不改动任何 agent / orchestrator 逻辑。
        out["references"] = _references_with_source(report.similar_shaders)
        # 顺手渲染源码（便于"分析"也能看到效果）
        png, png_err = render_code_to_png(code, opts)
        if png_err:
            out["diagnostics"].append("分析侧渲染: " + png_err)
        out["image"] = png_to_pil(png)
        out["ok"] = True
    except Exception as e:
        logger.exception("[ui] run_analyze 异常")
        out["error"] = f"{type(e).__name__}: {e}"
        out["diagnostics"].append(traceback.format_exc(limit=2))
    out["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


# ============================================================
# 任务 2：仅生成
# ============================================================

def run_generate(user_text: str, opts: AssemblyOptions,
                 *, palette: str = "", complexity: str = "",
                 dynamic: bool = True) -> dict[str, Any]:
    """生成 + 渲染。返回 dict：
       { ok, error, code, explanation, compile_ok, compile_errors,
         iterations, critique, image, elapsed_ms, diagnostics, references }"""
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "ok": False, "error": "",
        "code": "", "explanation": "",
        "compile_ok": False, "compile_errors": "",
        "iterations": 0, "critique": "",
        "image": None, "elapsed_ms": 0.0,
        "diagnostics": [], "references": [],
    }
    if not user_text or not user_text.strip():
        out["error"] = "请输入需求"
        return out
    try:
        asm = get_assembly(opts)
        out["diagnostics"] = list(asm.diagnostics)
        # 用 spec 直接喂 generator，比让 ParseSpec 二次解析更精确，
        # 同时把 UI 控件值带进去。
        spec = GenerationSpec(
            description=user_text,
            palette=palette,
            complexity=complexity or "simple",  # type: ignore[arg-type]
            dynamic=dynamic,
        )
        msg_out = asm.generator.handle(spec.to_message())
        if msg_out.payload_type != GeneratedShader.PAYLOAD_TYPE:
            out["error"] = f"Generator 返回非预期消息: {msg_out.content[:120]}"
            return out
        g = GeneratedShader(**msg_out.payload)
        out["code"] = g.code
        out["explanation"] = g.explanation
        out["compile_ok"] = bool(g.compile_result.ok)
        out["compile_errors"] = g.compile_result.errors or ""
        out["iterations"] = g.iterations
        if g.self_critique_rationale:
            out["critique"] = (
                f"score={g.self_critique_score:.2f} · {g.self_critique_rationale}"
            )
        out["references"] = [
            {"shader_id": s.shader_id, "name": s.name,
             "distance": round(s.distance, 4),
             "tags": s.tags_topic}
            for s in (g.references_used or [])
        ]
        png, png_err = render_code_to_png(g.code, opts)
        if png_err:
            out["diagnostics"].append("生成侧渲染: " + png_err)
        out["image"] = png_to_pil(png)
        out["ok"] = True
    except Exception as e:
        logger.exception("[ui] run_generate 异常")
        out["error"] = f"{type(e).__name__}: {e}"
        out["diagnostics"].append(traceback.format_exc(limit=2))
    out["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


# ============================================================
# 任务 3：协作 — 先分析再改写
# ============================================================

def run_collaborate(code: str, ask: str, opts: AssemblyOptions) -> dict[str, Any]:
    """analyze_then_generate。返回 dict 包含 report_md / new_code /
       原始 + 新版 image / iterations / critique。"""
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "ok": False, "error": "",
        "report_md": "", "report_json": {},
        "new_code": "", "new_explanation": "",
        "compile_ok": False, "compile_errors": "",
        "iterations": 0, "critique": "",
        "image_before": None, "image_after": None,
        "elapsed_ms": 0.0, "diagnostics": [],
    }
    if not code or not code.strip():
        out["error"] = "请输入参考的 GLSL 代码"
        return out
    if not ask or not ask.strip():
        out["error"] = "请输入改写指令（例如：保留 raymarching 算法但换成霓虹紫主题）"
        return out
    try:
        asm = get_assembly(opts)
        out["diagnostics"] = list(asm.diagnostics)

        # 先渲染原始
        png0, _ = render_code_to_png(code, opts)
        out["image_before"] = png_to_pil(png0)

        result = asm.orchestrator.analyze_then_generate(code, ask)
        report: AnalysisReport | None = result.get("report")
        gen: GeneratedShader | None = result.get("generated")
        if gen is None:
            out["error"] = "Remixer 未产出改写后的 shader"
            return out
        # 改写模式默认跑一次轻量分析（single 策略，一次 LLM 调用）；
        # 输出精简版"原代码简析"，格式与 Generator/Remixer 解释一致（3~6 句）。
        if report is not None:
            summary = report.algorithm_summary or ""
            techniques = ", ".join(report.techniques) if report.techniques else "通用"
            out["report_md"] = (
                f"**原代码简析** · 技术标签：{techniques}\n\n"
                f"{summary}"
            )
            out["report_json"] = report.model_dump()
        else:
            out["report_md"] = (
                "_本次改写未单独分析原代码。_\n\n"
                "下方「改写说明」已概述改动要点。"
            )
            out["report_json"] = {}
        out["new_code"] = gen.code
        out["new_explanation"] = gen.explanation
        out["compile_ok"] = bool(gen.compile_result.ok)
        out["compile_errors"] = gen.compile_result.errors or ""
        out["iterations"] = gen.iterations
        if gen.self_critique_rationale:
            out["critique"] = (
                f"score={gen.self_critique_score:.2f} · "
                f"{gen.self_critique_rationale}"
            )
        # 新版渲染
        png1, png_err = render_code_to_png(gen.code, opts)
        if png_err:
            out["diagnostics"].append("协作侧新版渲染: " + png_err)
        out["image_after"] = png_to_pil(png1)
        out["ok"] = True
    except Exception as e:
        logger.exception("[ui] run_collaborate 异常")
        out["error"] = f"{type(e).__name__}: {e}"
        out["diagnostics"].append(traceback.format_exc(limit=2))
    out["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


# ============================================================
# 落盘：session 产物（便于事后审查）
# ============================================================

def save_session(name: str, payload: dict[str, Any]) -> str:
    """把一次 UI 运行的产物落到 data/reports/ui_session_{ts}_{name}/。"""
    import json
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = settings.project_root / "data" / "reports" / f"ui_session_{ts}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable: dict[str, Any] = {}
    for k, v in payload.items():
        if k.startswith("image"):
            continue  # 图像走单独的 png 文件
        try:
            json.dumps(v, ensure_ascii=False)
            serializable[k] = v
        except Exception:
            serializable[k] = str(v)
    (out_dir / "payload.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if payload.get("report_md"):
        (out_dir / "report.md").write_text(payload["report_md"], encoding="utf-8")
    if payload.get("code"):
        (out_dir / "generated.glsl").write_text(payload["code"], encoding="utf-8")
    if payload.get("new_code"):
        (out_dir / "rewritten.glsl").write_text(payload["new_code"], encoding="utf-8")
    for img_key in ("image", "image_before", "image_after"):
        img = payload.get(img_key)
        if img is not None:
            try:
                img.save(out_dir / f"{img_key}.png")
            except Exception as e:
                logger.warning(f"[ui] 保存 {img_key} 失败: {e}")
    return str(out_dir)
