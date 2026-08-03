"""Gradio 回调适配层（重构后）。

**这个文件现在只做一件事：把 ShaderService 的纯数据结果翻译成 Gradio 需要的
形态**（PIL.Image、markdown 字符串、"失败也不抛异常"的 dict）。

重构前它同时承担了依赖装配、业务编排、异常兜底和 UI 适配四件事，导致：
  - HTTP 接口层要复用业务就得反向 import UI 模块；
  - 自动化测试只能通过 Gradio 回调间接验证业务，断言粒度很粗。

现在装配下沉到 `service/assembly.py`，业务下沉到 `service/shader_service.py`，
UI 与 HTTP 接口是同一份业务的两个前端。对外符号（AssemblyOptions /
get_assembly / run_analyze / run_generate / run_collaborate / save_session）
保持不变，`ui/app.py` 无需改动。
"""
from __future__ import annotations

import io
import time
import traceback
from typing import Any, Optional

from shader_agent.config.settings import settings
from shader_agent.observability import (
    get_current_trace_id,
    trace_span,
    update_current_trace,
)
# 装配层已下沉到 service，这里重导出以兼容既有调用方
from shader_agent.service.assembly import (  # noqa: F401
    AssemblyOptions,
    _cache_key,
    clear_assembly_cache,
    get_assembly,
)
from shader_agent.service.errors import ServiceError
from shader_agent.service.shader_service import (  # noqa: F401
    ShaderService,
    _references_with_source,
)
from shader_agent.utils.logger import logger

__all__ = [
    "AssemblyOptions", "get_assembly", "clear_assembly_cache",
    "render_code_to_png", "png_to_pil",
    "run_analyze", "run_generate", "run_collaborate", "save_session",
]


def _service(opts: AssemblyOptions) -> ShaderService:
    return ShaderService(opts)


# ============================================================
# 工具
# ============================================================

def render_code_to_png(code: str, opts: AssemblyOptions,
                       *, width: int = 768, height: int = 576,
                       time_s: float = 1.5) -> tuple[Optional[bytes], str]:
    """同步渲染一帧。返回 (png_bytes 或 None, 错误说明)。"""
    import base64
    try:
        data = _service(opts).render(code, width=width, height=height, time_s=time_s)
        return base64.b64decode(data["image_base64"]), ""
    except ServiceError as e:
        return None, f"渲染失败[{int(e.code)}]: {e.message}"
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


def _image_from_render(render: dict[str, Any] | None):
    if not render or not render.get("ok"):
        return None
    import base64
    try:
        return png_to_pil(base64.b64decode(render["image_base64"]))
    except Exception:
        return None


def _render_note(render: dict[str, Any] | None, prefix: str) -> str:
    if render and not render.get("ok"):
        return f"{prefix}: [{render.get('code')}] {render.get('message')}"
    return ""


# ============================================================
# 任务 1：仅分析
# ============================================================

def run_analyze(code: str, opts: AssemblyOptions) -> dict[str, Any]:
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "ok": False, "error": "", "report_md": "", "report_json": {},
        "image": None, "elapsed_ms": 0.0, "diagnostics": [], "references": [],
    }
    if not code or not code.strip():
        out["error"] = "请输入 GLSL 代码"
        return out
    try:
        res = _service(opts).analyze(
            code, with_render=True, with_reference_code=True)
        out["diagnostics"] = list(res.get("diagnostics") or [])
        out["report_md"] = res["report_md"]
        out["report_json"] = res["report"]
        out["references"] = res["references"]
        note = _render_note(res.get("render"), "分析侧渲染")
        if note:
            out["diagnostics"].append(note)
        out["image"] = _image_from_render(res.get("render"))
        out["ok"] = True
    except ServiceError as e:
        out["error"] = f"[{int(e.code)}] {e.message}"
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
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "ok": False, "error": "", "code": "", "explanation": "",
        "compile_ok": False, "compile_errors": "", "iterations": 0,
        "critique": "", "image": None, "elapsed_ms": 0.0,
        "diagnostics": [], "references": [], "trace_id": "",
    }
    if not user_text or not user_text.strip():
        out["error"] = "请输入需求"
        return out
    try:
        with trace_span("task.ui_generate",
                        input={"prompt": (user_text or "")[:500], "palette": palette,
                               "complexity": complexity, "dynamic": dynamic}) as span:
            update_current_trace(
                name="ui.generate",
                tags=list(settings.observability.tags or []) + ["ui", "generate"],
                metadata={"service": settings.observability.service_name,
                          "environment": settings.observability.environment},
            )
            out["trace_id"] = get_current_trace_id() or ""
            res = _service(opts).generate(
                user_text, palette=palette,
                complexity=complexity or "simple", dynamic=dynamic,
                with_render=True,
            )
            span.update(output={"compile_ok": res["compile_ok"],
                                "iterations": res["iterations"]})
        out["diagnostics"] = list(res.get("diagnostics") or [])
        out["code"] = res["code"]
        out["explanation"] = res["explanation"]
        out["compile_ok"] = res["compile_ok"]
        out["compile_errors"] = res["compile_errors"]
        out["iterations"] = res["iterations"]
        if res.get("self_critique_rationale"):
            out["critique"] = (f"score={res['self_critique_score']:.2f} · "
                               f"{res['self_critique_rationale']}")
        out["references"] = res["references"]
        note = _render_note(res.get("render"), "生成侧渲染")
        if note:
            out["diagnostics"].append(note)
        out["image"] = _image_from_render(res.get("render"))
        out["ok"] = True
    except ServiceError as e:
        out["error"] = f"[{int(e.code)}] {e.message}"
    except Exception as e:
        logger.exception("[ui] run_generate 异常")
        out["error"] = f"{type(e).__name__}: {e}"
        out["diagnostics"].append(traceback.format_exc(limit=2))
    out["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


# ============================================================
# 任务 3：改写
# ============================================================

def run_collaborate(code: str, ask: str, opts: AssemblyOptions) -> dict[str, Any]:
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "ok": False, "error": "", "report_md": "", "report_json": {},
        "new_code": "", "new_explanation": "", "compile_ok": False,
        "compile_errors": "", "iterations": 0, "critique": "",
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
        res = _service(opts).remix(code, ask, analyze_first=True, with_render=True)
        out["diagnostics"] = list(res.get("diagnostics") or [])
        analysis = res.get("analysis")
        if analysis:
            techniques = ", ".join(analysis.get("techniques") or []) or "通用"
            out["report_md"] = (f"**原代码简析** · 技术标签：{techniques}\n\n"
                                f"{analysis.get('algorithm_summary') or ''}")
            out["report_json"] = analysis
        else:
            out["report_md"] = ("_本次改写未单独分析原代码。_\n\n"
                                "下方「改写说明」已概述改动要点。")
        out["new_code"] = res["code"]
        out["new_explanation"] = res["explanation"]
        out["compile_ok"] = res["compile_ok"]
        out["compile_errors"] = res["compile_errors"]
        out["iterations"] = res["iterations"]
        if res.get("self_critique_rationale"):
            out["critique"] = (f"score={res['self_critique_score']:.2f} · "
                               f"{res['self_critique_rationale']}")
        note = _render_note(res.get("render"), "协作侧新版渲染")
        if note:
            out["diagnostics"].append(note)
        out["image_before"] = _image_from_render(res.get("render_before"))
        out["image_after"] = _image_from_render(res.get("render"))
        out["ok"] = True
    except ServiceError as e:
        out["error"] = f"[{int(e.code)}] {e.message}"
    except Exception as e:
        logger.exception("[ui] run_collaborate 异常")
        out["error"] = f"{type(e).__name__}: {e}"
        out["diagnostics"].append(traceback.format_exc(limit=2))
    out["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


# ============================================================
# 落盘：session 产物
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
            continue
        try:
            json.dumps(v, ensure_ascii=False)
            serializable[k] = v
        except Exception:
            serializable[k] = str(v)
    (out_dir / "payload.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
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
