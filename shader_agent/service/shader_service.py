"""ShaderService：业务门面层。

这一层的存在意义是把"业务能做什么"和"谁来调用"彻底分开：

    Gradio UI ─┐
    HTTP API  ─┼─→ ShaderService ─→ Orchestrator / Analyzer / Generator / Renderer
    自动化测试 ─┘

对外只暴露纯数据（dict / bytes），不含任何 Gradio 或 FastAPI 概念；
失败一律抛 `ServiceError`（带稳定错误码），不返回半成品字符串。
这样接口层只做协议转换，测试层可以直接对同一份业务语义做断言。
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from shader_agent.agents.schemas import (
    AnalysisReport,
    GeneratedShader,
    GenerationSpec,
)
from shader_agent.config.settings import settings
from shader_agent.service.assembly import AssemblyOptions, get_assembly
from shader_agent.service.errors import (
    ErrorCode,
    ServiceError,
    classify_retrieval_error,
    classify_upstream_error,
)
from shader_agent.service.glsl_rules import check_glsl, errors_of, summarize
from shader_agent.utils.logger import logger

MAX_CODE_CHARS = 20000
MAX_PROMPT_CHARS = 2000
MAX_RENDER_PIXELS = 1920 * 1080


# ============================================================
# 输入校验（前置，尽早失败，错误码明确）
# ============================================================

def _require_code(code: str, field: str = "code") -> str:
    if not code or not code.strip():
        raise ServiceError(ErrorCode.EMPTY_INPUT, f"`{field}` 不能为空")
    if len(code) > MAX_CODE_CHARS:
        raise ServiceError(
            ErrorCode.INPUT_TOO_LARGE,
            f"`{field}` 长度 {len(code)} 超过上限 {MAX_CODE_CHARS}",
        )
    return code


def _require_text(text: str, field: str, limit: int = MAX_PROMPT_CHARS) -> str:
    if not text or not text.strip():
        raise ServiceError(ErrorCode.EMPTY_INPUT, f"`{field}` 不能为空")
    if len(text) > limit:
        raise ServiceError(
            ErrorCode.INPUT_TOO_LARGE,
            f"`{field}` 长度 {len(text)} 超过上限 {limit}",
        )
    return text


# ============================================================
# 参考样本补全（原在 ui/runners.py，属于业务能力，随门面下沉）
# ============================================================

_SEED_CODE_CACHE: dict[str, str] | None = None
_CORPUS_CACHE: dict[str, str] = {}


def _seed_code_index() -> dict[str, str]:
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


def _load_clean_code(shader_id: str) -> str:
    if shader_id in _CORPUS_CACHE:
        return _CORPUS_CACHE[shader_id]
    path = (settings.project_root / settings.paths.shadertoy_corpus_dir
            / "clean" / f"{shader_id}.json")
    code = ""
    try:
        if path.exists():
            code = json.loads(Path(path).read_text(encoding="utf-8")).get("code_image", "") or ""
    except Exception:
        code = ""
    _CORPUS_CACHE[shader_id] = code
    return code


def _references_with_source(similar: list[Any], *, with_code: bool = True) -> list[dict[str, Any]]:
    seed_index = _seed_code_index() if with_code else {}
    out: list[dict[str, Any]] = []
    for s in similar or []:
        sid = getattr(s, "shader_id", "") or ""
        excerpt = getattr(s, "code_excerpt", "") or ""
        full = ""
        if with_code:
            full = seed_index.get(sid, "") or (_load_clean_code(sid) if sid else "")
        out.append({
            "shader_id": sid,
            "name": getattr(s, "name", "") or sid or "(未命名)",
            "distance": round(float(getattr(s, "distance", 0.0) or 0.0), 4),
            "tags": list(getattr(s, "tags_topic", []) or []),
            "code": (full or excerpt) if with_code else "",
            "is_excerpt": with_code and (not full) and bool(excerpt),
        })
    return out


# ============================================================
# 门面
# ============================================================

class ShaderService:
    """所有 Shader 业务能力的统一入口。"""

    def __init__(self, options: AssemblyOptions | None = None) -> None:
        self.options = options or AssemblyOptions()

    # ---------- 装配 ----------
    def _assembly(self, overrides: dict[str, Any] | None = None):
        opts = self.options
        if overrides:
            clean = {k: v for k, v in overrides.items() if v is not None}
            if clean:
                opts = AssemblyOptions(**{**vars(self.options), **clean})
        return get_assembly(opts)

    # ---------- 健康检查 ----------
    def health(self) -> dict[str, Any]:
        return {"status": "ok", "service": settings.observability.service_name,
                "profile": self.options.resolved_profile()}

    def readiness(self) -> dict[str, Any]:
        """就绪探针：逐依赖上报状态，任一 down 则整体 degraded。"""
        components: dict[str, Any] = {}
        overall = "ok"
        try:
            asm = self._assembly()
            if asm.profile == "test":
                llm_detail = "stub（确定性桩）"
            elif asm.llm_ready:
                llm_detail = settings.llm.chat_model
            else:
                # degraded 必须说清"为什么"。只回一个 degraded 会让值班同学
                # 从探针跳去翻日志，而这里本来就知道答案。凭据脱敏后再输出。
                llm_detail = (
                    "未配置 DEEPSEEK_API_KEY"
                    if not settings.has_llm_credentials
                    else f"客户端装配失败（key={settings.credential_status()['deepseek_api_key']}）"
                )
            components["llm"] = {
                "status": "ok" if asm.llm_ready else "degraded",
                "detail": llm_detail,
            }
            components["retrieval"] = {
                "status": "ok" if asm.retriever is not None else "degraded",
                "detail": asm.vstore_label,
            }
            components["render"] = {
                "status": "ok" if "mock" not in asm.backend_label else "degraded",
                "detail": asm.backend_label,
            }
            components["profile"] = {"status": "ok", "detail": asm.profile}
            if any(c.get("status") != "ok" for c in components.values()):
                overall = "degraded"
        except Exception as e:  # 装配本身炸了才算 down
            logger.exception("[service] readiness 装配失败")
            components["assembly"] = {"status": "down", "detail": f"{type(e).__name__}: {e}"}
            overall = "down"
        return {"status": overall, "components": components}

    # ---------- 能力 1：静态规则校验（不调 LLM、不碰 GL，最快的一层）----------
    def validate(self, code: str, *, require_dynamic: bool | None = None) -> dict[str, Any]:
        _require_code(code)
        return summarize(check_glsl(code, require_dynamic=require_dynamic))

    # ---------- 能力 2：编译 ----------
    def compile(self, code: str) -> dict[str, Any]:
        _require_code(code)
        asm = self._assembly()
        compiler = asm.generator._compiler  # type: ignore[attr-defined]
        if compiler is None:
            raise ServiceError(ErrorCode.RENDER_UNAVAILABLE, "编译器未装配")
        t0 = time.perf_counter()
        try:
            cr = compiler.compile(code)
        except Exception as e:
            raise classify_upstream_error(e) from e
        return {
            "ok": bool(cr.ok),
            "errors": cr.errors or "",
            "warnings": cr.warnings or "",
            "backend": asm.backend_label,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # ---------- 能力 3：渲染 ----------
    def render(self, code: str, *, width: int = 768, height: int = 576,
               time_s: float = 1.5) -> dict[str, Any]:
        _require_code(code)
        if width <= 0 or height <= 0:
            raise ServiceError(ErrorCode.INVALID_PARAM, "width/height 必须为正整数")
        if width * height > MAX_RENDER_PIXELS:
            raise ServiceError(
                ErrorCode.INVALID_PARAM,
                f"分辨率过大：{width}x{height} 超过 {MAX_RENDER_PIXELS} 像素上限")

        violations = errors_of(check_glsl(code))
        unsupported = [v for v in violations if v.rule_id == "GLSL020"]
        if unsupported:
            raise ServiceError(
                ErrorCode.UNSUPPORTED_SHADER, unsupported[0].message,
                detail=[v.to_dict() for v in violations])

        asm = self._assembly()
        renderer = asm.generator._renderer  # type: ignore[attr-defined]
        if renderer is None:
            raise ServiceError(ErrorCode.RENDER_UNAVAILABLE, "渲染器未装配")
        t0 = time.perf_counter()
        try:
            png = renderer.render(code, width=width, height=height, time=time_s)
        except Exception as e:
            text = f"{e}"
            if "compile" in text.lower():
                raise ServiceError(
                    ErrorCode.SHADER_COMPILE_ERROR, "shader 编译失败，无法渲染",
                    detail=text[:2000]) from e
            if "context" in text.lower() or "display" in text.lower():
                raise ServiceError(
                    ErrorCode.RENDER_UNAVAILABLE, f"渲染后端不可用：{text[:300]}",
                    retryable=True) from e
            raise classify_upstream_error(e) from e
        if not isinstance(png, (bytes, bytearray)):
            raise ServiceError(ErrorCode.INTERNAL,
                               f"渲染器返回了非 bytes：{type(png).__name__}")
        return {
            "image_base64": base64.b64encode(bytes(png)).decode("ascii"),
            "format": "png",
            "width": width,
            "height": height,
            "time": time_s,
            "bytes": len(png),
            "backend": asm.backend_label,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # ---------- 能力 4：检索 ----------
    def retrieve(self, query: str, *, top_k: int = 5,
                 tags: list[str] | None = None) -> dict[str, Any]:
        _require_text(query, "query", limit=500)
        if not 1 <= int(top_k) <= 20:
            raise ServiceError(ErrorCode.INVALID_PARAM, "top_k 需在 1~20 之间")
        asm = self._assembly()
        if asm.retriever is None:
            raise ServiceError(ErrorCode.RETRIEVAL_UNAVAILABLE,
                               f"检索不可用：{asm.vstore_label}")
        t0 = time.perf_counter()
        try:
            hits = asm.retriever.retrieve(query, top_k=int(top_k), want_tags=tags or [])
        except Exception as e:
            # 这里已经确定故障来自检索器，用专用归类而不是通用文本猜测：
            # 检索超时被通用归类判成 LLM_TIMEOUT 会把排查方向直接带偏。
            raise classify_retrieval_error(e) from e
        items = [
            {
                "shader_id": h.shader_id,
                "name": h.name,
                "score": round(float(h.fused_score), 4),
                "vec_rel": round(float(h.vec_rel), 4),
                "bm25": round(float(h.bm25_norm), 4),
                "tag_match": round(float(h.tag_match), 4),
                "quality": round(float(h.quality), 4),
                "tags": list(h.tags_topic or []),
                "matched_chunks": list(h.matched_chunks or []),
                "algorithm_summary": h.algorithm_summary or "",
            }
            for h in hits
        ]
        return {
            "query": query,
            "items": items,
            "total": len(items),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # ---------- 能力 5：分析 ----------
    def analyze(self, code: str, *, top_k: int | None = None,
                with_render: bool = False,
                with_reference_code: bool = False) -> dict[str, Any]:
        _require_code(code)
        asm = self._assembly({"top_k": top_k})
        t0 = time.perf_counter()
        try:
            result = asm.orchestrator.analyze_only(code)
        except Exception as e:
            logger.exception("[service] analyze 失败")
            raise classify_upstream_error(e) from e

        report: AnalysisReport | None = result.get("report")
        if report is None:
            raise ServiceError(ErrorCode.LLM_ERROR, "Analyzer 未产出分析报告")

        out: dict[str, Any] = {
            "report": report.model_dump(),
            "report_md": report.to_markdown(),
            "techniques": list(report.techniques or []),
            "references": _references_with_source(
                report.similar_shaders, with_code=with_reference_code),
            "n_references": len(report.similar_shaders or []),
            "model_used": report.model_used or "",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "diagnostics": list(asm.diagnostics),
        }
        if with_render:
            out["render"] = self._safe_render(code)
        return out

    # ---------- 能力 6：生成 ----------
    def generate(self, description: str, *, palette: str = "",
                 complexity: str = "simple", dynamic: bool = True,
                 effect_type: str = "", constraints: list[str] | None = None,
                 max_fix_loops: int | None = None, top_k: int | None = None,
                 enable_self_critique: bool | None = None,
                 with_render: bool = False) -> dict[str, Any]:
        _require_text(description, "description")
        if complexity not in ("minimal", "simple", "moderate", "complex"):
            raise ServiceError(
                ErrorCode.INVALID_PARAM,
                "complexity 只能是 minimal/simple/moderate/complex")
        asm = self._assembly({
            "max_fix_loops": max_fix_loops, "top_k": top_k,
            "enable_self_critique": enable_self_critique,
        })
        spec = GenerationSpec(
            description=description, palette=palette,
            complexity=complexity,  # type: ignore[arg-type]
            dynamic=dynamic, effect_type=effect_type,
            constraints=list(constraints or []),
        )
        t0 = time.perf_counter()
        try:
            msg_out = asm.generator.handle(spec.to_message())
        except Exception as e:
            logger.exception("[service] generate 失败")
            raise classify_upstream_error(e) from e
        if msg_out.payload_type != GeneratedShader.PAYLOAD_TYPE:
            raise ServiceError(
                ErrorCode.GENERATION_FAILED,
                f"Generator 返回了非预期消息：{msg_out.content[:200]}")
        gen = GeneratedShader(**msg_out.payload)
        out = self._generated_payload(gen, t0, asm)
        if with_render and gen.code:
            out["render"] = self._safe_render(gen.code)
        return out

    # ---------- 能力 7：改写 ----------
    def remix(self, code: str, instruction: str, *,
              analyze_first: bool = True, max_fix_loops: int | None = None,
              with_render: bool = False) -> dict[str, Any]:
        _require_code(code)
        _require_text(instruction, "instruction")
        asm = self._assembly({"max_fix_loops": max_fix_loops})
        t0 = time.perf_counter()
        try:
            result = asm.orchestrator.analyze_then_generate(
                code, instruction, analyze_first=analyze_first)
        except Exception as e:
            logger.exception("[service] remix 失败")
            raise classify_upstream_error(e) from e
        gen: GeneratedShader | None = result.get("generated")
        if gen is None:
            raise ServiceError(ErrorCode.GENERATION_FAILED, "Remixer 未产出改写结果")
        out = self._generated_payload(gen, t0, asm)
        report: AnalysisReport | None = result.get("report")
        out["analysis"] = report.model_dump() if report is not None else None
        out["base_code"] = code
        out["instruction"] = instruction
        if with_render and gen.code:
            out["render"] = self._safe_render(gen.code)
            out["render_before"] = self._safe_render(code)
        return out

    # ---------- 内部 ----------
    def _generated_payload(self, gen: GeneratedShader, t0: float, asm) -> dict[str, Any]:
        rule_report = summarize(check_glsl(
            gen.code,
            require_dynamic=(gen.spec.dynamic if gen.spec is not None else None),
        ))
        return {
            "code": gen.code,
            "explanation": gen.explanation,
            "compile_ok": bool(gen.compile_result.ok),
            "compile_errors": gen.compile_result.errors or "",
            "iterations": int(gen.iterations),
            "self_critique_score": float(gen.self_critique_score),
            "self_critique_rationale": gen.self_critique_rationale or "",
            "references": [
                {"shader_id": s.shader_id, "name": s.name,
                 "distance": round(float(s.distance), 4),
                 "tags": list(s.tags_topic or [])}
                for s in (gen.references_used or [])
            ],
            "rule_report": rule_report,
            "model_used": gen.model_used or "",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "diagnostics": list(asm.diagnostics),
        }

    def _safe_render(self, code: str) -> dict[str, Any]:
        """附带渲染：失败不阻断主流程，降级为结构化说明。"""
        try:
            return {"ok": True, **self.render(code)}
        except ServiceError as e:
            return {"ok": False, "code": int(e.code), "message": e.message}
        except Exception as e:  # pragma: no cover - 兜底
            return {"ok": False, "code": int(ErrorCode.INTERNAL), "message": str(e)}


_DEFAULT_SERVICE: ShaderService | None = None


def get_service(options: AssemblyOptions | None = None) -> ShaderService:
    """进程级默认门面（接口层依赖注入的默认实现）。"""
    global _DEFAULT_SERVICE
    if options is not None:
        return ShaderService(options)
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = ShaderService()
    return _DEFAULT_SERVICE


def reset_service() -> None:
    global _DEFAULT_SERVICE
    _DEFAULT_SERVICE = None
