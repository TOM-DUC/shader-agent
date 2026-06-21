"""GLSLCompiler：只编译，不渲染。

为什么单独做？的 Generator 修正循环每轮都要 validate，
但渲染（fbo + readback）相对昂贵。只编译的话单次约 5-30ms。
"""
from __future__ import annotations

from typing import Optional

from shader_agent.agents.schemas import CompileResult
from shader_agent.rendering.gl_worker import run_on_gl
from shader_agent.rendering.shadertoy_wrap import (
    VERTEX_SHADER_330,
    map_line_number,
    wrap_shadertoy_fragment,
)
from shader_agent.utils.logger import logger


class _ContextHolder:
    """共享一个 moderngl standalone context。"""
    _ctx = None
    _err: str = ""

    @classmethod
    def get(cls):
        if cls._ctx is not None or cls._err:
            return cls._ctx, cls._err
        try:
            import moderngl as mgl
        except ImportError as e:
            cls._err = f"moderngl 未安装：{e}。请 `pip install moderngl Pillow`"
            return None, cls._err
        try:
            cls._ctx = mgl.create_standalone_context(require=330)
        except Exception as e:
            # 给出按平台的安装建议
            cls._err = (
                f"创建 standalone GL context 失败：{e}\n"
                "可能原因：\n"
                "- Linux: 缺 libegl1-mesa 或 libgl1（apt install libegl1-mesa libgl1）\n"
                "- WSL / 容器: 装 mesa-utils + 使用 LIBGL_ALWAYS_SOFTWARE=1\n"
                "- macOS: 通常 OOTB；若失败检查 Python 版本是否 universal2\n"
                "- Windows: 装 glcontext (pip install glcontext)"
            )
            return None, cls._err
        return cls._ctx, ""

    @classmethod
    def release(cls):
        if cls._ctx is not None:
            try:
                cls._ctx.release()
            except Exception:
                pass
            cls._ctx = None


class GLSLCompiler:
    """对 Shadertoy fragment 做真实 GL 编译验证。

    使用：
        compiler, reason = GLSLCompiler.try_create()
        if compiler is None:
            print("renderer unavailable:", reason)
        else:
            cr = compiler.compile(user_code)
            print(cr.ok, cr.errors)
    """

    def __init__(self, ctx) -> None:
        self._ctx = ctx

    @classmethod
    def try_create(cls) -> tuple[Optional["GLSLCompiler"], str]:
        # 关键：context 必须在 GL 专用线程上创建，否则后续别的线程无法使用
        ctx, err = run_on_gl(_ContextHolder.get)
        if ctx is None:
            return None, err
        return cls(ctx), ""

    def compile(self, user_code: str) -> CompileResult:
        # 所有 GL 调用都派发到 GL 专用线程，避免 "cannot create program"
        return run_on_gl(self._compile_impl, user_code)

    def _compile_impl(self, user_code: str) -> CompileResult:
        wrapped_fs = wrap_shadertoy_fragment(user_code)
        try:
            program = self._ctx.program(
                vertex_shader=VERTEX_SHADER_330,
                fragment_shader=wrapped_fs,
            )
            try:
                program.release()
            except Exception:
                pass
            return CompileResult(ok=True, errors="", warnings="")
        except Exception as e:
            raw = str(e)
            mapped = map_line_number(raw, user_code)
            # 加引导，方便 LLM 修正
            errors = (
                "OpenGL fragment shader compile error (line numbers translated to user:N):\n"
                + mapped
            )
            logger.info(f"[compiler] failed: {raw[:200]}")
            return CompileResult(ok=False, errors=errors, warnings="")
