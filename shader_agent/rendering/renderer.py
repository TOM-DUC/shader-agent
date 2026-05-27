"""GLSLRenderer：编译 + 全屏 quad 渲染 + 读回 PNG。

阶段六的核心新能力。配合多模态 LLM 实现 GeneratedShader 的"渲染截图自评"。
"""
from __future__ import annotations

import io
from typing import Optional

from shader_agent.rendering.compiler import _ContextHolder
from shader_agent.rendering.gl_worker import run_on_gl
from shader_agent.rendering.shadertoy_wrap import (
    VERTEX_SHADER_330,
    map_line_number,
    wrap_shadertoy_fragment,
)
from shader_agent.utils.logger import logger


class GLSLRenderer:
    """渲染单帧到 PNG bytes。

    用法：
        r, reason = GLSLRenderer.try_create()
        if r is None:
            ...降级...
        else:
            png = r.render(user_code, width=512, height=512, time=1.5)
    """

    DEFAULT_W = 512
    DEFAULT_H = 384

    def __init__(self, ctx) -> None:
        self._ctx = ctx
        # VBO 延迟到 GL 线程上创建（见 _ensure_vbo）；不要在构造线程上建，
        # 否则与渲染线程不一致会触发 GL 错误。
        self._vbo = None

    def _ensure_vbo(self):
        """在 GL 线程内调用：懒创建并复用全屏 quad VBO。"""
        if self._vbo is not None:
            return self._vbo
        import numpy as np
        verts = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype="f4")
        self._vbo = self._ctx.buffer(verts.tobytes())
        return self._vbo

    @classmethod
    def try_create(cls) -> tuple[Optional["GLSLRenderer"], str]:
        # context 在 GL 专用线程上创建
        ctx, err = run_on_gl(_ContextHolder.get)
        if ctx is None:
            return None, err
        try:
            return cls(ctx), ""
        except Exception as e:
            return None, f"renderer init failed: {e}"

    def render(
        self,
        user_code: str,
        *,
        width: int = DEFAULT_W,
        height: int = DEFAULT_H,
        time: float = 1.5,
        frame: int = 90,
    ) -> bytes:
        """渲染一帧。失败时抛 RuntimeError，含编译器原文。返回 PNG bytes。

        所有 GL 调用都派发到 GL 专用线程，彻底规避线程亲和性问题。
        """
        return run_on_gl(
            self._render_impl, user_code,
            width=width, height=height, time=time, frame=frame,
        )

    def _render_impl(
        self,
        user_code: str,
        *,
        width: int = DEFAULT_W,
        height: int = DEFAULT_H,
        time: float = 1.5,
        frame: int = 90,
    ) -> bytes:
        # 预检：Shadertoy 多通道/纹理特性（iChannelN / sampler2D / 多 buffer）
        # 在本地单通道环境无法支持，提前给出友好提示而非一堆 GL 报错。
        import re as _re
        if _re.search(r"\biChannel[0-9]\b|\bsampler(2D|Cube|3D)\b|"
                      r"\biChannelResolution\b|\biChannelTime\b", user_code or ""):
            raise RuntimeError(
                "该 shader 使用了多通道/纹理输入（iChannel0~3、sampler2D 或 "
                "iChannelResolution 等），本地预览仅支持无外部纹理的单通道 "
                "Image shader。请改用不依赖 iChannel 的版本，或在 Shadertoy "
                "上查看原效果。"
            )
        vbo = self._ensure_vbo()
        wrapped_fs = wrap_shadertoy_fragment(user_code)
        try:
            program = self._ctx.program(
                vertex_shader=VERTEX_SHADER_330, fragment_shader=wrapped_fs,
            )
        except Exception as e:
            raw = str(e)
            mapped = map_line_number(raw, user_code)
            raise RuntimeError(
                f"GLSL compile failed during render:\n{mapped}"
            ) from e

        try:
            # 注入 uniforms
            self._set_uniforms(program, width, height, time, frame)

            vao = self._ctx.vertex_array(
                program, [(vbo, "2f", "in_position")]
            )

            # 离屏 framebuffer
            tex = self._ctx.texture((width, height), 4)  # RGBA8
            fbo = self._ctx.framebuffer(color_attachments=[tex])
            fbo.use()
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)
            import moderngl as mgl
            vao.render(mode=mgl.TRIANGLE_STRIP, vertices=4)

            # 读回像素
            raw = fbo.read(components=4, alignment=1)

            # 转 PNG
            try:
                from PIL import Image
            except ImportError as e:
                raise RuntimeError("Pillow 未安装：pip install Pillow") from e
            img = Image.frombytes("RGBA", (width, height), raw)
            # OpenGL 帧缓冲坐标系 Y 朝上，PIL Y 朝下：上下翻转
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            buf = io.BytesIO()
            # optimize=True 会显著拖慢 PNG 编码（且预览图不需要极致压缩），关闭之
            img.save(buf, format="PNG", compress_level=1)
            return buf.getvalue()
        finally:
            try:
                vao.release()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            try:
                fbo.release()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            try:
                tex.release()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            try:
                program.release()
            except Exception:
                pass

    @staticmethod
    def _set_uniforms(program, w: int, h: int, t: float, frame: int) -> None:
        """尽力设置 Shadertoy 内置 uniforms。未声明的 uniform 在 moderngl 里
        访问 program[name] 会抛 KeyError，所以用 try/except 安全注入。"""
        def _try(name, value):
            try:
                program[name].value = value
            except Exception:
                pass
        _try("iResolution", (float(w), float(h), 1.0))
        _try("iTime", float(t))
        _try("iTimeDelta", 1.0 / 60.0)
        _try("iFrame", int(frame))
        _try("iMouse", (0.0, 0.0, 0.0, 0.0))
        _try("iDate", (2026.0, 5.0, 24.0, 0.0))
        _try("iSampleRate", 44100.0)
        # 顶点着色器要用
        _try("u_resolution", (float(w), float(h)))

    def close(self) -> None:
        def _release():
            try:
                if self._vbo is not None:
                    self._vbo.release()
            except Exception:
                pass
        try:
            run_on_gl(_release)
        except Exception:
            pass
