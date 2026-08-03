"""无 GPU 环境下的确定性编译器 / 渲染器桩。

项目原有的 `rendering.mock.MockRenderer` 只返回一个 1×1 红色像素，够跑通流程，
但**没法对图像做任何断言**——渲染是这个项目最核心的产出，测试如果只能验证
"返回了 bytes"，质量保障就是空的。

这里的 `StubRenderer` 用 numpy 复算一遍与桩 shader 等价的图像：
- 从代码里解析基色向量 `vec3(r,g,b)`，按同样的距离场公式生成像素；
- 同一段代码 + 同一个 iTime ⇒ 同一张图（可做图像回归 / 感知哈希基线）；
- 不同 iTime ⇒ 不同图（可验证 dynamic=true 确实产生了动画）；
- 支持 `renderer_mode=blank` 产出纯黑图，用于验证"渲染成功但画面全黑"这类
  只有图像层校验才抓得到的缺陷。

真 GL 环境（本地 / nightly）下同一批断言直接跑在 moderngl 输出上，标记为
`@pytest.mark.gpu`。
"""
from __future__ import annotations

import io
import re
import time as _time

from shader_agent.agents.schemas import CompileResult
from shader_agent.rendering.mock import MockCompiler
from shader_agent.testing import faults

_VEC3_RE = re.compile(
    r"vec3\s*\(\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*\)"
)
_UNSUPPORTED_RE = re.compile(
    r"\biChannel[0-9]\b|\bsampler(2D|Cube|3D)\b|\biChannelResolution\b|\biChannelTime\b"
)


class RendererUnavailable(RuntimeError):
    """模拟渲染后端整体不可用（无 GL、驱动崩溃）。"""


class StubCompiler:
    """在 MockCompiler 之上叠加故障注入。"""

    def __init__(self) -> None:
        self._inner = MockCompiler()
        self.compile_calls = 0

    def compile(self, user_code: str) -> CompileResult:
        self.compile_calls += 1
        if faults.should_fail("compiler", "compiler_mode", "compiler_fail_times"):
            return CompileResult(
                ok=False,
                errors="OpenGL fragment shader compile error (injected):\n"
                       "0(12)(user:3): error C1503: undefined variable \"injected_fault\"",
            )
        return self._inner.compile(user_code)


class StubRenderer:
    """确定性离屏渲染桩：产出与桩 shader 语义一致的 PNG。"""

    DEFAULT_W = 512
    DEFAULT_H = 384

    def __init__(self) -> None:
        self.render_calls = 0
        self.last_code = ""

    def render(
        self,
        user_code: str,
        *,
        width: int = DEFAULT_W,
        height: int = DEFAULT_H,
        time: float = 1.5,
        frame: int = 90,
    ) -> bytes:
        cfg = faults.current()
        self.render_calls += 1
        self.last_code = user_code or ""

        if cfg.renderer_latency_ms:
            _time.sleep(cfg.renderer_latency_ms / 1000.0)
        if cfg.renderer_mode == "unavailable":
            raise RendererUnavailable(
                "创建 standalone GL context 失败：no usable display (injected)"
            )
        if cfg.renderer_mode == "slow":
            _time.sleep(0.5)

        if _UNSUPPORTED_RE.search(user_code or ""):
            raise RuntimeError(
                "该 shader 使用了多通道/纹理输入（iChannel0~3、sampler2D 等），"
                "本地预览仅支持无外部纹理的单通道 Image shader。"
            )
        if "mainImage" not in (user_code or ""):
            raise RuntimeError("GLSL compile failed during render: missing mainImage")

        blank = cfg.renderer_mode == "blank"
        return _render_png(user_code, width, height, time, blank=blank)


def _base_color(code: str) -> tuple[float, float, float]:
    """取代码中最后一个 vec3 常量作为基色（生成模板里基色在 fragColor 之前）。"""
    hits = _VEC3_RE.findall(code or "")
    for r, g, b in reversed(hits):
        try:
            vals = (float(r), float(g), float(b))
        except ValueError:
            continue
        if any(v > 0.0 for v in vals) and all(0.0 <= v <= 1.0 for v in vals):
            return vals
    return (0.5, 0.5, 0.5)


def _render_png(code: str, width: int, height: int, t: float, *, blank: bool) -> bytes:
    import numpy as np
    from PIL import Image

    width = max(1, int(width))
    height = max(1, int(height))
    if blank:
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        return _encode(Image.fromarray(arr))

    r, g, b = _base_color(code)
    dynamic = "iTime" in (code or "")
    xs = (np.arange(width) + 0.5 - 0.5 * width) / float(height)
    ys = (np.arange(height) + 0.5 - 0.5 * height) / float(height)
    gx, gy = np.meshgrid(xs, ys)
    d = np.sqrt(gx * gx + gy * gy)
    phase = (t * 1.5) if dynamic else 0.0
    wave = 0.5 + 0.5 * np.sin(6.2831 * d * 3.0 - phase)
    base = np.array([r, g, b], dtype="f4").reshape(1, 1, 3)
    col = base * wave[:, :, None] + 0.06 * base
    arr = np.clip(col, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return _encode(Image.fromarray(arr, mode="RGB"))


def _encode(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()
