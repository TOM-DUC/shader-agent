"""Mock 编译器 / 渲染器：单测与无 GL 环境演示。

行为：
- MockCompiler.compile(): 用规则检查（与的 ValidateCodeAction 类似，
  但更严格地模拟"真编译器"，例如检测 undeclared identifier）
- MockRenderer.render(): 返回固定 1x1 PNG，足以让 self_critique 走通

这两个类的目的是让的所有测试可以在 CI / 容器 / 离线环境跑通，
不依赖真 OpenGL。
"""
from __future__ import annotations

import re

from shader_agent.agents.schemas import CompileResult


class MockCompiler:
    """模拟真编译器；比 ValidateCodeAction 更严格。"""

    def __init__(self, *, force_error: str = "") -> None:
        # 测试用：构造时可指定一段强制返回的错误，方便测修正循环
        self._force_error = force_error

    def compile(self, user_code: str) -> CompileResult:
        if self._force_error:
            return CompileResult(ok=False, errors=self._force_error)
        # 简单 lint：检测未声明变量（用 `vec3 foo = bar;` 但 bar 未定义）
        # 这是"真编译器才能发现"的典型错误，用来对比 ValidateCodeAction 的局限
        errors: list[str] = []
        if not re.search(r"\bvoid\s+mainImage\s*\(", user_code):
            errors.append("0(13)(user:1): missing mainImage entry")
        # 检测明显的 vec3 = vec4 / vec2 = vec3 之类的维度不匹配
        for m in re.finditer(r"\bvec3\s+\w+\s*=\s*vec4\(", user_code):
            errors.append(f"0(?)(user:?): cannot assign vec4 to vec3 at `{m.group(0)}`")
        if errors:
            return CompileResult(
                ok=False,
                errors="OpenGL fragment shader compile error (mocked):\n"
                       + "\n".join(errors),
            )
        return CompileResult(ok=True, errors="", warnings="")


class MockRenderer:
    """返回 1x1 RGBA PNG，避免测试依赖真 GL。"""

    # 1x1 红色像素的 PNG（手工构造，89 字节）
    RED_PIXEL_PNG = bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15"
        "C4890000000D49444154789C63F8CF000000020001E221BC330000000049454"
        "E44AE426082"
    )

    def __init__(self) -> None:
        self.render_calls = 0
        self.last_code: str = ""

    def render(self, user_code: str, **kwargs) -> bytes:
        self.render_calls += 1
        self.last_code = user_code
        # 简单识别下"有 mainImage 才返回截图"，否则像真编译器一样抛
        if "mainImage" not in user_code:
            raise RuntimeError("GLSL compile failed during render: missing mainImage")
        return self.RED_PIXEL_PNG
