"""Shadertoy → GLSL 330 包装层。

Shadertoy 编辑器只让用户写：
    void mainImage(out vec4 fragColor, in vec2 fragCoord) { ... }

实际编译时它会包成一个完整 fragment shader，注入它的若干内置 uniforms。
我们这里复现该包装，以让用户给我们的代码能在 moderngl 里编译运行。

注入的 uniforms 与 Shadertoy 一致（仅支持无外部纹理的子集，与阶段二的语料库清洗策略对齐）：
    iResolution    vec3
    iTime          float
    iTimeDelta     float
    iFrame         int
    iMouse         vec4
    iDate          vec4
    iSampleRate    float

不支持的 iChannel0..3 / iChannelTime[] / iChannelResolution[] 在 ValidateCodeAction
已被静态拒绝，到这里假设用户代码不会引用它们。
"""
from __future__ import annotations

VERTEX_SHADER_330 = """\
#version 330 core
in vec2 in_position;
out vec2 v_frag_coord;
uniform vec2 u_resolution;
void main() {
    // in_position 是 [-1,1] 的全屏 quad；映射到 fragCoord 像素坐标
    v_frag_coord = (in_position * 0.5 + 0.5) * u_resolution;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""


FRAGMENT_PROLOGUE_330 = """\
#version 330 core
precision highp float;

in vec2 v_frag_coord;
out vec4 _out_fragColor;

uniform vec3  iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int   iFrame;
uniform vec4  iMouse;
uniform vec4  iDate;
uniform float iSampleRate;

// ===== USER CODE BEGIN =====
"""


FRAGMENT_EPILOGUE_330 = """
// ===== USER CODE END =====

void main() {
    vec4 _col = vec4(0.0);
    mainImage(_col, v_frag_coord);
    _out_fragColor = _col;
}
"""


def wrap_shadertoy_fragment(user_code: str) -> str:
    """把 Shadertoy 风格的 fragment 代码包成完整 GLSL 330。"""
    return FRAGMENT_PROLOGUE_330 + (user_code or "") + FRAGMENT_EPILOGUE_330


def map_line_number(wrapped_log: str, user_code: str) -> str:
    """把编译器报告里的"wrapped 文件行号"翻译成"用户代码行号"。

    OpenGL 编译错误格式因驱动而异，常见有：
      "0(15) : error C1503: ..."         NVIDIA
      "0:15: 'x' : undeclared ..."       Mesa / AMD
      "ERROR: 0:15: ..."

    我们做轻量替换：把 `0(N)` 与 `0:N:` 中的 N 减去 prologue 的行数（如果落在用户区）。
    超出用户区的行号不动（让人能看出是包装层错误）。
    """
    import re
    prologue_lines = FRAGMENT_PROLOGUE_330.count("\n")
    user_lines = (user_code or "").count("\n") + 1

    def _fix(n_str: str) -> str:
        n = int(n_str)
        if n <= prologue_lines:
            return f"{n_str}(prologue)"
        rel = n - prologue_lines
        if rel <= user_lines:
            return f"{n_str}(user:{rel})"
        return f"{n_str}(epilogue)"

    out = wrapped_log
    out = re.sub(r"\b0\((\d+)\)", lambda m: f"0({_fix(m.group(1))})", out)
    out = re.sub(r"\b0:(\d+):", lambda m: f"0:{_fix(m.group(1))}:", out)
    return out
