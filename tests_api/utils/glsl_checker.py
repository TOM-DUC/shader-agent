"""测试侧的 GLSL 规则校验（与服务端 `service/glsl_rules.py` **独立实现**）。

刻意不复用服务端那份：如果测试直接调用被测代码的规则函数，那就是"用同一套逻辑
验证自己"，规则写错时两边一起错，测试永远绿。这里按"Shadertoy 上贴进去能不能
跑"这个外部标准重新实现一遍，两份实现互为交叉验证。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAIN_IMAGE = re.compile(
    r"void\s+mainImage\s*\(\s*out\s+vec4\s+\w+\s*,\s*(?:in\s+)?vec2\s+\w+\s*\)")
CHANNEL = re.compile(r"iChannel\d|sampler2D|samplerCube|iChannelResolution")
ES3_ONLY = re.compile(r"\b(round|roundEven|trunc)\s*\(")
FENCE = re.compile(r"```")
CUSTOM_UNIFORM = re.compile(r"^\s*uniform\s+\w+\s+(\w+)", re.MULTILINE)
BUILTIN_UNIFORMS = {"iResolution", "iTime", "iTimeDelta", "iFrame", "iMouse",
                    "iDate", "iSampleRate", "iFrameRate"}
VEC3_LITERAL = re.compile(
    r"vec3\s*\(\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*\)")

# 调色板 → 期望的主色分量顺序（哪个通道应当最大）
PALETTE_DOMINANT = {
    "blue": "b",
    "purple": "b",
    "warm": "r",
    "green": "g",
}


@dataclass
class ShaderIssue:
    rule: str
    message: str


def check_shader(code: str, *, dynamic: bool | None = None) -> list[ShaderIssue]:
    """返回问题列表，空列表代表符合 Shadertoy 可运行契约。"""
    issues: list[ShaderIssue] = []
    src = code or ""

    if not src.strip():
        return [ShaderIssue("empty", "代码为空")]
    if not MAIN_IMAGE.search(src):
        issues.append(ShaderIssue("entry", "缺少合法的 mainImage(out vec4, in vec2) 入口"))
    if CHANNEL.search(src):
        issues.append(ShaderIssue("channel", "引用了 iChannel/sampler，本地单通道环境不支持"))
    m = ES3_ONLY.search(src)
    if m:
        issues.append(ShaderIssue("es3", f"使用了 WebGL 1.0 不支持的 {m.group(1)}()"))
    if FENCE.search(src):
        issues.append(ShaderIssue("fence", "残留 markdown 代码围栏"))
    for um in CUSTOM_UNIFORM.finditer(src):
        if um.group(1) not in BUILTIN_UNIFORMS:
            issues.append(ShaderIssue("uniform", f"自定义 uniform `{um.group(1)}`"))
            break
    if src.count("{") != src.count("}"):
        issues.append(ShaderIssue("braces", "花括号不配对"))
    if src.count("(") != src.count(")"):
        issues.append(ShaderIssue("parens", "圆括号不配对"))
    if dynamic is True and "iTime" not in src:
        issues.append(ShaderIssue("dynamic", "要求动态但未使用 iTime"))
    if dynamic is False and "iTime" in src:
        issues.append(ShaderIssue("static", "要求静态但使用了 iTime"))
    return issues


def assert_shader_ok(code: str, *, dynamic: bool | None = None) -> None:
    issues = check_shader(code, dynamic=dynamic)
    assert not issues, (
        "生成的 shader 不符合可运行契约：\n"
        + "\n".join(f"  - [{i.rule}] {i.message}" for i in issues)
        + f"\n--- 代码 ---\n{code[:1500]}"
    )


def dominant_channel_of_code(code: str) -> str | None:
    """从代码里的基色常量推断主色通道，用于校验"生成结果符合调色板"。"""
    hits = VEC3_LITERAL.findall(code or "")
    for r, g, b in reversed(hits):
        try:
            vals = {"r": float(r), "g": float(g), "b": float(b)}
        except ValueError:
            continue
        if all(0.0 <= v <= 1.0 for v in vals.values()) and any(vals.values()):
            return max(vals, key=lambda k: vals[k])
    return None


def assert_palette(code: str, palette_key: str) -> None:
    want = PALETTE_DOMINANT.get(palette_key)
    if want is None:
        return
    got = dominant_channel_of_code(code)
    assert got == want, (
        f"调色板 {palette_key} 期望主色通道 {want}，实际 {got}\n"
        f"--- 代码 ---\n{code[:800]}")


def structural_similarity(before: str, after: str) -> float:
    """改写前后的行级保留率，用来验证"最小改动"而不是整段重写。"""
    a = [ln.strip() for ln in (before or "").splitlines() if ln.strip()]
    b = {ln.strip() for ln in (after or "").splitlines() if ln.strip()}
    if not a:
        return 0.0
    kept = sum(1 for ln in a if ln in b)
    return kept / len(a)
