"""Shader 代码规则校验（不依赖 GL，纯静态）。

定位：介于"接口返回 200"和"GLSL 真编译通过"之间的一层。很多缺陷在编译层是
发现不了的——比如模型把 markdown 围栏一起吐出来、偷偷引用了 `iChannel0`
（本地单通道环境根本没有）、或者用了 WebGL 1.0 不支持的 `round()`
（桌面 GL 编得过，贴回 Shadertoy 就红）。

规则分两级：
  - error : 必然导致不可用，接口应拦截 / 测试应判失败
  - warn  : 可用但不合预期，进报告不阻断
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

MAX_CODE_CHARS = 20000
MAX_CODE_LINES = 800

_MAIN_IMAGE_RE = re.compile(
    r"\bvoid\s+mainImage\s*\(\s*out\s+vec4\s+\w+\s*,\s*(?:in\s+)?vec2\s+\w+\s*\)"
)
_MAIN_IMAGE_LOOSE_RE = re.compile(r"\bvoid\s+mainImage\s*\(")
_CHANNEL_RE = re.compile(r"\biChannel[0-9]\b|\bsampler(?:2D|Cube|3D)\b|"
                         r"\biChannelResolution\b|\biChannelTime\b")
_ES3_ONLY_RE = re.compile(r"\b(round|roundEven|trunc|isnan|isinf|textureGather)\s*\(")
_FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)
_VERSION_RE = re.compile(r"^\s*#version\b", re.MULTILINE)
_MAIN_RE = re.compile(r"\bvoid\s+main\s*\(\s*(?:void)?\s*\)")
_CUSTOM_UNIFORM_RE = re.compile(r"^\s*uniform\s+\w+\s+(\w+)", re.MULTILINE)
_ALLOWED_UNIFORMS = {
    "iResolution", "iTime", "iTimeDelta", "iFrame", "iMouse", "iDate",
    "iSampleRate", "iFrameRate",
}
_WHILE_TRUE_RE = re.compile(r"\bwhile\s*\(\s*true\s*\)")


@dataclass
class RuleViolation:
    rule_id: str
    level: str          # "error" | "warn"
    message: str
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_glsl(code: str, *, require_dynamic: bool | None = None) -> list[RuleViolation]:
    """返回规则违例列表；空列表代表全部通过。"""
    v: list[RuleViolation] = []
    src = code or ""

    if not src.strip():
        return [RuleViolation("GLSL000", "error", "代码为空")]

    if len(src) > MAX_CODE_CHARS:
        v.append(RuleViolation(
            "GLSL001", "error",
            f"代码长度 {len(src)} 超过上限 {MAX_CODE_CHARS}"))
    if src.count("\n") + 1 > MAX_CODE_LINES:
        v.append(RuleViolation("GLSL002", "warn", f"代码行数超过 {MAX_CODE_LINES}"))

    if not _MAIN_IMAGE_LOOSE_RE.search(src):
        v.append(RuleViolation(
            "GLSL010", "error", "缺少 mainImage 入口",
            "Shadertoy Image pass 必须实现 void mainImage(out vec4, in vec2)"))
    elif not _MAIN_IMAGE_RE.search(src):
        v.append(RuleViolation(
            "GLSL011", "error", "mainImage 签名不合法",
            "签名必须为 void mainImage(out vec4 fragColor, in vec2 fragCoord)"))

    if _MAIN_RE.search(src):
        v.append(RuleViolation(
            "GLSL012", "warn", "同时定义了 void main()",
            "包装层会自行生成 main()，用户代码里不应再定义"))

    if _CHANNEL_RE.search(src):
        v.append(RuleViolation(
            "GLSL020", "error", "引用了多通道/纹理输入（iChannelN / sampler2D）",
            "本地离屏渲染只支持无外部纹理的单通道 Image shader"))

    for m in _ES3_ONLY_RE.finditer(src):
        v.append(RuleViolation(
            "GLSL021", "warn", f"使用了 WebGL 1.0 不支持的函数 {m.group(1)}()",
            "改用 floor(x + 0.5) 等兼容写法，否则贴回 Shadertoy 会报错"))
        break

    for m in _CUSTOM_UNIFORM_RE.finditer(src):
        if m.group(1) not in _ALLOWED_UNIFORMS:
            v.append(RuleViolation(
                "GLSL022", "error", f"声明了自定义 uniform `{m.group(1)}`",
                "只允许使用 Shadertoy 内置 uniform"))
            break

    if _FENCE_RE.search(src):
        v.append(RuleViolation(
            "GLSL030", "error", "代码中残留 markdown 代码围栏 ```",
            "模型输出未清洗干净"))
    if _VERSION_RE.search(src):
        v.append(RuleViolation(
            "GLSL031", "warn", "代码中出现 #version 指令",
            "版本号由包装层注入，用户代码不应重复声明"))

    for opener, closer, rid in (("{", "}", "GLSL040"), ("(", ")", "GLSL041")):
        if src.count(opener) != src.count(closer):
            v.append(RuleViolation(
                rid, "error",
                f"括号不配对：`{opener}` {src.count(opener)} 个 / "
                f"`{closer}` {src.count(closer)} 个"))

    if _WHILE_TRUE_RE.search(src) and "break" not in src:
        v.append(RuleViolation(
            "GLSL050", "error", "存在没有 break 的 while(true) 死循环",
            "会导致 GPU 挂起或驱动重置"))

    if require_dynamic is True and "iTime" not in src:
        v.append(RuleViolation(
            "GLSL060", "warn", "需求要求动态效果，但代码未使用 iTime"))
    if require_dynamic is False and "iTime" in src:
        v.append(RuleViolation(
            "GLSL061", "warn", "需求要求静态效果，但代码使用了 iTime"))

    return v


def errors_of(violations: list[RuleViolation]) -> list[RuleViolation]:
    return [x for x in violations if x.level == "error"]


def summarize(violations: list[RuleViolation]) -> dict[str, Any]:
    errs = errors_of(violations)
    return {
        "passed": not errs,
        "n_error": len(errs),
        "n_warn": len(violations) - len(errs),
        "violations": [x.to_dict() for x in violations],
    }
