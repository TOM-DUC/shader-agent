"""确定性 Stub LLM。

为什么需要它：Shader Agent 的每条链路都要调 1~5 次大模型，如果测试直接打真实
DeepSeek，会同时踩三个坑——**不确定**（同 prompt 不同输出，断言只能写得极弱）、
**慢**（一条用例几十秒，无法进 CI）、**贵**。

因此把 LLM 抽成 `Callable[[messages], str]` 后，在 test profile 下换成这里的桩：
- 输出**由 prompt 决定**，同输入必得同输出，可以对内容做强断言；
- 通过 `shader_agent.testing.faults` 可切换成超时 / 限流 / 非法 JSON / 编不过的
  代码等异常形态，用来验证系统的重试、降级与错误映射；
- 记录调用次数，供"多次修复仍失败""重试了几次"这类断言使用。

真实链路的正确性由带 `@pytest.mark.live` 标记的少量冒烟用例覆盖（需要 key，
只在 nightly 跑），日常 CI 全部走桩。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from shader_agent.testing import faults

# --------------------------------------------------------------------
# 调色板 → 基色。让"生成结果符合 palette"成为可断言的确定性事实。
# --------------------------------------------------------------------
PALETTE_RGB: dict[str, tuple[float, float, float]] = {
    "blue": (0.20, 0.45, 0.95),
    "purple": (0.62, 0.25, 0.95),
    "warm": (0.98, 0.42, 0.18),
    "green": (0.20, 0.85, 0.45),
    "mono": (0.75, 0.75, 0.75),
    "default": (0.35, 0.60, 0.90),
}

_PALETTE_ALIASES: list[tuple[tuple[str, ...], str]] = [
    (("蓝", "blue", "青", "cyan", "海洋"), "blue"),
    (("紫", "purple", "violet", "霓虹", "neon"), "purple"),
    (("暖", "warm", "红", "red", "橙", "orange", "sunset", "日落", "火"), "warm"),
    (("绿", "green", "森林", "emerald"), "green"),
    (("单色", "灰", "mono", "grey", "gray", "黑白"), "mono"),
]


def palette_key(text: str) -> str:
    """把任意调色板描述归一到受控词表，供生成与断言共用。"""
    t = (text or "").lower()
    for words, key in _PALETTE_ALIASES:
        if any(w in t for w in words):
            return key
    return "default"


def _spec_field(prompt: str, field: str, default: str = "") -> str:
    m = re.search(rf"^-\s*{field}\s*:\s*(.*)$", prompt, flags=re.MULTILINE)
    return (m.group(1).strip() if m else default)


def _join(messages: list[dict[str, str]]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages or [])


# --------------------------------------------------------------------
# GLSL 模板
# --------------------------------------------------------------------

_VALID_TEMPLATE = """void mainImage( out vec4 fragColor, in vec2 fragCoord )
{{
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float d = length(uv);
    float wave = 0.5 + 0.5 * sin(6.2831 * d * 3.0 - {time_term});
    vec3 base = vec3({r:.3f}, {g:.3f}, {b:.3f});
    vec3 col = base * wave + 0.06 * base;
    fragColor = vec4(col, 1.0);
}}"""

# 维度不匹配：MockCompiler 与真 GL 都会拒绝，用来驱动修复循环
_BROKEN_TEMPLATE = """void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = vec4(uv, 0.5, 1.0);
    fragColor = vec4(col, 1.0);
}"""

# 使用了本地不支持的多通道纹理，用来验证规则校验拦截
_UNSUPPORTED_TEMPLATE = """void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 tex = texture(iChannel0, uv);
    fragColor = vec4(tex.rgb, 1.0);
}"""


def build_shader(palette: str = "", dynamic: bool = True) -> str:
    """按调色板与动态开关产出确定性 GLSL。测试断言与它共享同一套规则。"""
    r, g, b = PALETTE_RGB[palette_key(palette)]
    time_term = "iTime * 1.5" if dynamic else "0.0"
    return _VALID_TEMPLATE.format(r=r, g=g, b=b, time_term=time_term)


# --------------------------------------------------------------------
# 故障模拟
# --------------------------------------------------------------------

class StubLLMTimeout(TimeoutError):
    """模拟上游超时（会被 classify_upstream_error 归类为 LLM_TIMEOUT）。"""


class StubLLMRateLimited(RuntimeError):
    """模拟 429 限流。"""


def _maybe_fail() -> None:
    """按当前故障配置决定是否抛异常 / 拖慢。"""
    cfg = faults.current()
    if cfg.llm_latency_ms:
        time.sleep(cfg.llm_latency_ms / 1000.0)
    if not faults.should_fail("llm", "llm_mode", "llm_fail_times"):
        return
    mode = cfg.llm_mode
    if mode == "timeout":
        raise StubLLMTimeout("stub llm: request timed out after 120s")
    if mode == "rate_limit":
        raise StubLLMRateLimited("stub llm: 429 Too Many Requests (rate limit exceeded)")
    if mode == "auth_error":
        raise RuntimeError("stub llm: 401 invalid_api_key")
    if mode == "slow":
        time.sleep(0.3)


def _degraded_payload(kind: str) -> str | None:
    """非异常型故障：返回"内容有问题"的响应，考验解析容错。"""
    cfg = faults.current()
    if cfg.llm_mode == "ok":
        return None
    if not faults.should_fail(f"llm_payload_{kind}", "llm_mode", "llm_fail_times"):
        return None
    if cfg.llm_mode == "malformed_json":
        return '{"algorithm_summary": "缺右括号'
    if cfg.llm_mode == "empty":
        return ""
    if cfg.llm_mode == "invalid_code" and kind == "code":
        return "// EXPLAIN: 故意产出维度不匹配的代码，用于驱动修复循环。\n" + _BROKEN_TEMPLATE
    if cfg.llm_mode == "unsupported" and kind == "code":
        return "// EXPLAIN: 故意引用 iChannel0，用于验证规则校验拦截。\n" + _UNSUPPORTED_TEMPLATE
    return None


# --------------------------------------------------------------------
# 五个 llm_fn 的桩实现
# --------------------------------------------------------------------

class StubLLM:
    """持有调用统计的桩工厂。一个实例对应一次装配。"""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.last_prompt: dict[str, str] = {}

    def _record(self, kind: str, prompt: str) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1
        self.last_prompt[kind] = prompt

    # ---- code_fn：生成 / 修复 / 改写 ----
    def code_fn(self, messages: list[dict[str, str]]) -> str:
        prompt = _join(messages)
        self._record("code", prompt)
        _maybe_fail()
        degraded = _degraded_payload("code")
        if degraded is not None:
            return degraded

        is_fix = "修正模式" in prompt or "需要修复的上一轮代码" in prompt
        is_rewrite = "ShaderRemixer" in prompt

        palette = _spec_field(prompt, "palette")
        dynamic = _spec_field(prompt, "dynamic", "True").lower() != "false"

        if is_rewrite:
            # 改写：保留原代码主干，仅替换基色，便于测试断言"最小改动"
            base_code = _extract_base_code(prompt)
            # 改写分支的提示词里没有 `- description:` 行，指令在"改写指令："后面
            m = re.search(r"改写指令：(.*)", prompt)
            ask = (m.group(1).strip() if m else "") or _spec_field(prompt, "description")
            new_palette = palette or ask
            if base_code:
                code = _swap_palette(base_code, new_palette)
            else:
                code = build_shader(new_palette, dynamic)
            return (
                "// EXPLAIN: 保留原有的 uv 归一化与距离场主干，仅替换基色向量以满足改写指令。\n"
                "// 未改动函数划分与动画项，便于与原始代码逐行对比。\n" + code
            )

        code = build_shader(palette, dynamic)
        head = (
            "// EXPLAIN: 修复上一轮的类型不匹配问题，保留原有算法结构。\n"
            if is_fix else
            "// EXPLAIN: 采用极坐标距离场与正弦波纹实现同心圆动画。\n"
            "// 基色向量直接体现指定调色板，动画由 iTime 驱动。\n"
        )
        return head + code

    # ---- json_fn：结构化分析（walkthrough / summary）----
    def json_fn(self, messages: list[dict[str, str]]) -> str:
        prompt = _join(messages)
        self._record("json", prompt)
        _maybe_fail()
        degraded = _degraded_payload("json")
        if degraded is not None:
            return degraded
        return _analysis_json()

    # ---- chat_fn：自由文本（视觉推断 / 对照）----
    def chat_fn(self, messages: list[dict[str, str]]) -> str:
        prompt = _join(messages)
        self._record("chat", prompt)
        _maybe_fail()
        degraded = _degraded_payload("chat")
        if degraded is not None:
            return degraded
        # 有些 Action（如 ExplainShaderAction）走 chat_fn 但要求 JSON 输出，
        # 按提示词里是否声明了 JSON Schema 来决定返回形态。
        if "JSON" in prompt or "json" in prompt:
            return _analysis_json()
        return (
            "画面表现为从中心向外扩散的同心圆波纹，主色调稳定，"
            "亮度随距离周期变化，整体呈现连续的呼吸感动画。"
        )

    # ---- vision_fn / text_critique_fn：自评 ----
    def vision_critique_fn(self, code: str, spec_text: str, image_b64: str) -> str:
        self._record("vision_critique", spec_text)
        _maybe_fail()
        degraded = _degraded_payload("critique")
        if degraded is not None:
            return degraded
        return json.dumps(
            {"score": 0.82, "rationale": "渲染结果与需求描述的主色调与动态一致。",
             "suggested_diff": ""},
            ensure_ascii=False,
        )

    def text_critique_fn(self, code: str, spec_text: str, compile_info: str) -> str:
        self._record("text_critique", spec_text)
        _maybe_fail()
        degraded = _degraded_payload("critique")
        if degraded is not None:
            return degraded
        ok = "编译通过" in (compile_info or "")
        return json.dumps(
            {"score": 0.78 if ok else 0.30,
             "rationale": "代码结构与 spec 吻合。" if ok else "存在编译错误，需先修复。",
             "suggested_diff": ""},
            ensure_ascii=False,
        )



def _analysis_json() -> str:
    """分析类 Action 的统一 JSON 响应。

    取各 Action 所需键的并集：每个 Action 只解析自己关心的字段，
    因此一份 payload 就能覆盖 walkthrough / summary / effect / compare 全部动作。
    """
    payload: dict[str, Any] = {
        "walkthrough": {
            "mainImage": "把片元坐标归一化到以屏幕中心为原点的坐标系，再按距离生成波纹。",
            "color": "用基色向量乘以波纹强度得到最终颜色，保证主色调稳定。",
        },
        "key_variables": {
            "uv": "归一化后的屏幕坐标，y 轴对齐保证不同分辨率下比例一致。",
            "d": "到中心的距离，作为距离场输入。",
            "base": "基色向量，决定整体色调。",
        },
        "algorithm_summary": (
            "该 shader 以屏幕中心为原点做极坐标变换，用 sin 对距离取波纹，"
            "再乘以基色得到同心圆动画，属于典型的 2D 距离场图案。"
        ),
        "techniques": ["2d-pattern"],
        "visual_effect": "屏幕中心向外扩散的同心圆波纹，随时间连续运动。",
        "summary": "同心圆波纹动画。",
    }
    return json.dumps(payload, ensure_ascii=False)


def _extract_base_code(prompt: str) -> str:
    """从改写提示词里取出原始代码块（```glsl ... ``` 或裸代码段）。"""
    m = re.search(r"```(?:glsl)?\s*\n(.*?)```", prompt, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(void\s+mainImage\s*\(.*)", prompt, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def _swap_palette(code: str, palette: str) -> str:
    """把代码里第一处 vec3 常量替换成目标调色板基色；找不到则追加一行注释。

    刻意做成"最小改动"，让改写用例可以断言：原有函数与结构行仍然存在。
    """
    r, g, b = PALETTE_RGB[palette_key(palette)]
    new_vec = f"vec3({r:.3f}, {g:.3f}, {b:.3f})"
    swapped, n = re.subn(
        r"vec3\s*\(\s*[-+0-9.]+\s*,\s*[-+0-9.]+\s*,\s*[-+0-9.]+\s*\)",
        new_vec, code, count=1,
    )
    if n == 0:
        swapped = code.replace(
            "fragColor = vec4(", f"    // palette -> {new_vec}\n    fragColor = vec4(", 1,
        )
    return swapped


def make_stub_llm_fns() -> tuple[Callable, Callable, Callable, Callable, Callable, StubLLM]:
    """返回 (chat_fn, json_fn, code_fn, vision_fn, text_critique_fn, stub)。"""
    stub = StubLLM()
    return (
        stub.chat_fn, stub.json_fn, stub.code_fn,
        stub.vision_critique_fn, stub.text_critique_fn, stub,
    )
