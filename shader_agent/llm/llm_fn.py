"""LLM 适配器：把 DeepSeekClient 包装成符合 `llm_fn(messages) -> str` 签名的函数。

为什么不直接传 deepseek.chat 给 Action？
1. Action 的 llm_fn 签名只要 (messages) -> str；DeepSeekClient.chat 有更多可选参数。
2. 阶段四需要：JSON 模式开关 + 重试 + 调试缓存 + token 统计聚合，
   这些是 LLM 调用层关心的事，不应污染 Action 代码。

提供两个工厂：
- make_chat_fn(...)  : 通用 chat（用于 walkthrough / summary / compare）
- make_json_fn(...)  : 强制 JSON 输出（用于 explain / parse_spec_llm 等需结构化的场景）
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

from shader_agent.config.settings import settings
from shader_agent.llm.deepseek_client import deepseek
from shader_agent.utils.logger import logger


# llm_fn 签名：messages -> 文本回复
LLMFn = Callable[[list[dict[str, str]]], str]


class _CallStats:
    """轻量级调用统计，进程级单例。"""
    def __init__(self) -> None:
        self.calls = 0
        self.cached = 0
        self.total_chars_in = 0
        self.total_chars_out = 0
        self.total_seconds = 0.0

    def reset(self) -> None:
        self.__init__()

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "cached": self.cached,
            "total_chars_in": self.total_chars_in,
            "total_chars_out": self.total_chars_out,
            "total_seconds": round(self.total_seconds, 3),
        }


stats = _CallStats()


# ---------------- 缓存层 ----------------

def _cache_dir() -> Path:
    d = settings.project_root / "data" / "cache" / "llm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(messages: list[dict[str, str]], model: str, temperature: float,
               json_mode: bool) -> str:
    h = hashlib.sha1()
    h.update(f"{model}|{temperature}|{json_mode}|".encode())
    for m in messages:
        h.update(m.get("role", "").encode())
        h.update(b"\x00")
        h.update((m.get("content", "") or "").encode("utf-8", errors="ignore"))
        h.update(b"\x01")
    return h.hexdigest()


def _cache_lookup(key: str) -> str | None:
    p = _cache_dir() / f"{key}.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj.get("content")
    except Exception:
        return None


def _cache_store(key: str, content: str, meta: dict) -> None:
    p = _cache_dir() / f"{key}.json"
    try:
        p.write_text(
            json.dumps({"content": content, "meta": meta}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[llm_fn] cache write failed: {e}")


def cache_enabled() -> bool:
    return os.environ.get("SHADER_AGENT_LLM_CACHE", "1") != "0"


# ---------------- 工厂 ----------------

def make_code_fn(
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    use_cache: bool | None = None,
) -> LLMFn:
    """代码生成路径：默认走 coder_model（settings.llm.coder_model），
    温度低、max_tokens 大。

    Generator 的 DraftCodeAction 使用此工厂。"""
    _model = model or settings.llm.coder_model
    _temp = 0.1 if temperature is None else temperature
    _maxt = max_tokens or max(settings.llm.max_tokens, 4096)
    _cache = cache_enabled() if use_cache is None else use_cache

    def _fn(messages: list[dict[str, str]]) -> str:
        return _do_call(messages, model=_model, temperature=_temp,
                        max_tokens=_maxt, json_mode=False, use_cache=_cache)

    _fn.__name__ = f"code_fn[{_model}]"
    return _fn


def make_chat_fn(
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    use_cache: bool | None = None,
) -> LLMFn:
    """普通 chat。"""
    _model = model or settings.llm.chat_model
    _temp = settings.llm.temperature if temperature is None else temperature
    _maxt = max_tokens or settings.llm.max_tokens
    _cache = cache_enabled() if use_cache is None else use_cache

    def _fn(messages: list[dict[str, str]]) -> str:
        return _do_call(messages, model=_model, temperature=_temp,
                        max_tokens=_maxt, json_mode=False, use_cache=_cache)

    _fn.__name__ = f"chat_fn[{_model}]"
    return _fn


def make_json_fn(
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    use_cache: bool | None = None,
) -> LLMFn:
    """强制 JSON 输出。DeepSeek 的 OpenAI 兼容接口支持 response_format。"""
    _model = model or settings.llm.chat_model
    _temp = 0.0 if temperature is None else temperature
    _maxt = max_tokens or settings.llm.max_tokens
    _cache = cache_enabled() if use_cache is None else use_cache

    def _fn(messages: list[dict[str, str]]) -> str:
        return _do_call(messages, model=_model, temperature=_temp,
                        max_tokens=_maxt, json_mode=True, use_cache=_cache)

    _fn.__name__ = f"json_fn[{_model}]"
    return _fn


# ---------------- 实际调用 ----------------

def _do_call(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    use_cache: bool,
) -> str:
    in_chars = sum(len(m.get("content", "")) for m in messages)
    key = _cache_key(messages, model, temperature, json_mode)

    if use_cache:
        cached = _cache_lookup(key)
        if cached is not None:
            stats.calls += 1
            stats.cached += 1
            stats.total_chars_in += in_chars
            stats.total_chars_out += len(cached)
            logger.debug(f"[llm_fn] cache hit {key[:8]} ({len(cached)} chars)")
            return cached

    t0 = time.perf_counter()
    extra: dict = {}
    if json_mode:
        # DeepSeek 兼容 OpenAI 的 response_format
        extra["response_format"] = {"type": "json_object"}
    try:
        text = deepseek.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
    except TypeError:
        # 若底层 client 不接受 response_format，退化为不带 JSON 模式
        if "response_format" in extra:
            text = deepseek.chat(messages, model=model,
                                temperature=temperature, max_tokens=max_tokens)
        else:
            raise
    elapsed = time.perf_counter() - t0

    stats.calls += 1
    stats.total_chars_in += in_chars
    stats.total_chars_out += len(text or "")
    stats.total_seconds += elapsed
    logger.info(
        f"[llm_fn] call model={model} json={json_mode} "
        f"in={in_chars}ch out={len(text or '')}ch took={elapsed:.2f}s"
    )

    if use_cache:
        _cache_store(key, text, meta={
            "model": model, "temperature": temperature,
            "json_mode": json_mode, "elapsed": elapsed,
        })
    return text


# ---------------- 阶段六：多模态自评 ----------------

# 多模态 critique 的签名：(code, spec_text, image_b64) -> str(JSON)
# 与 SelfCritiqueAction 的 critique_fn 钩子签名对齐。
VisionCritiqueFn = "Callable[[str, str, str], str]"


_CRITIQUE_SYSTEM = (
    "你是 ShaderCritic。下面给出一段 GLSL fragment shader、它对应的需求 spec，"
    "以及它渲染出来的一帧截图（PNG）。\n"
    "请评估这帧图像是否符合 spec 期望，输出**严格 JSON**：\n"
    "{\n"
    '  "score": <0.0~1.0，越高越符合>,\n'
    '  "rationale": "<60~150 字中文评语，先说看到了什么，再说与 spec 的吻合/偏差>",\n'
    '  "suggested_diff": "<可选；若 score<0.6，给一句修改建议；否则留空字符串>"\n'
    "}\n"
    "评估维度：颜色/调色板、构图、动态感、是否符合 effect_type、是否有明显的渲染瑕疵。"
    "不要 markdown 包裹，不要多余文字。"
)


def make_vision_critique_fn(
    *,
    model: str | None = None,
    use_cache: bool | None = None,
    text_only_fallback: bool = True,
):
    """构造多模态自评函数。

    返回的函数签名：(code, spec_text, image_b64) -> str(JSON)

    工作原理：
      - 优先尝试 OpenAI-兼容 vision messages 格式（content 是 list[{type:'text'},{type:'image_url'}]）
      - 如果底层模型不支持（DeepSeek v4-pro 当前文本-only），自动 fallback 到纯文本版：
        把图片描述为 "(图像数据已省略)"，让 LLM 仅基于代码与 spec 评估。
        这种降级仍能给出有意义的 score。

    用法：
        from shader_agent.llm.llm_fn import make_vision_critique_fn
        fn = make_vision_critique_fn()
        json_str = fn(code, "raymarching neon blue", base64_png)
    """
    _model = model or settings.llm.chat_model
    _cache = cache_enabled() if use_cache is None else use_cache

    def _fn(code: str, spec_text: str, image_b64: str) -> str:
        # 不缓存 image_b64 全量到 key 里（太大），用 hash 摘要做 key
        import hashlib
        img_digest = hashlib.sha1(image_b64.encode("utf-8")).hexdigest()[:16] if image_b64 else "no_img"
        cache_msgs = [
            {"role": "system", "content": _CRITIQUE_SYSTEM},
            {"role": "user", "content": f"[code={hashlib.sha1(code.encode()).hexdigest()[:8]}"
                                         f"|spec={spec_text}|img={img_digest}]"},
        ]
        if _cache:
            key = _cache_key(cache_msgs, _model, 0.0, False)
            cached = _cache_lookup(key)
            if cached is not None:
                stats.calls += 1; stats.cached += 1
                return cached

        # 优先尝试多模态
        text_only_user = (
            f"需求 spec: {spec_text}\n\n"
            f"代码:\n```glsl\n{code[:4000]}\n```\n\n"
            f"(图像数据已省略，请仅根据代码与 spec 评估)"
        )
        text_only_msgs = [
            {"role": "system", "content": _CRITIQUE_SYSTEM},
            {"role": "user", "content": text_only_user},
        ]

        result_text = ""
        if image_b64:
            vision_msgs = [
                {"role": "system", "content": _CRITIQUE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text":
                            f"需求 spec: {spec_text}\n\n"
                            f"代码:\n```glsl\n{code[:3500]}\n```"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }},
                    ],
                },
            ]
            try:
                result_text = deepseek.chat(
                    vision_msgs,  # type: ignore[arg-type]
                    model=_model,
                    temperature=0.0,
                    max_tokens=512,
                )
            except Exception as e:
                logger.info(f"[critique] vision call failed, falling back: {e}")
                if not text_only_fallback:
                    raise
                result_text = ""

        if not result_text:
            # text-only 模式
            result_text = deepseek.chat(
                text_only_msgs, model=_model, temperature=0.0, max_tokens=512,
            )

        if _cache:
            _cache_store(key, result_text, meta={"model": _model, "vision": bool(image_b64)})
        stats.calls += 1
        return result_text

    _fn.__name__ = f"vision_critique[{_model}]"
    return _fn


# ---------------- 纯文本自评（无多模态也能用） ----------------

_TEXT_CRITIQUE_SYSTEM = (
    "你是 ShaderCritic。下面给出一段 GLSL fragment shader、它对应的需求 spec，"
    "以及它的**编译结果**（没有渲染截图）。\n"
    "请仅基于代码、spec 和编译结果做评估，输出**严格 JSON**，不要 markdown 包裹：\n"
    "{\n"
    '  "score": <0.0~1.0，越高越好；若编译失败请给 <=0.4>,\n'
    '  "rationale": "<80~160 字中文评语：先说代码是否实现了 spec 要点'
    '（effect_type/palette/dynamic），再说代码质量；若编译失败，明确指出错误'
    '原因与最可能的修复方向>",\n'
    '  "suggested_diff": "<可选；若有明显问题给一句具体修改建议，否则留空字符串>"\n'
    "}\n"
    "评估维度：是否实现 effect_type、是否体现 palette 调性、dynamic 是否用了 iTime、"
    "结构是否清晰、是否能编译通过。务必客观，不要夸大。"
)


def make_text_critique_fn(
    *,
    model: str | None = None,
    use_cache: bool | None = None,
):
    """构造**纯文本**自评函数（无需多模态模型）。

    返回的函数签名：(code, spec_text, compile_info) -> str(JSON)

    与 SelfCritiqueAction 的 text_critique_fn 钩子对齐。即便没有渲染截图，
    也能让 LLM 基于代码 + spec + 编译结果给出评分与编译错误分析。
    """
    _model = model or settings.llm.chat_model
    _cache = cache_enabled() if use_cache is None else use_cache

    def _fn(code: str, spec_text: str, compile_info: str) -> str:
        user = (
            f"需求 spec: {spec_text}\n\n"
            f"编译结果: {compile_info}\n\n"
            f"代码:\n```glsl\n{code[:4000]}\n```\n\n"
            f"请输出自评 JSON。"
        )
        messages = [
            {"role": "system", "content": _TEXT_CRITIQUE_SYSTEM},
            {"role": "user", "content": user},
        ]
        return _do_call(
            messages, model=_model, temperature=0.0,
            max_tokens=512, json_mode=True, use_cache=_cache,
        )

    _fn.__name__ = f"text_critique[{_model}]"
    return _fn
