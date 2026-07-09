"""DeepEval 评审模型（LLM-as-a-judge）适配器。

为什么不用 deepeval 内置的 `DeepSeekModel`：
  1. 本项目已有一套带**重试、缓存、超时、统计**的 DeepSeek 客户端（llm/），
     评审走同一条链路，能复用缓存（同一 golden 重复评估不重复付费），
     也能被 Langfuse 一并记录为 generation；
  2. 内置实现会另起一个 OpenAI 客户端，绕过上述能力，且要求单独的环境变量。

因此这里继承 `DeepEvalBaseLLM`，把 deepeval 的评审请求转接到 `deepseek.chat`。

结构化输出：新版 deepeval 会给 `generate()` 传 `schema`（一个 pydantic 模型），
要求返回该模型的实例。我们通过 JSON 模式 + 校验 + 一次「修复重试」来满足；
失败时抛出，由 deepeval 自行降级/报错。

若 deepeval 未安装，本模块的 import 不会炸：`build_judge_model()` 返回 None，
上层据此跳过评估（与项目一贯的"可降级"风格一致）。
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from shader_agent.config.settings import settings
from shader_agent.utils.logger import logger


def _deepeval_available() -> bool:
    try:
        import deepeval  # noqa: F401
        return True
    except Exception:
        return False


# deepeval 未安装时，用一个占位基类，保证本模块可被安全 import
if _deepeval_available():
    from deepeval.models import DeepEvalBaseLLM as _BaseLLM
else:  # pragma: no cover
    class _BaseLLM:  # type: ignore[no-redef]
        pass


_JSON_INSTRUCTION = (
    "\n\n你必须只输出一个合法的 JSON 对象，不要使用 markdown 代码围栏，"
    "不要输出任何解释性文字。"
)


def _extract_json(text: str) -> str:
    """从 LLM 回复里抠出第一个 JSON 对象（容忍围栏与前后缀噪声）。"""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    # 退化：取第一个 { 到最后一个 } 之间
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        return s[i: j + 1]
    return s


class DeepSeekJudge(_BaseLLM):
    """把 deepeval 的评审调用转接到项目自带的 DeepSeek 客户端。

    用法：
        judge = DeepSeekJudge()
        metric = GEval(name="...", model=judge, ...)
    """

    def __init__(
        self,
        model_name: str = "",
        temperature: float | None = None,
    ) -> None:
        self._model = model_name or settings.evaluation.judge_model or settings.llm.chat_model
        self._temperature = (
            settings.evaluation.judge_temperature if temperature is None else temperature
        )
        try:
            super().__init__(model_name=self._model)
        except Exception:
            # 老版本 DeepEvalBaseLLM 的 __init__ 签名可能不同
            pass

    # ---------- DeepEvalBaseLLM 契约 ----------

    def load_model(self) -> Any:
        """deepeval 要求实现；本适配器无需加载权重，返回客户端单例即可。"""
        from shader_agent.llm.deepseek_client import deepseek
        return deepseek

    def get_model_name(self) -> str:
        return f"deepseek-judge[{self._model}]"

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
        """同步评审调用。schema 非空时返回该 pydantic 模型的实例。"""
        from shader_agent.llm.llm_fn import _do_call  # 复用缓存 + 统计 + 重试

        json_mode = schema is not None
        content = prompt + (_JSON_INSTRUCTION if json_mode else "")
        messages = [{"role": "user", "content": content}]

        text = _do_call(
            messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=settings.llm.max_tokens,
            json_mode=json_mode,
            use_cache=True,
        )
        if schema is None:
            return text

        # 结构化：解析 → 失败则让模型自我修复一次
        raw = _extract_json(text)
        try:
            return schema(**json.loads(raw))
        except Exception as e:
            logger.warning(f"[judge] schema 解析失败，尝试修复一次: {e}")
            fix_messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": text[:2000]},
                {"role": "user", "content":
                    f"上面的输出不是合法 JSON 或不符合要求（错误：{e}）。"
                    f"请只重新输出修正后的 JSON 对象。"},
            ]
            text2 = _do_call(
                fix_messages,
                model=self._model,
                temperature=0.0,
                max_tokens=settings.llm.max_tokens,
                json_mode=True,
                use_cache=False,
            )
            return schema(**json.loads(_extract_json(text2)))

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
        """异步契约：DeepSeek 客户端是同步的，这里丢到线程池避免阻塞事件循环。"""
        import anyio
        return await anyio.to_thread.run_sync(
            lambda: self.generate(prompt, schema, **kwargs)
        )


def build_judge_model() -> Optional[DeepSeekJudge]:
    """构造评审模型。deepeval 未安装或缺 API key 时返回 None。"""
    if not _deepeval_available():
        logger.warning("[judge] deepeval 未安装，跳过 LLM-as-a-judge 指标")
        return None
    if not settings.deepseek_api_key or settings.deepseek_api_key.startswith("sk-your"):
        logger.warning("[judge] DEEPSEEK_API_KEY 未配置，跳过 LLM-as-a-judge 指标")
        return None
    return DeepSeekJudge()
