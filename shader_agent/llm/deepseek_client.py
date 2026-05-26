"""DeepSeek 客户端封装。

四种验证模式：
  1. chat        —— 普通对话（一次性返回）
  2. coder       —— 代码生成（语义上等同 chat，使用 coder_model + 更低 temperature）
  3. stream      —— 流式输出
  4. function    —— Function Calling / Tool Use

阶段三以后，agent 通过本模块统一访问 LLM。
"""
from __future__ import annotations

from typing import Any, Generator, Iterable

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from shader_agent.config.settings import settings
from shader_agent.utils.logger import logger


class DeepSeekClient:
    """对 OpenAI SDK 的薄封装，专门指向 DeepSeek。"""

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=settings.llm.timeout_seconds,
        )
        self.cfg = settings.llm

    # ---------- 内部：重试装饰器 ----------
    def _retry(self, fn):
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential(
                multiplier=self.cfg.retry_backoff_seconds, min=1, max=20
            ),
            retry=retry_if_exception_type(Exception),
        )(fn)

    # ---------- 模式 1：chat ----------
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """普通对话，返回文本。"""

        @self._retry
        def _call() -> ChatCompletion:
            return self._client.chat.completions.create(
                model=model or self.cfg.chat_model,
                messages=messages,
                temperature=temperature if temperature is not None else self.cfg.temperature,
                max_tokens=max_tokens or self.cfg.max_tokens,
                top_p=self.cfg.top_p,
                **kwargs,
            )

        resp = _call()
        content = resp.choices[0].message.content or ""
        logger.debug(
            f"[chat] model={resp.model} tokens={resp.usage.total_tokens if resp.usage else '?'}"
        )
        return content

    # ---------- 模式 2：coder ----------
    def code(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> str:
        """代码生成：使用 coder_model，温度更低，stop 不变。"""
        return self.chat(
            messages,
            model=self.cfg.coder_model,
            temperature=temperature,
            **kwargs,
        )

    # ---------- 模式 3：stream ----------
    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """流式输出，逐 token yield。"""
        stream: Iterable[ChatCompletionChunk] = self._client.chat.completions.create(
            model=model or self.cfg.chat_model,
            messages=messages,
            temperature=temperature if temperature is not None else self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            top_p=self.cfg.top_p,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ---------- 模式 4：function calling ----------
    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict = "auto",
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Function Calling 入口，返回完整的 ChatCompletion，上层自行解析 tool_calls。

        tools 的格式遵循 OpenAI 规范：
            [
              {
                "type": "function",
                "function": {
                  "name": "...",
                  "description": "...",
                  "parameters": { "type": "object", "properties": {...}, "required": [...] }
                }
              }
            ]
        """

        @self._retry
        def _call() -> ChatCompletion:
            return self._client.chat.completions.create(
                model=model or self.cfg.chat_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                **kwargs,
            )

        return _call()


# 模块级单例
deepseek = DeepSeekClient()
