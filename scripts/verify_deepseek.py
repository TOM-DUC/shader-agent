"""阶段一验收脚本：依次跑通 chat / coder / stream / function calling。

用法（在项目根目录）：
    python -m scripts.verify_deepseek
"""
from __future__ import annotations

import json
import sys
from textwrap import shorten

from rich.console import Console
from rich.panel import Panel

from shader_agent.llm.deepseek_client import deepseek
from shader_agent.utils.logger import logger

console = Console()


def section(title: str) -> None:
    console.rule(f"[bold cyan]{title}[/bold cyan]")


# ---------------- 1. chat ----------------
def test_chat() -> bool:
    section("1/4  chat 模式")
    try:
        resp = deepseek.chat(
            [
                {"role": "system", "content": "You are a concise GLSL tutor."},
                {
                    "role": "user",
                    "content": "用一句中文解释 shadertoy 里 mainImage 函数的两个参数。",
                },
            ],
            max_tokens=256,
        )
        console.print(Panel(resp.strip(), title="chat 输出", border_style="green"))
        return True
    except Exception as e:
        logger.exception(f"chat 失败: {e}")
        return False


# ---------------- 2. coder ----------------
def test_coder() -> bool:
    section("2/4  coder 模式（代码生成）")
    try:
        resp = deepseek.code(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an expert GLSL programmer. "
                        "Return ONLY raw GLSL code, no markdown fences, no comments outside code."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Write a minimal Shadertoy fragment shader that draws "
                        "a horizontal gradient from black on the left to white on the right. "
                        "Use mainImage(out vec4 fragColor, in vec2 fragCoord)."
                    ),
                },
            ],
            max_tokens=512,
        )
        console.print(Panel(resp.strip(), title="coder 输出", border_style="magenta"))
        # 简单合理性校验
        ok = "mainImage" in resp and "fragColor" in resp
        if not ok:
            logger.warning("coder 输出未包含 mainImage / fragColor，请人工检查。")
        return ok
    except Exception as e:
        logger.exception(f"coder 失败: {e}")
        return False


# ---------------- 3. stream ----------------
def test_stream() -> bool:
    section("3/4  stream 模式（流式）")
    try:
        console.print("[dim]逐 token 输出 →[/dim] ", end="")
        total = []
        for piece in deepseek.chat_stream(
            [
                {"role": "system", "content": "你是简洁的助理。"},
                {"role": "user", "content": "用一句话说出 shader 是什么。"},
            ]
        ):
            sys.stdout.write(piece)
            sys.stdout.flush()
            total.append(piece)
        print()  # 换行
        full = "".join(total)
        return len(full.strip()) > 0
    except Exception as e:
        logger.exception(f"stream 失败: {e}")
        return False


# ---------------- 4. function calling ----------------
def test_function_calling() -> bool:
    section("4/4  function calling 模式")

    # 一个假工具，模拟"查 Shadertoy 某个 shader 的元信息"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_shadertoy",
                "description": "Look up metadata of a shader on Shadertoy by its id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shader_id": {
                            "type": "string",
                            "description": "The 6-character shadertoy id, e.g. 'XsXXDn'.",
                        }
                    },
                    "required": ["shader_id"],
                },
            },
        }
    ]

    try:
        resp = deepseek.chat_with_tools(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that uses tools when helpful.",
                },
                {
                    "role": "user",
                    "content": "帮我查一下 shadertoy 上 id 为 XsXXDn 的 shader 的元信息。",
                },
            ],
            tools=tools,
        )
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            console.print(
                Panel(
                    f"未触发工具调用，模型直接回答：\n{shorten(msg.content or '', 200)}",
                    title="function 输出",
                    border_style="yellow",
                )
            )
            return False
        for tc in tool_calls:
            console.print(
                Panel(
                    f"name: {tc.function.name}\nargs: {tc.function.arguments}",
                    title="function 调用",
                    border_style="green",
                )
            )
            # 校验参数是合法 JSON
            json.loads(tc.function.arguments)
        return True
    except Exception as e:
        logger.exception(f"function calling 失败: {e}")
        return False


def main() -> int:
    results = {
        "chat": test_chat(),
        "coder": test_coder(),
        "stream": test_stream(),
        "function": test_function_calling(),
    }
    section("验收结果汇总")
    for k, v in results.items():
        tag = "[green]PASS[/green]" if v else "[red]FAIL[/red]"
        console.print(f"  {k:<10} {tag}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
