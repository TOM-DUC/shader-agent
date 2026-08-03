"""DeepSeek 真实调用冒烟测试；仅在安装依赖且配置 API Key 时运行。"""
import os

import pytest


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="未配置 DEEPSEEK_API_KEY，跳过真实调用测试。",
)
def test_chat_smoke():
    # 必须放在测试函数内部，避免 pytest 收集阶段要求 openai。
    pytest.importorskip(
        "openai",
        reason="未安装 openai，跳过 DeepSeek 真实调用测试。",
    )

    from shader_agent.llm.deepseek_client import deepseek

    resp = deepseek.chat(
        [{"role": "user", "content": "Reply with just the word pong."}],
        max_tokens=100,
    )

    assert isinstance(resp, str)
    assert len(resp.strip()) > 0
