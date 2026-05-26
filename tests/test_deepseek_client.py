"""最小冒烟测试。需要真实 DEEPSEEK_API_KEY 才能跑，CI 上可加 marker 跳过。"""
import os
import pytest

from shader_agent.llm.deepseek_client import deepseek


@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="未配置 DEEPSEEK_API_KEY，跳过真实调用测试。",
)
def test_chat_smoke():
    resp = deepseek.chat(
        [{"role": "user", "content": "Reply with just the word pong."}],
        max_tokens=100,
    )
    assert isinstance(resp, str)
    assert len(resp.strip()) > 0
