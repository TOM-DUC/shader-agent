"""最小冒烟测试。需要真实 DEEPSEEK_API_KEY 才能跑，CI 上可加 marker 跳过。"""
import os
import pytest

# CI 的 requirements-ci.txt 刻意不含 openai（test profile 用确定性桩）。
# 缺依赖时跳过整个模块而不是 collection 阶段 ImportError——
# 否则 `pytest tests -m config` 这种只筛用例的收集也会被它拖垮。
pytest.importorskip("openai")

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
