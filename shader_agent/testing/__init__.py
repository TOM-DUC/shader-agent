"""可测试性支撑包（随应用发布，仅在 test profile 下被装配）。

包含三类"替身"：
  - stub_llm     : 确定性大模型桩 + 超时/限流/脏数据故障模式
  - fake_render  : 确定性编译器/渲染器桩（产出可断言的真实图像）
  - fake_retriever: 内存语料检索桩（确定性排序，可断言检索相关性）

以及 `faults` 故障开关。业务代码不 import 本包（装配层按 profile 决定），
因此生产链路零侵入。
"""
from shader_agent.testing import faults  # noqa: F401

__all__ = ["faults"]
