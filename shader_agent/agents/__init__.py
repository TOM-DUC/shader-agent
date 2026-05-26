"""Agent 子模块（阶段三）。

包含：
  - schemas    : 角色间共享的数据契约（Message / AnalysisReport / GenerationSpec ...）
  - memory     : 角色私有的记忆模块
  - actions    : Action 抽象基类与具体 Action 实现
  - role       : Role 抽象基类
  - analyzer   : ShaderAnalyzer 角色
  - generator  : ShaderGenerator 角色
  - orchestrator: 两个角色的串行调度器
"""
from shader_agent.agents.role import Role  # noqa: F401
from shader_agent.agents.schemas import (  # noqa: F401
    AnalysisReport,
    GenerationSpec,
    GeneratedShader,
    Message,
    SimilarShader,
)
