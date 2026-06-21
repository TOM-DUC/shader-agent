"""headless GLSL 渲染与编译模块。

对外暴露：
  - wrap_shadertoy_fragment(code)            : 把 Shadertoy fragment 包成完整 GLSL 330
  - GLSLCompiler.try_create() / .compile()   : 静态编译验证（只走 GL 编译，不渲染）
  - GLSLRenderer.try_create() / .render()    : 编译 + 全屏 quad + 读回 PNG
  - MockCompiler / MockRenderer              : 单测用桩
"""
from shader_agent.rendering.shadertoy_wrap import wrap_shadertoy_fragment  # noqa: F401
from shader_agent.rendering.compiler import GLSLCompiler  # noqa: F401
from shader_agent.rendering.renderer import GLSLRenderer  # noqa: F401
from shader_agent.rendering.mock import MockCompiler, MockRenderer  # noqa: F401
