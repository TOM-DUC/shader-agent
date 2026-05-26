"""阶段六：渲染验证闭环单测。

策略：
  - 默认全部用 MockCompiler / MockRenderer，离线可跑；
  - 真 GL 测试（test_real_*）用 pytest.skipif 在 moderngl 不可用时跳过。
"""
from __future__ import annotations

import importlib.util

import pytest

from shader_agent.agents.generator import ShaderGenerator
from shader_agent.agents.schemas import GeneratedShader, Message
from shader_agent.rendering.mock import MockCompiler, MockRenderer
from shader_agent.rendering.shadertoy_wrap import (
    map_line_number,
    wrap_shadertoy_fragment,
)


# ================================================================
# wrap_shadertoy_fragment / map_line_number
# ================================================================

def test_wrap_includes_user_code_between_markers():
    user = "void mainImage(out vec4 c, in vec2 p){ c=vec4(1.); }"
    wrapped = wrap_shadertoy_fragment(user)
    assert "#version 330" in wrapped
    assert "USER CODE BEGIN" in wrapped
    assert "USER CODE END" in wrapped
    assert "uniform vec3  iResolution" in wrapped
    assert user in wrapped
    # epilogue 必须调用 mainImage
    assert "mainImage(_col" in wrapped


def test_map_line_number_translates_user_region():
    """假设 prologue 占 N 行，错误的 user:M 应当是 N+M。"""
    user = "void mainImage(out vec4 c, in vec2 p){\n  bad_token;\n  c=vec4(1.);\n}"
    wrapped = wrap_shadertoy_fragment(user)
    prologue_lines = wrapped.split("USER CODE BEGIN")[0].count("\n") + 1
    # 模拟驱动 1：NVIDIA 格式
    err1 = f"0({prologue_lines + 1}) : error C1503"
    out1 = map_line_number(err1, user)
    assert "user:" in out1
    # 模拟驱动 2：Mesa 格式
    err2 = f"0:{prologue_lines + 2}: 'foo' : undeclared"
    out2 = map_line_number(err2, user)
    assert "user:" in out2


def test_map_line_number_marks_prologue_region():
    """超出用户区的行号应标 prologue/epilogue。"""
    err = "0(2) : something"  # prologue 区
    out = map_line_number(err, "void mainImage(out vec4 c, in vec2 p){}")
    assert "prologue" in out


# ================================================================
# MockCompiler
# ================================================================

def test_mock_compiler_detects_missing_main():
    cr = MockCompiler().compile("vec4 unrelated(){ return vec4(1.0); }")
    assert cr.ok is False
    assert "mainImage" in cr.errors


def test_mock_compiler_accepts_valid_code():
    cr = MockCompiler().compile(
        "void mainImage(out vec4 c, in vec2 p){ c=vec4(1.0); }"
    )
    assert cr.ok is True


def test_mock_compiler_detects_dim_mismatch():
    cr = MockCompiler().compile(
        "void mainImage(out vec4 c, in vec2 p){"
        " vec3 v = vec4(1.0);"  # vec3 = vec4：维度不匹配
        " c=vec4(v,1.0); }"
    )
    assert cr.ok is False
    assert "vec3" in cr.errors and "vec4" in cr.errors


def test_mock_compiler_force_error():
    """force_error 用于让"修正循环"测试可控。"""
    cr = MockCompiler(force_error="injected error msg").compile("anything")
    assert cr.ok is False
    assert "injected error msg" in cr.errors


# ================================================================
# MockRenderer
# ================================================================

def test_mock_renderer_returns_valid_png_for_valid_code():
    r = MockRenderer()
    png = r.render("void mainImage(out vec4 c, in vec2 p){ c=vec4(1.0); }")
    assert isinstance(png, bytes)
    assert png.startswith(b"\x89PNG")
    assert r.render_calls == 1


def test_mock_renderer_raises_on_invalid_code():
    r = MockRenderer()
    with pytest.raises(RuntimeError):
        r.render("vec4 unrelated(){ return vec4(1.0); }")


# ================================================================
# Generator 接入 Mock compiler：修正循环吃真编译错误
# ================================================================

def test_generator_uses_mock_compiler_and_fixes_error():
    """
    场景：第 1 轮 LLM 产出"看上去合法"但被真编译器拒绝的代码（vec3=vec4），
    Generator 应进入修正轮，第 2 轮 LLM 产出无维度问题的代码。
    """
    calls = {"n": 0}
    def stub_llm(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                "// EXPLAIN: bad\n"
                "void mainImage(out vec4 c, in vec2 p){"
                " vec3 v = vec4(1.0); c = vec4(v, 1.0); }"
            )
        return (
            "// EXPLAIN: fixed\n"
            "void mainImage(out vec4 c, in vec2 p){ c = vec4(1.0); }"
        )

    gen = ShaderGenerator(
        llm_fn=stub_llm,
        compiler=MockCompiler(),     # 阶段六注入点
        max_fix_loops=2,
    )
    out = gen.handle(Message(role="user", content="anything"))
    g = GeneratedShader(**out.payload)
    assert g.compile_result.ok is True
    assert g.iterations == 2
    assert calls["n"] == 2


def test_generator_with_real_render_and_critique_pipeline():
    """全套：compiler + renderer + critique_fn 都注入；走文本 critique_fn。"""
    def stub_llm(_):
        return (
            "// EXPLAIN: ok\n"
            "void mainImage(out vec4 c, in vec2 p){ c=vec4(p,iTime,1.); }"
        )

    captured = {}
    def stub_critique(code, spec_text, img_b64):
        captured["code"] = code
        captured["spec_text"] = spec_text
        captured["img"] = img_b64
        return '{"score": 0.92, "rationale": "looks great", "suggested_diff": ""}'

    gen = ShaderGenerator(
        llm_fn=stub_llm,
        compiler=MockCompiler(),
        renderer=MockRenderer(),
        critique_fn=stub_critique,
        enable_self_critique=True,
        max_fix_loops=1,
    )
    out = gen.handle(Message(role="user", content="x"))
    g = GeneratedShader(**out.payload)
    assert g.compile_result.ok is True
    assert g.self_critique_score == 0.92
    assert "looks great" in g.self_critique_rationale
    # critique_fn 应该真的拿到了 base64 图像（MockRenderer 出 PNG，被 base64 编码）
    assert captured["img"] != ""
    assert len(captured["img"]) > 20


def test_generator_without_renderer_falls_back_to_text_critique():
    """启用 enable_self_critique=True 但没注入 renderer/critique_fn，
    应该走 SelfCritiqueAction 的文本层弱自评，不报错。"""
    def stub_llm(_):
        return ("// EXPLAIN: ok\n"
                "void mainImage(out vec4 c, in vec2 p){ c=vec4(iTime); }")
    gen = ShaderGenerator(
        llm_fn=stub_llm,
        compiler=MockCompiler(),
        enable_self_critique=True,
        max_fix_loops=0,
    )
    out = gen.handle(Message(role="user", content="dynamic noise"))
    g = GeneratedShader(**out.payload)
    assert g.self_critique_rationale != ""
    # 文本层自评不会用 vision_critique 那种 score
    # 但 spec.dynamic=True 且代码含 iTime → 至少加一分


# ================================================================
# 真 GL 测试（条件跳过）
# ================================================================

_HAS_MODERNGL = importlib.util.find_spec("moderngl") is not None


def _real_gl_available() -> bool:
    if not _HAS_MODERNGL:
        return False
    from shader_agent.rendering import GLSLCompiler
    c, _err = GLSLCompiler.try_create()
    return c is not None


@pytest.mark.skipif(not _HAS_MODERNGL or not _real_gl_available(),
                    reason="moderngl 或 GL context 不可用")
def test_real_glsl_compiler_accepts_valid_seed():
    """跑真 GL 编译器：seed03 Raymarched Sphere 必须编译通过。"""
    from shader_agent.corpus.seed_shaders import get_seed_shaders
    from shader_agent.rendering import GLSLCompiler
    c, _ = GLSLCompiler.try_create()
    sphere = next(s for s in get_seed_shaders() if s.shader_id == "seed03")
    cr = c.compile(sphere.code_image)
    assert cr.ok is True, f"unexpected compile error: {cr.errors}"


@pytest.mark.skipif(not _HAS_MODERNGL or not _real_gl_available(),
                    reason="moderngl 或 GL context 不可用")
def test_real_glsl_compiler_rejects_bad_syntax():
    from shader_agent.rendering import GLSLCompiler
    c, _ = GLSLCompiler.try_create()
    cr = c.compile("void mainImage(out vec4 c, in vec2 p){ broken syntax }")
    assert cr.ok is False
    # 真编译错误应含行号
    assert ("user:" in cr.errors) or ("error" in cr.errors.lower())


@pytest.mark.skipif(not _HAS_MODERNGL or not _real_gl_available(),
                    reason="moderngl 或 GL context 不可用")
def test_real_glsl_renderer_produces_png():
    from shader_agent.rendering import GLSLRenderer
    r, _ = GLSLRenderer.try_create()
    code = "void mainImage(out vec4 c, in vec2 p){ c = vec4(p / iResolution.xy, 0.5, 1.0); }"
    png = r.render(code, width=128, height=96, time=1.0)
    assert isinstance(png, bytes)
    assert png.startswith(b"\x89PNG")
    # 128*96 即使最强压缩也不至于 0
    assert len(png) > 100
