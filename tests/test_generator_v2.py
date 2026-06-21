"""单测：Generator（修正 prompt 分支 / validate / 自评 / Role 编排）。

不调真 LLM，全部用 stub。
"""
from __future__ import annotations

from shader_agent.agents.actions.generator_actions import (
    DraftCodeAction, DraftCodeIn,
    SelfCritiqueAction, SelfCritiqueIn,
    ValidateCodeAction, ValidateCodeIn,
)
from shader_agent.agents.generator import ShaderGenerator
from shader_agent.agents.schemas import (
    AnalysisReport, GeneratedShader, GenerationSpec, Message,
)


# ===========================================================
# DraftCodeAction prompt 分支
# ===========================================================

def test_draft_first_round_uses_creation_prompt():
    captured = {}
    def stub(messages):
        captured["sys"] = messages[0]["content"]
        captured["user"] = messages[1]["content"]
        return "// EXPLAIN: x\nvoid mainImage(out vec4 c, in vec2 p){ c=vec4(1.); }"
    r = DraftCodeAction(llm_fn=stub).run(DraftCodeIn(
        spec=GenerationSpec(description="raymarch sphere", effect_type="raymarching"),
    ))
    assert r.ok
    # 首轮 system prompt 含 "硬性约束"，不含 "修正模式"
    assert "硬性约束" in captured["sys"]
    assert "修正模式" not in captured["sys"]
    # user 段没有上一轮代码
    assert "需要修复的上一轮代码" not in captured["user"]


def test_draft_fix_round_uses_fix_prompt():
    captured = {}
    def stub(messages):
        captured["sys"] = messages[0]["content"]
        captured["user"] = messages[1]["content"]
        return "// EXPLAIN: fixed\nvoid mainImage(out vec4 c, in vec2 p){ c=vec4(1.); }"
    r = DraftCodeAction(llm_fn=stub).run(DraftCodeIn(
        spec=GenerationSpec(description="x"),
        prev_code="void other(){}",
        prev_errors="missing mainImage entry function",
    ))
    assert r.ok
    # 修正轮 system prompt 应含 "修正模式"
    assert "修正模式" in captured["sys"]
    # user 段以 "需要修复的上一轮代码" 开头
    assert captured["user"].startswith("需要修复的上一轮代码")
    # 上一轮代码应被嵌入
    assert "void other()" in captured["user"]
    assert "missing mainImage" in captured["user"]


# ===========================================================
# ValidateCodeAction 强化
# ===========================================================

def test_validate_detects_hlsl_misuse():
    code = (
        "void mainImage(out vec4 c, in vec2 p){\n"
        "  float x = lerp(0.0, 1.0, 0.5);\n"  # HLSL 误用
        "  c = vec4(saturate(x));\n"
        "}"
    )
    r = ValidateCodeAction().run(ValidateCodeIn(code=code))
    assert r.ok  # action 本身成功
    res = r.data.result
    assert res.ok is False
    assert "lerp" in res.errors and "mix" in res.errors
    assert "saturate" in res.errors


def test_validate_detects_signature_mismatch():
    code = "void mainImage(vec4 c, vec2 p){ c=vec4(1.); }"  # 缺 out/in
    r = ValidateCodeAction().run(ValidateCodeIn(code=code))
    assert r.data.result.ok is False
    assert "signature" in r.data.result.errors.lower()


def test_validate_comment_brace_does_not_break():
    """注释里的 { } 不应被算入配对。"""
    code = (
        "// this comment has { unmatched\n"
        "/* and { inside block too */\n"
        "void mainImage(out vec4 c, in vec2 p){ c=vec4(1.); }"
    )
    r = ValidateCodeAction().run(ValidateCodeIn(code=code))
    assert r.data.result.ok is True


def test_validate_balanced_braces_with_proper_signature():
    code = (
        "float sd(vec3 p){ return length(p)-1.0; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
        "  vec2 uv = fragCoord / iResolution.xy;\n"
        "  fragColor = vec4(uv, 0.5, 1.0);\n"
        "}"
    )
    r = ValidateCodeAction().run(ValidateCodeIn(code=code))
    assert r.data.result.ok is True


def test_validate_brace_count_report_in_error():
    code = "void mainImage(out vec4 c, in vec2 p){ if(true){ c=vec4(1.); }"  # 少一个 }
    r = ValidateCodeAction().run(ValidateCodeIn(code=code))
    assert r.data.result.ok is False
    # 错误信息应包含 count，帮 LLM 修正
    assert "count=" in r.data.result.errors


# ===========================================================
# SelfCritiqueAction（占位 / 文本层）
# ===========================================================

def test_self_critique_text_only_when_no_image_and_no_fn():
    code = "void mainImage(out vec4 c, in vec2 p){ c=vec4(iTime); }"
    spec = GenerationSpec(description="x", effect_type="raymarching", dynamic=True)
    r = SelfCritiqueAction().run(SelfCritiqueIn(code=code, spec=spec))
    assert r.ok
    # 应该说明含 iTime（动态被满足）
    assert "iTime" in r.data.rationale or "时间" in r.data.rationale
    # 未明确 raymarching 特征 → score 不为满
    assert 0.0 <= r.data.score <= 1.0


def test_self_critique_with_image_calls_critique_fn():
    captured = {}
    def critique(code, spec_text, img_b64):
        captured["code"] = code; captured["spec_text"] = spec_text
        captured["img"] = img_b64
        return '{"score": 0.85, "rationale": "looks good", "suggested_diff": ""}'
    r = SelfCritiqueAction(critique_fn=critique).run(SelfCritiqueIn(
        code="void mainImage(out vec4 c, in vec2 p){ c=vec4(1.); }",
        spec=GenerationSpec(description="x"),
        rendered_image_b64="FAKE_B64",
    ))
    assert r.ok
    assert r.data.score == 0.85
    assert "looks good" in r.data.rationale
    assert captured["img"] == "FAKE_B64"


def test_self_critique_handles_bad_json_from_critique_fn():
    def critique(*a, **k):
        return "not json"
    r = SelfCritiqueAction(critique_fn=critique).run(SelfCritiqueIn(
        code="void mainImage(out vec4 c, in vec2 p){ c=vec4(1.); }",
        spec=GenerationSpec(description="x"),
        rendered_image_b64="FAKE",
    ))
    assert r.ok
    assert r.data.score == 0.0
    assert "parse failed" in r.data.rationale


def test_self_critique_action_is_non_critical():
    """SelfCritiqueAction.critical 应为 False，失败不阻断主流程。"""
    assert SelfCritiqueAction().critical is False


# ===========================================================
# Generator Role 端到端（含自评开关）
# ===========================================================

def test_generator_runs_self_critique_when_enabled():
    """启用 self_critique 时，最终 GeneratedShader.self_critique_* 字段应被填充。"""
    def stub_llm(messages):
        return "// EXPLAIN: ok\nvoid mainImage(out vec4 c, in vec2 fragCoord){ c=vec4(iTime); }"
    gen = ShaderGenerator(
        llm_fn=stub_llm, enable_self_critique=True, max_fix_loops=0,
    )
    out = gen.handle(Message(role="user", content="raymarch dynamic"))
    g = GeneratedShader(**out.payload)
    assert g.compile_result.ok is True
    # 启用了自评（无 critique_fn 也走文本层）
    assert g.self_critique_rationale != ""
    assert 0.0 <= g.self_critique_score <= 1.0


def test_generator_skips_self_critique_when_disabled():
    def stub_llm(_):
        return "// EXPLAIN: ok\nvoid mainImage(out vec4 c, in vec2 p){ c=vec4(1.); }"
    gen = ShaderGenerator(
        llm_fn=stub_llm, enable_self_critique=False, max_fix_loops=0,
    )
    out = gen.handle(Message(role="user", content="anything"))
    g = GeneratedShader(**out.payload)
    assert g.self_critique_rationale == ""
    assert g.self_critique_score == 0.0


def test_generator_fix_loop_with_strict_validator():
    """前 2 轮 LLM 出 HLSL 误用，validate 必报错；第 3 轮才正确。"""
    calls = {"n": 0}
    def stub_llm(messages):
        calls["n"] += 1
        if calls["n"] <= 2:
            return (
                "// EXPLAIN: bad\n"
                "void mainImage(out vec4 c, in vec2 p){"
                " c=vec4(saturate(p.x)); }"  # saturate 是 HLSL
            )
        return (
            "// EXPLAIN: good\n"
            "void mainImage(out vec4 c, in vec2 p){"
            " c=vec4(clamp(p.x,0.0,1.0)); }"
        )
    gen = ShaderGenerator(llm_fn=stub_llm, max_fix_loops=3)
    out = gen.handle(Message(role="user", content="anything"))
    g = GeneratedShader(**out.payload)
    assert g.compile_result.ok is True
    assert g.iterations == 3


def test_generator_with_reference_report_carries_into_prompt():
    """analyze_then_generate 风格：spec.reference_report 应被 DraftCodeAction 看见。"""
    captured = {}
    def stub_llm(messages):
        captured["user"] = messages[1]["content"]
        return "// EXPLAIN: x\nvoid mainImage(out vec4 c, in vec2 p){ c=vec4(1.); }"

    rep = AnalysisReport(
        source_code="x",
        algorithm_summary="raymarching a sphere with phong lighting",
        techniques=["raymarching", "sdf", "lighting"],
        key_variables={"ro": "ray origin", "rd": "ray direction"},
    )
    spec = GenerationSpec(description="改成霓虹紫", reference_report=rep)
    msg = spec.to_message()

    gen = ShaderGenerator(llm_fn=stub_llm, max_fix_loops=0)
    gen.handle(msg)

    # 参考的算法摘要应出现在 prompt
    assert "raymarching a sphere" in captured["user"]
    # 关键变量名应出现
    assert "ro" in captured["user"]


# ===========================================================
# Markdown 渲染（含自评显示）
# ===========================================================

def test_generated_shader_markdown_includes_self_critique():
    g = GeneratedShader(
        code="void mainImage(out vec4 c, in vec2 p){ c=vec4(1.); }",
        explanation="hello",
        iterations=2,
        self_critique_score=0.75,
        self_critique_rationale="✓ 含 iTime",
        model_used="deepseek-v4-flash",
    )
    md = g.to_markdown()
    assert "自评" in md
    assert "0.75" in md
    assert "deepseek-v4-flash" in md
    assert "iterations" in md
