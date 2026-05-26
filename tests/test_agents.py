"""阶段三离线单测：覆盖 schemas / memory / actions / role / orchestrator。

约束：
  - 不调真 LLM；
  - 不依赖向量库（vector_store=None 走无检索分支）。
"""
from __future__ import annotations

import json

from shader_agent.agents.actions.analyzer_actions import (
    ExplainShaderAction,
    ExplainShaderIn,
    ParseShaderAction,
    ParseShaderIn,
    RetrieveSimilarAction,
    RetrieveSimilarIn,
    SynthesizeReportAction,
    SynthesizeReportIn,
)
from shader_agent.agents.actions.base import Action, ActionResult
from shader_agent.agents.actions.generator_actions import (
    DraftCodeAction,
    DraftCodeIn,
    ParseSpecAction,
    ParseSpecIn,
    ValidateCodeAction,
    ValidateCodeIn,
)
from shader_agent.agents.analyzer import ShaderAnalyzer
from shader_agent.agents.generator import ShaderGenerator
from shader_agent.agents.memory import Memory
from shader_agent.agents.orchestrator import Orchestrator
from shader_agent.agents.schemas import (
    AnalysisReport,
    GeneratedShader,
    GenerationSpec,
    Message,
    SimilarShader,
)
from shader_agent.corpus.seed_shaders import get_seed_shaders


# =====================================================================
# schemas
# =====================================================================

def test_message_short_truncation():
    m = Message(role="user", content="a" * 200)
    s = m.short(limit=50)
    assert s.startswith("[user] ")
    assert "…" in s


def test_analysis_report_to_message_roundtrip():
    rep = AnalysisReport(
        source_code="void mainImage(){}",
        algorithm_summary="abc",
        techniques=["raymarching", "sdf"],
        key_variables={"t": "ray parameter"},
    )
    msg = rep.to_message()
    assert msg.payload_type == "AnalysisReport"
    rep2 = AnalysisReport(**msg.payload)
    assert rep2.algorithm_summary == "abc"
    assert rep2.techniques == ["raymarching", "sdf"]


def test_analysis_report_markdown_nonempty():
    rep = AnalysisReport(source_code="x", algorithm_summary="hi",
                        techniques=["noise"], visual_effect="blue")
    md = rep.to_markdown()
    assert "noise" in md and "hi" in md and "blue" in md


def test_generation_spec_to_message():
    s = GenerationSpec(description="画个圆")
    m = s.to_message()
    assert m.payload_type == "GenerationSpec"
    s2 = GenerationSpec(**m.payload)
    assert s2.description == "画个圆"


# =====================================================================
# memory
# =====================================================================

def test_memory_basic_ops():
    mem = Memory()
    a = Message(role="user", content="a")
    b = Message(role="analyzer", content="b", parent_id=a.msg_id)
    mem.add(a)
    mem.add(b)
    assert len(mem) == 2
    assert mem.find(a.msg_id) is not None and mem.find(a.msg_id).msg_id == a.msg_id
    assert mem.by_role("analyzer")[0].msg_id == b.msg_id
    assert mem.latest(1)[0].msg_id == b.msg_id
    chain = mem.lineage(b.msg_id)
    assert [m.msg_id for m in chain] == [a.msg_id, b.msg_id]


# =====================================================================
# Action base
# =====================================================================

class _DummyAction(Action):
    name = "dummy"
    input_schema = ParseShaderIn  # 复用
    output_schema = ParseSpecIn   # 任意
    def _run(self, inp):
        raise ValueError("boom")


def test_action_critical_failure_returns_result_not_raises():
    a = _DummyAction()
    r = a.run(ParseShaderIn(code=""))
    assert isinstance(r, ActionResult)
    assert r.ok is False
    assert "boom" in r.error


def test_action_accepts_dict_input():
    a = ParseShaderAction()
    r = a.run({"code": "void mainImage(){}"})  # 传 dict
    assert r.ok
    assert r.data.has_main_image is True


# =====================================================================
# Analyzer actions
# =====================================================================

def test_parse_shader_action_extracts_funcs():
    code = """
float sdSphere(vec3 p, float r){ return length(p)-r; }
vec3 calcNormal(vec3 p){ return p; }
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    fragColor = vec4(iTime*0.0, iResolution.xy, 1.0);
}
"""
    r = ParseShaderAction().run(ParseShaderIn(code=code))
    assert r.ok
    out = r.data
    assert out.has_main_image is True
    assert "sdSphere" in out.custom_functions
    assert "calcNormal" in out.custom_functions
    assert "mainImage" in out.custom_functions
    assert "iTime" in out.used_builtins
    assert "iResolution" in out.used_builtins


def test_retrieve_similar_no_vector_store_returns_empty():
    r = RetrieveSimilarAction().run(RetrieveSimilarIn(code="void mainImage(){}"))
    assert r.ok and r.data.items == []


def test_explain_action_with_stub_llm_returns_json():
    """注入 stub llm_fn 返回合法 JSON。"""
    def stub_llm(messages):
        return json.dumps({
            "algorithm_summary": "raymarching a sphere",
            "key_variables": {"t": "march distance", "ro": "ray origin"},
            "techniques": ["raymarching", "sdf"],
            "visual_effect": "a blue sphere",
            "section_walkthrough": {"main loop": "march until hit"},
        })
    a = ExplainShaderAction(llm_fn=stub_llm)
    parse_out = ParseShaderAction().run(ParseShaderIn(code="x")).data
    r = a.run(ExplainShaderIn(code="x", parse_result=parse_out, similar=[]))
    assert r.ok
    out = r.data
    assert "raymarching" in out.techniques
    assert out.key_variables["t"] == "march distance"


def test_explain_action_handles_dirty_llm_output():
    """LLM 把 JSON 包在 ```json ``` 围栏里也要能解析。"""
    def stub_llm(_):
        return ("```json\n" +
                '{"algorithm_summary":"x","key_variables":{},'
                '"techniques":["noise"],"visual_effect":"y",'
                '"section_walkthrough":{}}' +
                "\n```")
    a = ExplainShaderAction(llm_fn=stub_llm)
    parse_out = ParseShaderAction().run(ParseShaderIn(code="x")).data
    r = a.run(ExplainShaderIn(code="x", parse_result=parse_out, similar=[]))
    assert r.ok and r.data.techniques == ["noise"]


def test_explain_action_no_llm_falls_back():
    """无 llm_fn 时用 fallback。"""
    a = ExplainShaderAction(llm_fn=None)
    parse_out = ParseShaderAction().run(
        ParseShaderIn(code="float sdSphere(vec3 p){return 0.0;}\n"
                          "void mainImage(out vec4 f, in vec2 c){}")
    ).data
    r = a.run(ExplainShaderIn(
        code="float sdSphere(vec3 p){return 0.0;}\n"
             "void mainImage(out vec4 f, in vec2 c){}",
        parse_result=parse_out,
    ))
    assert r.ok
    assert "sdf" in r.data.techniques  # fallback 能识别 sdSphere


def test_synthesize_report_action():
    parse_out = ParseShaderAction().run(
        ParseShaderIn(code="void mainImage(out vec4 f, in vec2 c){}")
    ).data
    explain_r = ExplainShaderAction(llm_fn=None).run(
        ExplainShaderIn(code="void mainImage(out vec4 f, in vec2 c){}",
                       parse_result=parse_out)
    )
    r = SynthesizeReportAction().run(SynthesizeReportIn(
        code="void mainImage(out vec4 f, in vec2 c){}",
        explain=explain_r.data,
        similar=[SimilarShader(shader_id="s1", name="x", distance=0.1)],
    ))
    assert r.ok
    rep = r.data
    assert isinstance(rep, AnalysisReport)
    assert len(rep.similar_shaders) == 1


# =====================================================================
# Generator actions
# =====================================================================

def test_parse_spec_keywords():
    r = ParseSpecAction().run(ParseSpecIn(user_text="做一个简单的 raymarching 球，冷色调，不要用纹理，<= 80 行"))
    assert r.ok
    spec = r.data.spec
    assert spec.effect_type == "raymarching"
    assert spec.palette == "cool blue"
    assert spec.complexity == "simple"
    assert any("no external textures" in c for c in spec.constraints)
    assert any("<= 80 lines" in c for c in spec.constraints)


def test_parse_spec_inherit_palette_from_base():
    base = GenerationSpec(description="prev", palette="neon")
    r = ParseSpecAction().run(ParseSpecIn(user_text="加点动画", inherit_from=base))
    assert r.ok and r.data.spec.palette == "neon"


def test_draft_code_stub_when_no_llm():
    r = DraftCodeAction().run(DraftCodeIn(spec=GenerationSpec(description="x", effect_type="raymarching")))
    assert r.ok
    assert "mainImage" in r.data.code


def test_draft_code_with_stub_llm_strips_fences():
    def stub_llm(_):
        return ("```glsl\n"
                "// EXPLAIN: 一个最小 shader\n"
                "void mainImage(out vec4 c, in vec2 p){ c = vec4(1.0); }\n"
                "```")
    r = DraftCodeAction(llm_fn=stub_llm).run(
        DraftCodeIn(spec=GenerationSpec(description="x"))
    )
    assert r.ok
    assert r.data.explanation.startswith("一个最小 shader")
    assert "mainImage" in r.data.code
    assert "```" not in r.data.code


def test_validate_code_detects_problems():
    r = ValidateCodeAction().run(ValidateCodeIn(code="void other(){ {"))
    assert r.ok  # action 本身成功，只是 compile_result.ok = False
    assert r.data.result.ok is False
    errs = r.data.result.errors
    assert "missing mainImage" in errs
    assert "unbalanced" in errs


def test_validate_code_rejects_external_sampler():
    code = ("uniform sampler2D iChannel0;\n"
            "void mainImage(out vec4 c, in vec2 p){ c = texture(iChannel0, p); }")
    r = ValidateCodeAction().run(ValidateCodeIn(code=code))
    assert r.data.result.ok is False
    assert "sampler" in r.data.result.errors


def test_validate_code_passes_valid_minimal():
    code = "void mainImage(out vec4 c, in vec2 p){ c = vec4(1.0); }"
    r = ValidateCodeAction().run(ValidateCodeIn(code=code))
    assert r.data.result.ok is True


# =====================================================================
# Role + Orchestrator
# =====================================================================

def test_analyzer_role_handles_minimal_code():
    seeds = get_seed_shaders()
    sphere = next(s for s in seeds if s.name == "Raymarched Sphere")
    analyzer = ShaderAnalyzer(vector_store=None, llm_fn=None)
    in_msg = Message(role="user", content=sphere.code_image,
                    payload={"code": sphere.code_image})
    out = analyzer.handle(in_msg)
    assert out.payload_type == "AnalysisReport"
    rep = AnalysisReport(**out.payload)
    assert "sdf" in rep.techniques or "raymarching" in rep.techniques


def test_generator_role_no_llm_produces_runnable_stub():
    gen = ShaderGenerator(vector_store=None, llm_fn=None, max_fix_loops=0)
    out = gen.handle(Message(role="user", content="raymarching 球"))
    assert out.payload_type == "GeneratedShader"
    g = GeneratedShader(**out.payload)
    assert "mainImage" in g.code
    assert g.spec is not None and g.spec.effect_type == "raymarching"


def test_orchestrator_analyze_then_generate_carries_reference():
    seeds = get_seed_shaders()
    sphere = next(s for s in seeds if s.name == "Raymarched Sphere")
    orch = Orchestrator(
        analyzer=ShaderAnalyzer(vector_store=None, llm_fn=None),
        generator=ShaderGenerator(vector_store=None, llm_fn=None, max_fix_loops=0),
    )
    result = orch.analyze_then_generate(
        code=sphere.code_image,
        ask="保持算法不变，颜色换成霓虹紫",
    )
    rep = result["report"]
    gen = result["generated"]
    assert rep is not None and gen is not None
    # 关键契约：reference_report 必须穿越到 generator
    assert gen.spec is not None
    assert gen.spec.reference_report is not None
    assert gen.spec.reference_report.algorithm_summary == rep.algorithm_summary


def test_generator_fix_loop_runs_when_compile_fails():
    """模拟前两轮 LLM 返回坏代码（无 mainImage），第三轮才返回好代码。
    期望 iterations==3。"""
    calls = {"n": 0}
    def stub_llm(_):
        calls["n"] += 1
        if calls["n"] < 3:
            return "// EXPLAIN: bad\nfloat oops(){ return 1.0;\n"  # 无 mainImage + 括号不平
        return "// EXPLAIN: good\nvoid mainImage(out vec4 c, in vec2 p){ c=vec4(1.0); }"
    gen = ShaderGenerator(vector_store=None, llm_fn=stub_llm, max_fix_loops=3)
    out = gen.handle(Message(role="user", content="anything"))
    g = GeneratedShader(**out.payload)
    assert g.compile_result.ok is True
    assert g.iterations == 3
