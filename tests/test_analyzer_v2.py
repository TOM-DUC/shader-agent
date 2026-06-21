"""单测：四段式 Analyzer 的 Actions。

约束：
  - 不调真 LLM，全部用 stub；
  - 不依赖向量库；
  - 不依赖网络。
"""
from __future__ import annotations

import json

from shader_agent.agents.actions.analyzer_actions import (
    ParseShaderAction, ParseShaderIn,
)
from shader_agent.agents.actions.analyzer_actions_v2 import (
    CompareAction, CompareIn,
    EffectInferAction, EffectInferIn,
    SummaryAction, SummaryIn,
    TECHNIQUE_VOCAB,
    WalkthroughAction, WalkthroughIn,
    split_code_into_sections,
)
from shader_agent.agents.analyzer import ShaderAnalyzer
from shader_agent.agents.schemas import (
    AnalysisReport,
    Message,
    SimilarShader,
)
from shader_agent.corpus.seed_shaders import get_seed_shaders


# =====================================================================
# split_code_into_sections
# =====================================================================

def test_split_extracts_each_function_and_globals():
    code = """#define PI 3.14159
uniform float foo;

float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

vec3 calcNormal(vec3 p) {
    return normalize(p);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(1.0);
}
"""
    parse = ParseShaderAction().run(ParseShaderIn(code=code)).data
    sections = split_code_into_sections(code, parse)
    assert "sdSphere" in sections
    assert "calcNormal" in sections
    assert "mainImage" in sections
    assert "globals" in sections
    assert "#define PI" in sections["globals"]
    # 每段应包含开 { 和闭 }
    assert sections["sdSphere"].startswith("float sdSphere")
    assert sections["sdSphere"].rstrip().endswith("}")


def test_split_handles_no_functions():
    sections = split_code_into_sections("// nothing here\n",
                                        ParseShaderAction().run(ParseShaderIn(code="")).data)
    # 只有 globals
    assert list(sections.keys()) == ["globals"] or sections == {}


# =====================================================================
# WalkthroughAction
# =====================================================================

def test_walkthrough_with_stub_llm_returns_dict():
    def stub(messages):
        return json.dumps({
            "walkthrough": {
                "mainImage": "标准化坐标，计算渐变颜色，作为最终输出。",
                "globals": "声明 uniform 与常量。",
            },
            "key_variables": {
                "uv": "标准化屏幕坐标",
                "iTime": "时间秒",
                "fragColor": "输出颜色",
            },
        })
    code = "void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(1.);}"
    parse = ParseShaderAction().run(ParseShaderIn(code=code)).data
    r = WalkthroughAction(llm_fn=stub).run(WalkthroughIn(code=code, parse_result=parse))
    assert r.ok
    assert "mainImage" in r.data.walkthrough
    assert r.data.key_variables["uv"]


def test_walkthrough_handles_markdown_fence():
    def stub(_):
        return ("```json\n"
                '{"walkthrough":{"a":"x"},"key_variables":{"v":"y"}}\n```')
    code = "void mainImage(out vec4 c, in vec2 p){c=vec4(1.);}"
    parse = ParseShaderAction().run(ParseShaderIn(code=code)).data
    r = WalkthroughAction(llm_fn=stub).run(WalkthroughIn(code=code, parse_result=parse))
    assert r.ok
    assert r.data.walkthrough == {"a": "x"}


def test_walkthrough_falls_back_on_bad_json():
    def stub(_):
        return "this is not json at all"
    code = "void mainImage(out vec4 c, in vec2 p){c=vec4(1.);}"
    parse = ParseShaderAction().run(ParseShaderIn(code=code)).data
    r = WalkthroughAction(llm_fn=stub).run(WalkthroughIn(code=code, parse_result=parse))
    assert r.ok  # Action 仍成功，只是走 fallback
    # fallback 会写一个 _note 段
    assert "_note" in r.data.walkthrough


def test_walkthrough_no_llm_uses_fallback():
    code = "void mainImage(out vec4 c, in vec2 p){c=vec4(1.);}"
    parse = ParseShaderAction().run(ParseShaderIn(code=code)).data
    r = WalkthroughAction(llm_fn=None).run(WalkthroughIn(code=code, parse_result=parse))
    assert r.ok
    assert "占位" in next(iter(r.data.walkthrough.values()))


# =====================================================================
# SummaryAction
# =====================================================================

def test_summary_strips_out_of_vocab_techniques():
    def stub(_):
        return json.dumps({
            "algorithm_summary": "x" * 100,
            "techniques": ["raymarching", "totally_made_up", "sdf"],
        })
    code = "void mainImage(out vec4 c, in vec2 p){c=vec4(1.);}"
    parse = ParseShaderAction().run(ParseShaderIn(code=code)).data
    r = SummaryAction(llm_fn=stub).run(SummaryIn(code=code, parse_result=parse))
    assert r.ok
    assert set(r.data.techniques) == {"raymarching", "sdf"}
    assert "totally_made_up" not in r.data.techniques


def test_summary_no_llm_fallback_uses_static_heuristics():
    code = """
float sdSphere(vec3 p){return 0.0;}
void mainImage(out vec4 c, in vec2 p){
    float t = 0.0;
    for(int i=0;i<32;i++){ vec3 q = vec3(t); t += sdSphere(q); }
    c = vec4(t);
}
"""
    parse = ParseShaderAction().run(ParseShaderIn(code=code)).data
    r = SummaryAction(llm_fn=None).run(SummaryIn(code=code, parse_result=parse))
    assert r.ok
    assert "sdf" in r.data.techniques


# =====================================================================
# EffectInferAction
# =====================================================================

def test_effect_infer_strips_fences():
    def stub(_):
        return "```\n蓝色球体在中心，随时间脉动。\n```"
    code = "void mainImage(out vec4 c, in vec2 p){c=vec4(1.);}"
    parse = ParseShaderAction().run(ParseShaderIn(code=code)).data
    r = EffectInferAction(llm_fn=stub).run(
        EffectInferIn(code=code, parse_result=parse, summary="x"),
    )
    assert r.ok
    assert r.data.visual_effect == "蓝色球体在中心，随时间脉动。"


def test_effect_infer_no_llm_placeholder():
    code = "void mainImage(out vec4 c, in vec2 p){c=vec4(1.);}"
    parse = ParseShaderAction().run(ParseShaderIn(code=code)).data
    r = EffectInferAction(llm_fn=None).run(
        EffectInferIn(code=code, parse_result=parse, summary=""),
    )
    assert r.ok
    assert "占位" in r.data.visual_effect


# =====================================================================
# CompareAction
# =====================================================================

def test_compare_no_similar_returns_early():
    r = CompareAction(llm_fn=lambda _: "should not be called").run(
        CompareIn(code="void mainImage(out vec4 c, in vec2 p){}", similar=[])
    )
    assert r.ok
    assert "无相似样本" in r.data.comparison


def test_compare_with_llm():
    def stub(_):
        return "本代码与参考样本均采用 raymarching，但材质参数更暖。"
    sim = [SimilarShader(shader_id="s1", name="Ref Sphere", distance=0.1,
                         tags_topic=["raymarching", "sdf"],
                         code_excerpt="void mainImage(){}")]
    r = CompareAction(llm_fn=stub).run(
        CompareIn(code="void mainImage(out vec4 c, in vec2 p){}",
                 summary="raymarching a sphere",
                 similar=sim)
    )
    assert r.ok
    assert "raymarching" in r.data.comparison.lower()


# =====================================================================
# Analyzer Role with fourstage strategy
# =====================================================================

def _make_stub_llm():
    """模拟四段 LLM 的统一桩。匹配关键词必须互不冲突，按更具体优先。"""
    def fn(messages):
        sys_content = messages[0]["content"]
        # 顺序敏感：summary 必须先匹配（含"算法摘要"），否则会被 walkthrough 抢到
        if "算法摘要" in sys_content and "techniques" in sys_content:
            return json.dumps({
                "algorithm_summary": "这段 shader 使用 raymarching 与 sphere SDF "
                                    "在屏幕空间步进光线，对命中点用四点差分估法向，"
                                    "再以兰伯特模型计算颜色。" + "x" * 60,
                "techniques": ["raymarching", "sdf", "lighting"],
            })
        if "对照分析" in sys_content:
            return "本代码与参考 1 同样使用 raymarching + SDF，但本代码光照更冷。"
        if "视觉效果" in sys_content:
            return "一个蓝色球体悬浮在黑色背景中，光照来自右上方。"
        if "逐段讲解" in sys_content:
            return json.dumps({
                "walkthrough": {
                    "mainImage": "在 NDC 空间发射光线，遍历最多 64 步求 SDF 最近距离。",
                    "calcNormal": "通过四点差分估算法向量，用于光照。",
                    "sdSphere": "返回点到球心的距离减半径。",
                },
                "key_variables": {
                    "ro": "ray origin",
                    "rd": "ray direction",
                    "t": "marching parameter",
                },
            })
        return ""
    return fn


def test_analyzer_fourstage_end_to_end():
    seeds = get_seed_shaders()
    sphere = next(s for s in seeds if s.name == "Raymarched Sphere")
    stub = _make_stub_llm()
    analyzer = ShaderAnalyzer(
        vector_store=None,
        walkthrough_llm=stub,
        summary_llm=stub,
        effect_llm=stub,
        compare_llm=stub,
        llm_fn=stub,
        strategy="fourstage",
    )
    in_msg = Message(role="user", content=sphere.code_image,
                     payload={"code": sphere.code_image})
    out = analyzer.handle(in_msg)
    rep = AnalysisReport(**out.payload)

    # 四段都生效
    assert len(rep.algorithm_summary) >= 80
    assert "raymarching" in rep.techniques and "sdf" in rep.techniques
    assert "calcNormal" in rep.section_walkthrough or \
           "mainImage" in rep.section_walkthrough
    assert "ro" in rep.key_variables
    assert "球体" in rep.visual_effect or "球" in rep.visual_effect


def test_analyzer_strategy_single_still_works():
    """single 策略不能被破坏。"""
    code = "void mainImage(out vec4 c, in vec2 p){c=vec4(1.);}"
    analyzer = ShaderAnalyzer(strategy="single", llm_fn=None)
    out = analyzer.handle(Message(role="user", content=code, payload={"code": code}))
    rep = AnalysisReport(**out.payload)
    assert rep.techniques  # fallback 也会给标签


def test_markdown_render_has_8_sections():
    rep = AnalysisReport(
        source_code="void mainImage(out vec4 c, in vec2 p){c=vec4(1.);}",
        algorithm_summary="abc",
        techniques=["sdf"],
        key_variables={"x": "y"},
        visual_effect="蓝色",
        section_walkthrough={
            "mainImage": "讲解 mainImage",
            "对照参考样本": "比较内容",
        },
        similar_shaders=[SimilarShader(shader_id="s1", name="A", distance=0.1)],
        model_used="deepseek-chat",
    )
    md = rep.to_markdown()
    # 关键段都在
    for marker in ["视觉效果", "算法摘要", "分段讲解", "关键变量",
                   "相似样本", "对照参考样本", "源码", "deepseek-chat"]:
        assert marker in md, f"missing: {marker}"
    # comparison 不应再在 walkthrough 段
    assert "## 分段讲解" in md
    # 对照应该在自己的章节
    assert md.index("## 对照参考样本") > md.index("## 分段讲解")
