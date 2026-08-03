"""分析接口：结构化产物 + 检索一致性。

分析结果是给人看的长文本，很容易写成"只要不报错就算过"。这里把可判定的部分
挑出来做强断言：报告必须结构化可消费（下游 Generator 直接吃这个对象）、
技术标签必须落在受控词表内、参考样本必须与检索接口口径一致。
"""
from __future__ import annotations

import pytest

from tests_api.utils.assertions import assert_error, assert_ok, assert_schema

pytestmark = [pytest.mark.api]

# 与 corpus.tagger 的受控词表保持一致；越界即视为模型幻觉出了新标签
TOPIC_VOCAB = {
    "raymarching", "sdf", "noise", "fractal", "2d-pattern", "voronoi",
    "post-processing", "lighting", "animation", "geometry", "color",
}


@pytest.mark.smoke
def test_analyze_returns_structured_report(shader_api, valid_shader):
    data = assert_ok(shader_api.analyze(valid_shader, top_k=2))
    report = data["report"]
    assert report["source_code"].strip() == valid_shader.strip(), (
        "报告必须回带原始代码，供前后对照与回溯")
    assert report["algorithm_summary"], "算法摘要不应为空"
    assert isinstance(report["key_variables"], dict)
    assert data["report_md"].startswith("#"), "markdown 视图应有标题结构"


@pytest.mark.contract
def test_analyze_response_matches_schema(shader_api, valid_shader):
    result = shader_api.analyze(valid_shader, top_k=1)
    assert_ok(result)
    assert_schema(result.raw, "analyze_response.json")


def test_analyze_techniques_stay_in_controlled_vocabulary(shader_api, valid_shader):
    """技术标签必须来自受控词表，否则下游按标签检索会全部落空。"""
    data = assert_ok(shader_api.analyze(valid_shader))
    unknown = set(data["techniques"]) - TOPIC_VOCAB
    assert not unknown, f"出现词表外的技术标签：{unknown}"


def test_analyze_top_k_controls_reference_count(shader_api, valid_shader):
    """top_k 必须真的生效——参数写了却不透传是很常见的静默缺陷。"""
    one = assert_ok(shader_api.analyze(valid_shader, top_k=1))
    three = assert_ok(shader_api.analyze(valid_shader, top_k=3))
    assert one["n_references"] <= 1
    assert three["n_references"] <= 3
    assert three["n_references"] >= one["n_references"]


def test_analyze_references_are_consistent_with_retrieval(
        shader_api, retrieval_api, valid_shader):
    """分析里的参考样本，应当能被检索接口以同样口径复现。

    跨接口一致性是纯单接口测试覆盖不到的：两条链路各自"看起来正常"，
    但用的是两份不同的语料或不同的阈值，最终表现为"讲解里引用的样本，
    用户在检索页怎么也搜不到"。
    """
    data = assert_ok(shader_api.analyze(valid_shader, top_k=3))
    if data["n_references"] == 0:
        pytest.skip("本次分析未命中参考样本")
    ref_ids = {r["shader_id"] for r in data["references"]}
    search = assert_ok(retrieval_api.search("同心圆 波纹 距离场 图案", top_k=6))
    search_ids = {i["shader_id"] for i in search["items"]}
    assert ref_ids & search_ids, (
        f"分析引用了 {ref_ids}，但检索接口返回 {search_ids}，两侧语料口径不一致")


def test_analyze_with_render_returns_preview(shader_api, valid_shader):
    data = assert_ok(shader_api.analyze(valid_shader, with_render=True))
    assert data["render"]["ok"] is True
    assert data["render"]["image_base64"]


def test_analyze_render_failure_does_not_break_analysis(
        shader_api, unsupported_shader):
    """附带渲染失败时，分析主流程仍应成功——这是明确的降级约定。"""
    data = assert_ok(shader_api.analyze(unsupported_shader, with_render=True))
    assert data["report"]["source_code"]
    assert data["render"]["ok"] is False
    assert data["render"]["code"] == 40004


@pytest.mark.parametrize("bad,expect_code", [
    ({"code": ""}, 40001),
    ({}, 40001),
    ({"code": "void mainImage(out vec4 c, in vec2 p){}", "top_k": 0}, 40001),
    ({"code": "void mainImage(out vec4 c, in vec2 p){}", "top_k": 11}, 40001),
])
def test_analyze_invalid_params(shader_api, bad, expect_code):
    assert_error(shader_api.analyze(payload=bad), expect_code, http=422)
