"""端到端业务流。

单接口用例证明"每个零件是好的"，端到端用例证明"装起来能用"。这里串的是产品
真实的使用路径，并且每一步的产物都作为下一步的输入——只要中间任何一环的契约
发生漂移（字段改名、代码被包上围栏、渲染尺寸被内部改写），链路就会断在那一步。
"""
from __future__ import annotations

import pytest

from tests_api.utils import glsl_checker as gc
from tests_api.utils import image_checker as img
from tests_api.utils.assertions import assert_ok

pytestmark = [pytest.mark.e2e]


@pytest.mark.smoke
def test_generate_to_render_full_chain(shader_api):
    """需求 → 生成 → 静态校验 → 编译 → 渲染，五层依次验证。"""
    gen = assert_ok(shader_api.generate(
        "生成一个蓝色调的同心圆波纹动画", palette="霓虹蓝", dynamic=True))
    code = gen["code"]

    # L3 规则层
    gc.assert_shader_ok(code, dynamic=True)
    rule = assert_ok(shader_api.validate(code, require_dynamic=True))
    assert rule["passed"] is True, f"生成结果未通过规则校验：{rule['violations']}"

    # L4 编译层：用独立接口复核，而不是只信生成接口自报的 compile_ok
    compiled = assert_ok(shader_api.compile(code))
    assert compiled["ok"] is True, compiled["errors"]
    assert compiled["ok"] == gen["compile_ok"], "两个接口对同一段代码的编译结论不一致"

    # L5 图像层
    rendered = assert_ok(shader_api.render(code, width=256, height=192))
    stats = img.assert_image_ok(rendered["image_base64"], width=256, height=192)
    assert stats.dominant_channel == "b", f"蓝色需求未体现在画面上：{stats.mean}"


def test_analyze_then_remix_then_verify(shader_api, valid_shader):
    """读代码 → 理解 → 改写 → 复核：Shader Agent 的核心教学闭环。"""
    analysis = assert_ok(shader_api.analyze(valid_shader, top_k=2))
    assert analysis["report"]["algorithm_summary"]

    remixed = assert_ok(shader_api.remix(
        valid_shader, "保留原有算法，把主色调换成暖色橙红", with_render=True))
    new_code = remixed["code"]

    # 改写后的产物同样要过全部校验层
    gc.assert_shader_ok(new_code)
    assert assert_ok(shader_api.compile(new_code))["ok"] is True
    kept = gc.structural_similarity(valid_shader, new_code)
    assert kept >= 0.6, f"改写保留率仅 {kept:.0%}，退化成重写"

    before = img.load_stats(remixed["render_before"]["image_base64"])
    after = img.load_stats(remixed["render"]["image_base64"])
    assert before.dominant_channel != after.dominant_channel, "改写未改变画面主色"


def test_search_then_analyze_the_hit(shader_api, retrieval_api):
    """检索 → 拿到样本 → 再送去分析，验证跨模块的数据可衔接。"""
    hits = assert_ok(retrieval_api.search("raymarching sdf 球体", top_k=1))
    assert hits["total"] == 1
    top = hits["items"][0]
    assert "raymarching" in top["tags"]

    # 检索结果里应带有可供下游消费的算法摘要
    assert top["algorithm_summary"], "检索结果缺少算法摘要，下游 prompt 无法注入参考"


@pytest.mark.slow
def test_repeated_generation_is_stable(shader_api):
    """同需求连续生成 3 次，产物必须稳定合规。

    这条用例针对的是"偶发不合规"：抽一次能过，连着跑就有一次带上 iChannel 或
    忘了 iTime。稳定性问题只有重复执行才暴露得出来。
    """
    results = [
        assert_ok(shader_api.generate("生成一个绿色的噪声图案",
                                      palette="森林绿", dynamic=True))
        for _ in range(3)
    ]
    for i, data in enumerate(results):
        gc.assert_shader_ok(data["code"], dynamic=True)
        assert data["compile_ok"] is True, f"第 {i+1} 次生成编译失败"
        gc.assert_palette(data["code"], "green")
