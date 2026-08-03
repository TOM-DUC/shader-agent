"""生成与改写：从"接口通了"到"产物真的能用"。

生成类接口最容易出现的假绿：返回 200、字段齐全、`code` 里也确实有字符串，
但那段 GLSL 贴回 Shadertoy 根本编不过，或者压根没体现用户要的调色板。
所以每条正向用例都串到规则层与编译层，端到端用例再串到图像层。
"""
from __future__ import annotations

import pytest

from tests_api.utils import glsl_checker as gc
from tests_api.utils import image_checker as img
from tests_api.utils.assertions import assert_error, assert_expect, assert_ok, assert_schema
from tests_api.utils.yaml_loader import parametrize, resolve_payload

pytestmark = [pytest.mark.api]


# ============================================================
# /generate
# ============================================================

@parametrize("generate_cases.yaml")
def test_generate_cases(shader_api, shaders, case):
    payload = resolve_payload(case, shaders)
    assert_expect(shader_api.generate(payload=payload), case["expect"])


@pytest.mark.smoke
def test_generated_code_satisfies_shadertoy_contract(shader_api):
    """产物必须满足"能贴回 Shadertoy 直接跑"的契约。"""
    data = assert_ok(shader_api.generate("生成一个蓝色调的波纹动画",
                                         palette="霓虹蓝", dynamic=True))
    gc.assert_shader_ok(data["code"], dynamic=True)
    assert data["compile_ok"] is True
    assert data["rule_report"]["passed"] is True


@pytest.mark.contract
def test_generate_response_matches_schema(shader_api):
    result = shader_api.generate("生成一个简单的渐变")
    assert_ok(result)
    assert_schema(result.raw, "generate_response.json")


@pytest.mark.parametrize("palette,key", [
    ("霓虹蓝 冷色调", "blue"),
    ("暖色 日落橙", "warm"),
    ("森林绿", "green"),
    ("霓虹紫", "purple"),
])
def test_palette_requirement_reaches_the_code(shader_api, palette, key):
    """调色板是用户能一眼看出对错的需求，必须逐条覆盖而不是抽查一个。"""
    data = assert_ok(shader_api.generate(f"生成一个{palette}的图案", palette=palette))
    gc.assert_palette(data["code"], key)


@pytest.mark.parametrize("dynamic", [True, False])
def test_dynamic_flag_controls_itime_usage(shader_api, dynamic):
    data = assert_ok(shader_api.generate("生成一个圆形图案", dynamic=dynamic))
    gc.assert_shader_ok(data["code"], dynamic=dynamic)


def test_generate_reports_iteration_count(shader_api):
    """一次成功时 iterations 应为 1（首轮即成品）——这是判"自愈能力"的基准值。"""
    data = assert_ok(shader_api.generate("生成一个简单的同心圆"))
    assert data["iterations"] == 1
    assert isinstance(data["references"], list)


def test_generate_with_render_produces_visible_image(shader_api):
    data = assert_ok(shader_api.generate(
        "生成一个蓝色波纹", palette="蓝色", with_render=True))
    assert data["render"]["ok"] is True
    img.assert_image_ok(data["render"]["image_base64"])


def test_generate_rejects_unknown_field(shader_api):
    """多传字段直接 422：静默忽略会让"参数没生效"变成最难查的问题。"""
    assert_error(
        shader_api.generate(payload={"description": "生成", "palete": "蓝"}),
        40001, http=422)


# ============================================================
# /remix
# ============================================================

@pytest.mark.smoke
def test_remix_returns_compilable_result(shader_api, valid_shader):
    data = assert_ok(shader_api.remix(valid_shader, "把主色调换成暖色"))
    gc.assert_shader_ok(data["code"])
    assert data["compile_ok"] is True
    assert data["base_code"].strip() == valid_shader.strip()


def test_remix_is_minimal_change_not_rewrite(shader_api, valid_shader):
    """改写的核心价值是可对比 diff：结构保留率过低说明模型在偷偷重写。"""
    data = assert_ok(shader_api.remix(valid_shader, "把主色调换成暖色"))
    kept = gc.structural_similarity(valid_shader, data["code"])
    assert kept >= 0.6, (
        f"原代码仅保留 {kept:.0%} 的行，改写退化成了重写\n"
        f"--- 改写后 ---\n{data['code'][:800]}")


def test_remix_actually_applies_the_instruction(shader_api, valid_shader):
    """既要"改得少"，也要"确实改了"——两个方向的断言必须成对出现。"""
    data = assert_ok(shader_api.remix(valid_shader, "把主色调换成暖色橙红"))
    assert data["code"].strip() != valid_shader.strip(), "改写指令未生效"
    gc.assert_palette(data["code"], "warm")


@pytest.mark.image
def test_remix_changes_the_rendered_frame(shader_api, valid_shader):
    """最终裁判是画面：改写前后渲染结果必须可见地不同。"""
    data = assert_ok(shader_api.remix(
        valid_shader, "把主色调换成暖色橙红", with_render=True))
    assert data["render"]["ok"] and data["render_before"]["ok"]
    before = img.load_stats(data["render_before"]["image_base64"])
    after = img.load_stats(data["render"]["image_base64"])
    assert before.dominant_channel == "b" and after.dominant_channel == "r", (
        f"改写前后主色未变化：{before.mean} → {after.mean}")


@pytest.mark.parametrize("bad,code", [
    ({"code": "", "instruction": "换个颜色"}, 40001),
    ({"code": "void mainImage(out vec4 c, in vec2 p){}"}, 40001),
    ({"code": "void mainImage(out vec4 c, in vec2 p){}",
      "instruction": "换个颜色", "max_fix_loops": 9}, 40001),
])
def test_remix_invalid_params(shader_api, bad, code):
    assert_error(shader_api.remix(payload=bad), code, http=422)
