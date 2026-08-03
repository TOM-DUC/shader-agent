"""渲染接口：从"返回了 bytes"推进到"画面是对的"。

图像层能抓到而其它层抓不到的缺陷，至少有四类：
  - 渲染成功但画面全黑（编译过了，但 uniform 没注入 / 颜色写错）
  - 分辨率被内部逻辑改写（请求 320×240 实际给了 512×384）
  - 上下颠倒（OpenGL 与 PNG 的 Y 轴方向不一致，常见回归）
  - 动画失效（iTime 没接上，不同时间返回同一帧）
"""
from __future__ import annotations

import pytest

from tests_api.utils import image_checker as img
from tests_api.utils.assertions import assert_error, assert_expect, assert_ok, assert_schema
from tests_api.utils.yaml_loader import parametrize, resolve_payload

pytestmark = [pytest.mark.api, pytest.mark.image]


@parametrize("render_cases.yaml")
def test_render_cases(shader_api, shaders, case):
    payload = resolve_payload(case, shaders)
    assert_expect(shader_api.render(payload=payload), case["expect"])


@pytest.mark.smoke
def test_render_returns_decodable_png_with_requested_size(shader_api, valid_shader):
    data = assert_ok(shader_api.render(valid_shader, width=256, height=192))
    stats = img.assert_image_ok(data["image_base64"], width=256, height=192)
    assert stats.std > 0.02, "画面应有明显明暗变化"


@pytest.mark.contract
def test_render_response_matches_schema(shader_api, valid_shader):
    result = shader_api.render(valid_shader, width=128, height=128)
    assert_ok(result)
    assert_schema(result.raw, "render_response.json")


def test_render_is_deterministic_for_same_input(shader_api, valid_shader):
    """同代码同 iTime 必须复现同一帧——这是图像回归基线成立的前提。"""
    a = assert_ok(shader_api.render(valid_shader, width=128, height=128, time=1.5))
    b = assert_ok(shader_api.render(valid_shader, width=128, height=128, time=1.5))
    img.assert_same_image(a["image_base64"], b["image_base64"])


def test_render_animation_actually_changes_over_time(shader_api, valid_shader):
    """iTime 变化必须带来画面变化，否则"动态效果"就是假的。"""
    a = assert_ok(shader_api.render(valid_shader, width=128, height=128, time=0.0))
    b = assert_ok(shader_api.render(valid_shader, width=128, height=128, time=1.0))
    img.assert_different_image(a["image_base64"], b["image_base64"])


def test_static_shader_is_stable_over_time(shader_api, shaders):
    """反过来，不含 iTime 的 shader 在不同时间必须完全一致。"""
    code = shaders["static_shader"]
    a = assert_ok(shader_api.render(code, width=128, height=128, time=0.0))
    b = assert_ok(shader_api.render(code, width=128, height=128, time=9.0))
    img.assert_same_image(a["image_base64"], b["image_base64"])


@pytest.mark.parametrize("palette_ref,channel", [
    ("valid_plasma", "b"),
    ("static_shader", "r"),
])
def test_render_dominant_color_matches_source(shader_api, shaders, palette_ref, channel):
    """基色向量应真实体现在画面上，而不只是写在代码里。"""
    data = assert_ok(shader_api.render(shaders[palette_ref], width=128, height=128))
    img.assert_dominant_channel(data["image_base64"], channel)


def test_render_rejects_unsupported_shader_with_precise_code(
        shader_api, unsupported_shader):
    """多通道 shader 应被 40004 明确拒绝，而不是抛一堆 GL 报错变成 500。"""
    assert_error(shader_api.render(unsupported_shader), 40004, http=422,
                 message_contains="多通道")


@pytest.mark.fault
def test_blank_frame_is_detected(shader_api, valid_shader, faults):
    """注入"渲染成功但输出全黑"，验证图像层断言真的能抓到它。

    这条用例本质上是在测**测试自身的有效性**——如果全黑图也能过，
    那前面所有渲染用例的绿色都不可信。
    """
    faults.set(renderer_mode="blank")
    data = assert_ok(shader_api.render(valid_shader, width=128, height=128))
    with pytest.raises(AssertionError, match="全黑|纯色|渲染失败|颜色数"):
        img.assert_image_ok(data["image_base64"], width=128, height=128)


@pytest.mark.gpu
def test_render_on_real_gl(shader_api, valid_shader, require_gpu):
    """有真 GL 时同一批断言直接跑在 moderngl 输出上（nightly 执行）。"""
    data = assert_ok(shader_api.render(valid_shader, width=256, height=192))
    img.assert_image_ok(data["image_base64"], width=256, height=192)
