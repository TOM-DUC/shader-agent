"""异常与稳定性：依赖挂掉时系统应该怎么表现。

这类系统的可靠性风险几乎全在"外部依赖"上：大模型会超时、会限流、会吐脏数据；
GPU 可能没有；向量库可能空。功能测试跑得再全，也覆盖不到这些路径——因为正常
环境下它们根本不发生。所以用故障注入把它们变成**可重复的用例**。

每条用例都回答同一个问题：*依赖出问题时，接口是给出可归因的错误码并保住主流程，
还是抛一个 500 让调用方两眼一抹黑。*
"""
from __future__ import annotations

import pytest

from tests_api.utils import glsl_checker as gc
from tests_api.utils.assertions import assert_envelope, assert_error, assert_ok, assert_schema

pytestmark = [pytest.mark.api, pytest.mark.fault]


# ============================================================
# 大模型异常
# ============================================================

@pytest.mark.smoke
def test_llm_timeout_maps_to_504(shader_api, faults):
    """上游超时必须映射成 50401/504，而不是笼统的 500。

    区分"超时"和"内部错误"不是洁癖：前者可重试、可降级、可给用户"稍后再试"，
    后者必须报警。错误码合并了，运维手段也就合并了。
    """
    faults.set(llm_mode="timeout")
    result = shader_api.generate("生成一个蓝色波纹")
    assert_error(result, 50401, http=504)
    assert_envelope(result)
    assert_schema(result.raw, "error_response.json")


def test_llm_rate_limit_maps_to_429(shader_api, faults):
    faults.set(llm_mode="rate_limit")
    assert_error(shader_api.generate("生成一个蓝色波纹"), 42901, http=429)


def test_llm_auth_error_maps_to_503(shader_api, faults):
    faults.set(llm_mode="auth_error")
    assert_error(shader_api.generate("生成一个蓝色波纹"), 50301, http=503)


def test_llm_recovers_after_transient_failures(shader_api, faults):
    """前 2 次超时、第 3 次成功：验证重试链路真的能自愈，而不是首错即死。"""
    faults.set(llm_mode="timeout", llm_fail_times=2)
    last = None
    for _ in range(3):
        last = shader_api.generate("生成一个蓝色波纹")
        if last.ok:
            break
    assert last is not None and last.ok, f"重试后仍未恢复：{last}"


def test_malformed_llm_json_degrades_instead_of_crashing(shader_api, valid_shader, faults):
    """模型吐非法 JSON 时，分析链路应降级到静态解析，而不是 500。"""
    faults.set(llm_mode="malformed_json")
    data = assert_ok(shader_api.analyze(valid_shader))
    assert data["report"]["source_code"], "降级后仍应产出可用报告骨架"


def test_empty_llm_response_is_handled(shader_api, valid_shader, faults):
    faults.set(llm_mode="empty")
    result = shader_api.analyze(valid_shader)
    assert_envelope(result)
    assert result.code in (0, 50002), f"空响应应被明确处理，实际 {result}"


# ============================================================
# 编译失败与自愈
# ============================================================

def test_generation_self_heals_after_compile_failure(shader_api, faults):
    """首轮产出编不过的代码，修复轮应把它救回来，且 iterations 记录真实轮数。"""
    faults.set(llm_mode="invalid_code", llm_fail_times=1)
    data = assert_ok(shader_api.generate("生成一个蓝色波纹", max_fix_loops=2))
    assert data["compile_ok"] is True, "修复循环未生效"
    assert data["iterations"] >= 2, "修复轮数未被记录，指标失真"
    gc.assert_shader_ok(data["code"])


def test_persistent_compile_failure_is_reported_honestly(shader_api, faults):
    """反复修复仍失败时，必须**如实**返回 compile_ok=false 并带回错误原文。

    这里守的是"不许假装成功"：把编不过的代码当成品返回，比直接报错更糟——
    用户拿到手贴进 Shadertoy 才发现是红的。
    """
    faults.set(compiler_mode="always_fail")
    data = assert_ok(shader_api.generate("生成一个蓝色波纹", max_fix_loops=1))
    assert data["compile_ok"] is False
    assert data["compile_errors"], "失败必须带回编译器原文"
    assert data["iterations"] >= 2, "应当尝试过修复"


def test_fix_loop_respects_budget(shader_api, faults):
    """修复轮数必须受 max_fix_loops 约束，不能无限烧钱。"""
    faults.set(compiler_mode="always_fail")
    data = assert_ok(shader_api.generate("生成一个波纹", max_fix_loops=1))
    # 总轮数 = 首轮 + 最多 max_fix_loops 次修复
    assert data["iterations"] <= 2, f"超出修复预算：iterations={data['iterations']}"


# ============================================================
# 渲染与检索异常
# ============================================================

def test_renderer_unavailable_maps_to_503(shader_api, valid_shader, faults):
    faults.set(renderer_mode="unavailable")
    assert_error(shader_api.render(valid_shader), 50302, http=503)


def test_render_failure_does_not_block_generation(shader_api, faults):
    """渲染是增强能力不是必需能力：GPU 挂了，生成主流程必须照常返回代码。"""
    faults.set(renderer_mode="unavailable")
    data = assert_ok(shader_api.generate("生成一个波纹", with_render=True))
    assert data["code"], "渲染不可用不应影响代码产出"
    assert data["render"]["ok"] is False
    assert data["render"]["code"] == 50302


def test_empty_retrieval_degrades_to_zero_shot(shader_api, faults):
    """检索为空时，生成应退化为无参考生成，而不是报错。"""
    faults.set(retriever_mode="empty")
    data = assert_ok(shader_api.generate("生成一个蓝色波纹"))
    assert data["references"] == [], "检索为空时不应凭空出现参考"
    gc.assert_shader_ok(data["code"])


def test_retrieval_error_does_not_break_analysis(shader_api, valid_shader, faults):
    """向量库连接失败时，分析应降级为纯静态 + LLM，不应整体 500。"""
    faults.set(retriever_mode="error")
    result = shader_api.analyze(valid_shader)
    assert_envelope(result)
    assert result.code in (0, 50303), f"检索异常未被归类：{result}"


def test_retrieval_error_surfaces_on_search_endpoint(retrieval_api, faults):
    """但直接调检索接口时必须如实报错——这是它的主职能，不能静默返回空列表。"""
    faults.set(retriever_mode="error")
    assert_error(retrieval_api.search("raymarching"), 50303, http=503)


# ============================================================
# 故障复位
# ============================================================

def test_faults_are_reset_between_cases(shader_api, valid_shader, faults):
    """显式验证复位机制本身——故障配置串味会造成极难排查的偶发失败。"""
    faults.set(renderer_mode="unavailable")
    assert_error(shader_api.render(valid_shader), 50302)
    faults.reset()
    assert_ok(shader_api.render(valid_shader, width=64, height=64))
