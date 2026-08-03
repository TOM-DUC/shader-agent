"""检索接口：召回顺序与融合分构成。

单独测检索的价值在于把"检索质量"从大模型输出里剥离出来度量——分析/生成链路
即使检索退化，模型也常常能编出看起来合理的结果，缺陷会被掩盖到线上。
"""
from __future__ import annotations

import pytest

from tests_api.utils.assertions import assert_expect, assert_ok, assert_schema
from tests_api.utils.yaml_loader import parametrize, resolve_payload

pytestmark = [pytest.mark.api]


@parametrize("search_cases.yaml")
def test_search_cases(retrieval_api, shaders, case):
    payload = resolve_payload(case, shaders)
    assert_expect(retrieval_api.search(payload=payload), case["expect"])


@pytest.mark.contract
def test_search_response_matches_schema(retrieval_api):
    result = retrieval_api.search("raymarching", top_k=3)
    assert_ok(result)
    assert_schema(result.raw, "search_response.json")


@pytest.mark.smoke
def test_results_are_sorted_by_score_desc(retrieval_api):
    """排序错乱是检索最典型的静默缺陷：结果都在，就是顺序不对。"""
    data = assert_ok(retrieval_api.search("noise 噪声", top_k=5))
    scores = [i["score"] for i in data["items"]]
    assert scores == sorted(scores, reverse=True), f"融合分未降序：{scores}"


def test_score_components_are_exposed_for_debugging(retrieval_api):
    """融合分必须可拆解，否则线上排序异常时无法归因到哪一路召回。"""
    data = assert_ok(retrieval_api.search("raymarching sdf", top_k=1))
    item = data["items"][0]
    for k in ("vec_rel", "bm25", "tag_match", "quality"):
        assert 0.0 <= item[k] <= 1.0, f"{k} 越界：{item[k]}"
    assert item["score"] > 0


def test_top_k_limits_result_count(retrieval_api):
    for k in (1, 2, 3):
        data = assert_ok(retrieval_api.search("shader 图案", top_k=k))
        assert data["total"] <= k, f"top_k={k} 却返回 {data['total']} 条"


def test_search_is_deterministic(retrieval_api):
    """同 query 两次调用结果必须一致，否则回归基线无从建立。"""
    a = assert_ok(retrieval_api.search("mandelbrot 分形", top_k=3))
    b = assert_ok(retrieval_api.search("mandelbrot 分形", top_k=3))
    assert [i["shader_id"] for i in a["items"]] == [i["shader_id"] for i in b["items"]]


def test_irrelevant_query_still_returns_valid_structure(retrieval_api):
    """完全不相关的 query 也不能报错，结构必须稳定（可以返回低分结果）。"""
    data = assert_ok(retrieval_api.search("今天午饭吃什么", top_k=3))
    assert isinstance(data["items"], list)
    assert data["total"] == len(data["items"])
