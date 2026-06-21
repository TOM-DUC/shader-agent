"""UI runners 离线单测。

不启动 Gradio 进程；只验证：
  1) AssemblyOptions cache_key 唯一性；
  2) get_assembly() 命中缓存；
  3) run_analyze / run_generate / run_collaborate 在缺 DEEPSEEK_API_KEY 时
     仍能优雅回退（Analyzer/Generator 走 fallback 路径），不抛异常；
  4) render_code_to_png 在 mock 后端下正常返回 bytes；
  5) save_session 把图像 + 报告 + 代码落盘到约定路径。

注意：以上不需要联网，也不需要 gradio 安装；这正是 UI 模块解耦的目的。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from shader_agent.ui import runners as rn


# ------------------------------------------------------------------ #
# 全部用例都强制 mock 后端 + 关掉向量库，避开 GL/嵌入模型依赖
# ------------------------------------------------------------------ #

def _mock_opts(**over) -> rn.AssemblyOptions:
    base = dict(
        render_backend="mock",
        use_vector_store="off",
        use_llm_cache=False,
        enable_self_critique=False,
        max_fix_loops=0,
        top_k=2,
    )
    base.update(over)
    return rn.AssemblyOptions(**base)


@pytest.fixture(autouse=True)
def _isolate_assembly_cache():
    """每个测试前后清一次缓存，避免互相污染。"""
    rn.clear_assembly_cache()
    yield
    rn.clear_assembly_cache()


@pytest.fixture
def no_deepseek(monkeypatch):
    """强制走 fallback 路径：清掉 DEEPSEEK_API_KEY；
       同时让 settings.deepseek_api_key 为空。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from shader_agent.config.settings import settings
    monkeypatch.setattr(settings, "deepseek_api_key", "", raising=False)


# ------------------------------------------------------------------ #
# 1) 缓存 key
# ------------------------------------------------------------------ #

def test_cache_key_distinct():
    a = _mock_opts(top_k=2)
    b = _mock_opts(top_k=3)
    c = _mock_opts(top_k=2)
    assert rn._cache_key(a) != rn._cache_key(b)
    assert rn._cache_key(a) == rn._cache_key(c)


def test_get_assembly_caches(no_deepseek):
    opts = _mock_opts()
    asm1 = rn.get_assembly(opts)
    asm2 = rn.get_assembly(opts)
    assert asm1 is asm2
    # mock 后端 label 里应能看到 "mock"
    assert "mock" in asm1.backend_label.lower()
    # 向量库关闭时 label 里应能看到关闭/不可用字样
    assert ("关闭" in asm1.vstore_label) or ("不可用" in asm1.vstore_label) or \
           ("空" in asm1.vstore_label)


def test_get_assembly_different_opts_makes_new(no_deepseek):
    a = rn.get_assembly(_mock_opts(top_k=2))
    b = rn.get_assembly(_mock_opts(top_k=4))
    assert a is not b


# ------------------------------------------------------------------ #
# 2) 渲染（mock 后端必返回 bytes）
# ------------------------------------------------------------------ #

_TINY_SHADER = (
    "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
    "  vec2 uv = fragCoord / iResolution.xy;\n"
    "  fragColor = vec4(uv, 0.5, 1.0);\n"
    "}\n"
)


def test_render_code_to_png_mock(no_deepseek):
    opts = _mock_opts()
    png, err = rn.render_code_to_png(_TINY_SHADER, opts, width=64, height=48)
    assert err == ""
    assert isinstance(png, (bytes, bytearray))
    assert len(png) > 0  # mock 后端会写一个最小有效 PNG


def test_render_empty_code_returns_error(no_deepseek):
    opts = _mock_opts()
    png, err = rn.render_code_to_png("", opts)
    assert png is None
    assert err  # 非空错误信息


# ------------------------------------------------------------------ #
# 3) run_analyze / run_generate / run_collaborate
#    缺 LLM 时应优雅降级（Analyzer/Generator 用 fallback 模板）
# ------------------------------------------------------------------ #

def test_run_analyze_fallback_path(no_deepseek):
    """没有 LLM key 时 Analyzer 仍能产出基础 report。"""
    res = rn.run_analyze(_TINY_SHADER, _mock_opts())
    # ok 视各 Action 的 fallback 实现，至少不该是空错误
    assert "elapsed_ms" in res
    # 不抛异常即可；如果 ok=False，error 必须有解释而不是空字符串
    if not res["ok"]:
        assert res["error"], "失败时必须有错误说明"
    else:
        assert "Shader Analysis Report" in res["report_md"]


def test_run_analyze_empty_input():
    res = rn.run_analyze("   ", _mock_opts())
    assert res["ok"] is False
    assert "代码" in res["error"]


def test_run_generate_empty_input():
    res = rn.run_generate("", _mock_opts())
    assert res["ok"] is False
    assert "需求" in res["error"]


def test_run_collaborate_empty_inputs():
    r1 = rn.run_collaborate("", "改写一下", _mock_opts())
    assert r1["ok"] is False
    r2 = rn.run_collaborate(_TINY_SHADER, "", _mock_opts())
    assert r2["ok"] is False


def test_run_generate_returns_string_keys_only(no_deepseek):
    """payload 全字段必须 JSON-safe，便于落盘。"""
    res = rn.run_generate("画一个简单的渐变", _mock_opts())
    import json
    # references 是 list[dict]，应当能直接 json.dumps
    json.dumps(res.get("references") or [])
    # code 永远是 str（fallback 时可能为空字符串）
    assert isinstance(res.get("code", ""), str)


# ------------------------------------------------------------------ #
# 4) save_session 落盘
# ------------------------------------------------------------------ #

def test_save_session_writes_files(tmp_path, monkeypatch):
    """重写 settings.project_root 到 tmp_path，然后保存一次伪 session。"""
    from shader_agent.config.settings import settings
    monkeypatch.setattr(settings, "project_root", tmp_path, raising=False)

    # 构造一份伪 payload，含图像
    from PIL import Image
    img = Image.new("RGB", (8, 8), color=(255, 0, 128))
    payload = {
        "ok": True,
        "error": "",
        "elapsed_ms": 12.3,
        "code": "void mainImage(){}",
        "report_md": "# Hello\n",
        "image": img,
        "image_before": img,
        "image_after": img,
        "new_code": "void mainImage(){ /* rewritten */ }",
        "references": [{"shader_id": "x", "name": "X", "distance": 0.1, "tags": []}],
        # 不可序列化字段，应当被降级成 str
        "diagnostics": [object()],
    }
    out_dir = rn.save_session("unit_test", payload)
    p = Path(out_dir)
    assert p.is_dir()
    assert (p / "payload.json").exists()
    assert (p / "report.md").exists()
    assert (p / "generated.glsl").exists()
    assert (p / "rewritten.glsl").exists()
    assert (p / "image.png").exists()
    assert (p / "image_before.png").exists()
    assert (p / "image_after.png").exists()

    # payload.json 必须是合法 JSON
    import json
    obj = json.loads((p / "payload.json").read_text(encoding="utf-8"))
    assert obj["ok"] is True
    assert obj["elapsed_ms"] == pytest.approx(12.3)


# ------------------------------------------------------------------ #
# 5) examples 完整性
# ------------------------------------------------------------------ #

def test_examples_loadable():
    from shader_agent.ui import examples as ex
    ax = ex.analyzer_examples()
    gx = ex.generator_examples()
    cx = ex.collaborate_examples()
    assert len(ax) >= 3 and all(len(r) == 2 and r[1].count("mainImage") >= 1 for r in ax)
    assert len(gx) >= 3 and all(len(r) == 4 for r in gx)
    assert len(cx) >= 2 and all(len(r) == 3 and r[1].count("mainImage") >= 1 for r in cx)
