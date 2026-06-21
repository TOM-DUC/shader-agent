"""离线单测：覆盖 models / cleaner / tagger，不触发网络与嵌入模型。

向量库 / 嵌入模型相关的集成测试需用 scripts.verify_corpus 跑（真实环境）。
"""
from __future__ import annotations

from shader_agent.corpus.cleaner import clean_records
from shader_agent.corpus.models import RenderPass, ShaderRecord
from shader_agent.corpus.seed_shaders import get_seed_shaders
from shader_agent.corpus.tagger import TOPIC_VOCAB, rule_tag, tag_records


def test_seed_shaders_loadable():
    seeds = get_seed_shaders()
    assert len(seeds) >= 5
    for s in seeds:
        assert s.shader_id
        assert s.code_image
        assert "mainImage" in s.code_image


def test_record_hash_dedup():
    a = ShaderRecord(
        shader_id="a", name="A", code_image="void mainImage(){}",
        passes=[RenderPass(type="image", code="void mainImage(){}")]
    )
    b = ShaderRecord(
        shader_id="b", name="B", code_image="void mainImage(){}",
        passes=[RenderPass(type="image", code="void mainImage(){}")]
    )
    assert a.compute_code_hash() == b.compute_code_hash()


def test_cleaner_filters_short_and_external():
    """short → drop, with external buffer input → drop."""
    too_short = ShaderRecord(
        shader_id="x1", name="x1",
        passes=[RenderPass(type="image", code="void mainImage(){}")]
    )
    external = ShaderRecord(
        shader_id="x2", name="x2", likes=999,
        passes=[RenderPass(
            type="image",
            code="void mainImage(){}" + "a" * 500,
            inputs=[{"ctype": "buffer"}],
        )],
    )
    seeds = get_seed_shaders()
    kept = clean_records(
        [too_short, external] + seeds,
        min_likes=0,
    )
    ids = {r.shader_id for r in kept}
    assert "x1" not in ids
    assert "x2" not in ids
    # seed 应当大部分保留（>=5）
    assert sum(1 for r in kept if r.source == "seed") >= 5


def test_rule_tag_covers_main_topics():
    """每个 seed 至少打到一个合理主题，且都来自受控词表。"""
    seeds = get_seed_shaders()
    # cleaner 顺手填 code_image
    seeds = clean_records(seeds, min_likes=0)
    tag_records(seeds, use_llm=False)
    vocab = set(TOPIC_VOCAB)
    expected_per_name = {
        "Raymarched Sphere": {"raymarching", "sdf"},
        "Value Noise": {"noise"},
        "Mandelbrot": {"fractal"},
        "Vignette Post Process": {"post-processing"},
        "Voronoi Cells": {"2d-pattern", "noise"},
        "Polar Kaleidoscope": {"2d-pattern"},
    }
    by_name = {s.name: s for s in seeds}
    for name, must in expected_per_name.items():
        assert name in by_name, f"missing seed {name}"
        got = set(by_name[name].tags_topic)
        # 每个主题词表内
        assert got.issubset(vocab), f"{name} produced out-of-vocab tags: {got}"
        # 至少命中一个期望主题
        assert got & must, f"{name} expected any of {must}, got {got}"


def test_doc_text_non_empty():
    seeds = get_seed_shaders()
    seeds = clean_records(seeds, min_likes=0)
    tag_records(seeds, use_llm=False)
    for s in seeds:
        text = s.to_doc_text()
        assert len(text) > 50
        meta = s.to_metadata()
        # ChromaDB 要求 metadata 字段都是基础类型
        for k, v in meta.items():
            assert isinstance(v, (str, int, float, bool)), f"{k} -> {type(v)}"


# =====================================================================
# 扩容验证
# =====================================================================

def test_v2_seed_count_and_diversity():
    """扩容后种子数 ≥ 25，且 TOPIC_VOCAB 中每个主题至少被一个 seed 覆盖。"""
    from shader_agent.corpus.tagger import TOPIC_VOCAB

    seeds = get_seed_shaders()
    assert len(seeds) >= 25, f"seed 数量 {len(seeds)} < 25，未达扩容目标"

    seeds = clean_records(seeds, min_likes=0)
    tag_records(seeds, use_llm=False)

    covered: set[str] = set()
    for s in seeds:
        covered.update(s.tags_topic)
    missing = set(TOPIC_VOCAB) - covered
    assert not missing, f"扩容后仍有主题未被任何 seed 覆盖: {missing}"


def test_v1_seeds_unchanged():
    """v1 段 seed01..seed08 名称必须保持不变（被多处脚本与测试硬编码）。"""
    by_id = {s.shader_id: s for s in get_seed_shaders()}
    expected = {
        "seed01": "Horizontal Gradient",
        "seed02": "Animated Circle",
        "seed03": "Raymarched Sphere",
        "seed04": "Value Noise",
        "seed05": "Mandelbrot",
        "seed06": "Voronoi Cells",
        "seed07": "Vignette Post Process",
        "seed08": "Polar Kaleidoscope",
    }
    for sid, name in expected.items():
        assert sid in by_id, f"v1 seed {sid} 缺失"
        assert by_id[sid].name == name, f"v1 seed {sid} 名称被改: {by_id[sid].name}"


# ---------- 本地导入 ----------

def test_local_loader_reads_glsl(tmp_path):
    from shader_agent.corpus.local_loader import load_local_dir

    code = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
        "  vec2 uv = fragCoord / iResolution.xy;\n"
        "  fragColor = vec4(uv, 0.5, 1.0);\n"
        "}\n"
    )
    (tmp_path / "my_test_shader.glsl").write_text(code, encoding="utf-8")
    (tmp_path / "my_test_shader.meta.json").write_text(
        '{"name":"My Test","description":"hi","tags_raw":["2d","custom"],"author":"me"}',
        encoding="utf-8",
    )
    # 一个不带 sidecar 的也要能读
    (tmp_path / "another.frag").write_text(code, encoding="utf-8")

    records = load_local_dir(tmp_path)
    assert len(records) == 2
    by_name = {r.name: r for r in records}
    assert "My Test" in by_name
    assert by_name["My Test"].username == "me"
    assert "custom" in by_name["My Test"].tags_raw
    assert all(r.source == "local" for r in records)
    # shader_id 须以 local_ 前缀
    assert all(r.shader_id.startswith("local_") for r in records)


def test_local_loader_skips_restricted_license(tmp_path):
    from shader_agent.corpus.local_loader import load_local_dir

    restricted = (
        "// (c) ACME Studios, 2024. All Rights Reserved.\n"
        "void mainImage(out vec4 c, in vec2 p){ c = vec4(1.0); }\n"
    )
    (tmp_path / "restricted.glsl").write_text(restricted, encoding="utf-8")
    assert load_local_dir(tmp_path) == []
    # 强制接受
    recs = load_local_dir(tmp_path, accept_restricted=True)
    assert len(recs) == 1


def test_local_loader_imported_through_cleaner(tmp_path):
    """local 来源应豁免 min_likes，不被 cleaner 误杀。"""
    from shader_agent.corpus.local_loader import load_local_dir

    code = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
        "  fragColor = vec4(1.0); }\n"  # 短，但 local 豁免长度
    )
    (tmp_path / "tiny.glsl").write_text(code, encoding="utf-8")
    recs = load_local_dir(tmp_path)
    kept = clean_records(recs, min_likes=1000)  # 用极高阈值；local 仍应通过
    assert len(kept) == 1


# ---------- Web scraper（mocked） ----------

def test_extract_shader_ids():
    from shader_agent.corpus.web_scraper import extract_shader_ids
    text = (
        "https://www.shadertoy.com/view/XlSSRV\n"
        "看看这个 https://www.shadertoy.com/view/MdX3Rr\n"
        "WdSXWy\n"  # 裸 id
        "# 不是 id（太长）: ABCDEFGHIJ\n"
    )
    ids = extract_shader_ids(text)
    assert "XlSSRV" in ids
    assert "MdX3Rr" in ids
    assert "WdSXWy" in ids


def test_web_scraper_to_record_handles_endpoint_shape():
    """to_record 应当处理 POST 端点返回的 dict（无 Shader 外壳）。"""
    from shader_agent.corpus.web_scraper import ShadertoyWebScraper
    raw = {
        "info": {
            "id": "TESTID",
            "name": "Test",
            "username": "alice",
            "description": "demo",
            "likes": 42,
            "viewed": 100,
            "tags": ["raymarching", "sdf"],
        },
        "renderpass": [
            {
                "name": "Image",
                "type": "image",
                "code": "void mainImage(out vec4 c, in vec2 p){ c=vec4(1.); }",
                "inputs": [],
                "outputs": [],
            }
        ],
    }
    rec = ShadertoyWebScraper.to_record(raw)
    assert rec is not None
    assert rec.shader_id == "TESTID"
    assert rec.name == "Test"
    assert rec.source == "shadertoy_scraped"
    assert "raymarching" in rec.tags_raw
    assert rec.passes[0].type == "image"


def test_web_scraper_caches_and_throttles(tmp_path, monkeypatch):
    """伪造 session.post，验证：
       1) 第一次抓后缓存被写到本地；
       2) 第二次抓直接命中缓存，不再调 post。
    """
    from shader_agent.corpus import web_scraper as ws_mod
    from shader_agent.corpus.web_scraper import ShadertoyWebScraper

    payload = [{
        "info": {"id": "AAAAAA", "name": "Mocked", "likes": 99,
                 "username": "x", "description": "", "viewed": 0, "tags": []},
        "renderpass": [{"name": "Image", "type": "image",
                        "code": "void mainImage(out vec4 c, in vec2 p){c=vec4(1.);}",
                        "inputs": [], "outputs": []}]
    }]

    call_count = {"n": 0}

    class FakeResp:
        status_code = 200
        def json(self): return payload
        text = ""

    def fake_post(self, url, data=None, headers=None, timeout=None):
        call_count["n"] += 1
        return FakeResp()

    monkeypatch.setattr("requests.sessions.Session.post", fake_post)

    scraper = ShadertoyWebScraper(min_interval=0.0, cache_dir=tmp_path)
    rec1 = scraper.fetch_by_id("AAAAAA")
    rec2 = scraper.fetch_by_id("AAAAAA")  # 第二次走 cache
    assert rec1 is not None and rec2 is not None
    assert rec1.shader_id == "AAAAAA"
    assert call_count["n"] == 1  # 仅请求一次

    # 缓存文件确实落盘
    cached = tmp_path / "AAAAAA.json"
    assert cached.exists()
