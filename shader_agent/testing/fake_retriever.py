"""内存版检索器桩。

真实检索依赖 ChromaDB + bge-m3（约 2GB 权重）+ BM25 索引，CI 里既装不动也不
稳定。这里用一份 6 条的内存语料复现 `HybridRetriever` 的对外契约：

- 返回同样的 `RetrievalHit`（融合分、标签匹配、命中子块），下游 Action 无感；
- 打分规则是**确定的**（关键词命中数 + 标签匹配 + 质量分），因此可以断言
  "查 raymarching 必须把 raymarch 样本排在第一"这种检索相关性；
- 支持 `retriever_mode=empty/error`，用来验证"检索为空要降级而不是报错"。
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any

from shader_agent.corpus.retriever import RetrievalHit
from shader_agent.testing import faults


class RetrieverUnavailable(RuntimeError):
    """模拟向量库连接失败。"""


@dataclass
class FakeDoc:
    shader_id: str
    name: str
    tags: list[str]
    quality: float
    keywords: list[str]
    code: str
    summary: str
    functions: list[str] = field(default_factory=list)


FAKE_CORPUS: list[FakeDoc] = [
    FakeDoc(
        shader_id="fake_rm_001", name="Neon Raymarch Sphere",
        tags=["raymarching", "sdf"], quality=0.92,
        keywords=["raymarching", "ray", "march", "sdf", "sphere", "距离场", "光线步进"],
        code=(
            "float sdSphere(vec3 p, float r){ return length(p) - r; }\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
            "  vec2 uv=(fragCoord-0.5*iResolution.xy)/iResolution.y;\n"
            "  vec3 ro=vec3(0.0,0.0,3.0), rd=normalize(vec3(uv,-1.5));\n"
            "  float t=0.0;\n"
            "  for(int i=0;i<64;i++){ float d=sdSphere(ro+rd*t,1.0);\n"
            "    if(d<0.001) break; t+=d; if(t>20.0){ t=-1.0; break; } }\n"
            "  vec3 col = t>0.0 ? vec3(0.35,0.65,0.95) : vec3(0.0);\n"
            "  fragColor=vec4(col,1.0);\n}\n"
        ),
        summary="经典球体 raymarching：SDF 求交后按命中距离着色。",
        functions=["sdSphere", "mainImage"],
    ),
    FakeDoc(
        shader_id="fake_noise_002", name="Value Noise Field",
        tags=["noise", "2d-pattern"], quality=0.81,
        keywords=["noise", "hash", "fbm", "噪声", "分形噪声", "value"],
        code=(
            "float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
            "  vec2 uv=fragCoord/iResolution.xy*8.0+iTime;\n"
            "  vec2 i=floor(uv), f=fract(uv); f=f*f*(3.0-2.0*f);\n"
            "  float n=mix(mix(hash(i),hash(i+vec2(1,0)),f.x),\n"
            "              mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);\n"
            "  fragColor=vec4(vec3(n),1.0);\n}\n"
        ),
        summary="双线性插值的值噪声场，配合 iTime 平移形成流动感。",
        functions=["hash", "mainImage"],
    ),
    FakeDoc(
        shader_id="fake_fractal_003", name="Mandelbrot Zoom",
        tags=["fractal"], quality=0.88,
        keywords=["fractal", "mandelbrot", "julia", "分形", "迭代", "复数"],
        code=(
            "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
            "  vec2 c=(fragCoord-0.5*iResolution.xy)/iResolution.y*2.5;\n"
            "  vec2 z=vec2(0.0); float it=0.0;\n"
            "  for(int i=0;i<128;i++){ z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+c;\n"
            "    if(dot(z,z)>4.0) break; it+=1.0; }\n"
            "  vec3 col=vec3(0.55,0.30,0.90)*(it/128.0);\n"
            "  fragColor=vec4(col,1.0);\n}\n"
        ),
        summary="Mandelbrot 集迭代逃逸时间着色。",
        functions=["mainImage"],
    ),
    FakeDoc(
        shader_id="fake_voronoi_004", name="Voronoi Cells",
        tags=["voronoi", "2d-pattern"], quality=0.76,
        keywords=["voronoi", "cell", "泰森多边形", "元胞", "distance"],
        code=(
            "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
            "  vec2 uv=fragCoord/iResolution.xy*6.0; float md=8.0;\n"
            "  for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++){\n"
            "    vec2 g=floor(uv)+vec2(float(x),float(y));\n"
            "    vec2 p=g+fract(sin(dot(g,vec2(12.9,78.2)))*43758.5);\n"
            "    md=min(md,length(p-uv)); }\n"
            "  vec3 col=vec3(0.20,0.80,0.45)*md;\n"
            "  fragColor=vec4(col,1.0);\n}\n"
        ),
        summary="网格邻域搜索实现的 Voronoi 元胞图案。",
        functions=["mainImage"],
    ),
    FakeDoc(
        shader_id="fake_post_005", name="Chromatic Post FX",
        tags=["post-processing", "2d-pattern"], quality=0.70,
        keywords=["post", "processing", "vignette", "后处理", "色散", "暗角"],
        code=(
            "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
            "  vec2 uv=fragCoord/iResolution.xy;\n"
            "  float v=smoothstep(1.0,0.3,length(uv-0.5)*1.6);\n"
            "  vec3 col=vec3(0.98,0.45,0.20)*v;\n"
            "  fragColor=vec4(col,1.0);\n}\n"
        ),
        summary="暗角与色调映射构成的后处理效果。",
        functions=["mainImage"],
    ),
    FakeDoc(
        shader_id="fake_plasma_006", name="Classic Plasma",
        tags=["2d-pattern"], quality=0.64,
        keywords=["plasma", "sin", "wave", "波纹", "同心圆", "circle"],
        code=(
            "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
            "  vec2 uv=(fragCoord-0.5*iResolution.xy)/iResolution.y;\n"
            "  float w=0.5+0.5*sin(6.2831*length(uv)*3.0-iTime*1.5);\n"
            "  vec3 col=vec3(0.35,0.60,0.90)*w;\n"
            "  fragColor=vec4(col,1.0);\n}\n"
        ),
        summary="正弦波纹构成的同心圆等离子图案。",
        functions=["mainImage"],
    ),
]


class FakeRetriever:
    """对齐 HybridRetriever 的最小可用实现。"""

    def __init__(self, corpus: list[FakeDoc] | None = None,
                 min_score: float = 0.05) -> None:
        self.corpus = corpus if corpus is not None else FAKE_CORPUS
        self.min_score = min_score
        self.calls = 0
        self.last_query = ""

    # 与真实检索器保持同名，便于 UI/装配层无差别替换
    def warmup(self) -> None:  # pragma: no cover - 桩无需预热
        return

    def retrieve(self, query: str, top_k: int = 5,
                 want_tags: list[str] | None = None) -> list[RetrievalHit]:
        self.calls += 1
        self.last_query = query or ""
        cfg = faults.current()
        if cfg.retriever_latency_ms:
            _time.sleep(cfg.retriever_latency_ms / 1000.0)
        if cfg.retriever_mode == "error":
            raise RetrieverUnavailable(
                "chromadb: could not connect to persistent client (injected)"
            )
        if cfg.retriever_mode == "empty":
            return []
        if cfg.retriever_mode == "slow":
            _time.sleep(0.4)
        if not query or not query.strip():
            return []

        q = query.lower()
        want = [t.lower() for t in (want_tags or [])]
        scored: list[tuple[float, float, float, FakeDoc]] = []
        for doc in self.corpus:
            kw_hits = sum(1 for k in doc.keywords if k.lower() in q)
            name_hit = 1 if doc.name.lower().split()[0] in q else 0
            bm25 = min(1.0, (kw_hits + name_hit) / 3.0)
            tag = 0.0
            if want:
                tag = len(set(want) & {t.lower() for t in doc.tags}) / float(len(want))
            vec = min(1.0, 0.35 + 0.2 * kw_hits)
            fused = 0.50 * vec + 0.25 * bm25 + 0.15 * tag + 0.10 * doc.quality
            scored.append((fused, vec, bm25, doc))

        scored.sort(key=lambda x: (-x[0], x[3].shader_id))
        hits: list[RetrievalHit] = []
        for fused, vec, bm25, doc in scored[: max(1, int(top_k))]:
            if fused < self.min_score:
                continue
            tag_match = 0.0
            if want:
                tag_match = len(set(want) & {t.lower() for t in doc.tags}) / float(len(want))
            hits.append(
                RetrievalHit(
                    shader_id=doc.shader_id,
                    name=doc.name,
                    fused_score=round(fused, 4),
                    vec_rel=round(vec, 4),
                    bm25_norm=round(bm25, 4),
                    tag_match=round(tag_match, 4),
                    quality=doc.quality,
                    tags_topic=list(doc.tags),
                    code=doc.code,
                    matched_chunks=list(doc.functions),
                    algorithm_summary=doc.summary,
                    key_functions=list(doc.functions),
                    visual_features=list(doc.tags),
                    source_url=f"https://example.invalid/{doc.shader_id}",
                    license="CC BY-NC-SA 3.0",
                    matched_chunk_texts=[
                        {"kind": "function", "title": fn, "text": doc.code, "score": 0.9}
                        for fn in doc.functions[:2]
                    ],
                )
            )
        return hits


class FakeVectorStore:
    """只实现装配层与健康检查用到的计数接口。"""

    def __init__(self, n: int | None = None) -> None:
        self._n = len(FAKE_CORPUS) if n is None else n

    def count(self) -> int:
        return self._n

    def chunk_count(self) -> int:
        return self._n * 2

    def query(self, *args: Any, **kwargs: Any) -> list[Any]:  # pragma: no cover
        return []
