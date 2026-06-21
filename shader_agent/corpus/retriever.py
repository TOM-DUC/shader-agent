"""混合检索器：统一编排向量召回、关键词召回、标签/质量过滤、融合排序、重排与阈值。

检索流程（父子分块）：

    向量召回 Top N（子块）  ┐
                            ├─► 按 parent_id 聚合到 shader 级
    BM25 召回 Top N（子块）  ┘
                            ↓
              标签匹配度 + 质量分 融合打分
                            ↓
                   可选交叉编码器重排
                            ↓
              相关度阈值过滤（低于阈值宁可不返回）
                            ↓
                       Top 3~5 shader

融合分（默认权重，可在 settings.retrieval 调整）：

    fused = 0.50 * vec_rel + 0.25 * bm25_norm + 0.15 * tag_match + 0.10 * quality

其中 vec_rel = 1 - cosine_distance；bm25_norm 为本次 query 内 min-max 归一化。

向后兼容：当子块集合 / 关键词索引 / 父文档表任一缺失时，自动降级为旧的 shader 级
向量检索，保证旧库无需重建也能工作。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shader_agent.config.settings import settings
from shader_agent.utils.logger import logger


@dataclass
class RetrievalHit:
    """一条 shader 级检索结果。"""

    shader_id: str
    name: str
    fused_score: float
    vec_rel: float
    bm25_norm: float
    tag_match: float
    quality: float
    tags_topic: list[str]
    code: str
    matched_chunks: list[str]  # 命中的子块标题（如函数名）
    algorithm_summary: str = ""
    key_functions: list[str] = field(default_factory=list)
    visual_features: list[str] = field(default_factory=list)
    source_url: str = ""
    license: str = ""
    matched_chunk_texts: list = field(default_factory=list)

    def build_reference_context(self, max_chars: int = 3600) -> str:
        sections = []
        meta = []
        if self.tags_topic: meta.append("tags=" + ", ".join(self.tags_topic[:8]))
        if self.visual_features: meta.append("visual=" + ", ".join(self.visual_features[:6]))
        if self.quality: meta.append(f"quality={self.quality:.2f}")
        if meta: sections.append("[meta] " + " | ".join(meta))
        if self.algorithm_summary: sections.append("[algorithm_summary]\n" + _clip(self.algorithm_summary, 700))
        added_titles = set()
        ranked = sorted(self.matched_chunk_texts, key=lambda x: (0 if x.get("kind") == "function" else 1, -float(x.get("score") or 0.0)))
        for ch in ranked[:4]:
            text = (ch.get("text") or "").strip()
            if not text: continue
            kind = ch.get("kind") or "chunk"; title = ch.get("title") or kind
            key = f"{kind}:{title}"
            if key in added_titles: continue
            added_titles.add(key)
            sections.append(f"[matched_{kind}: {title}]\n" + _clip(text, 1500))
        if "function:mainImage" not in added_titles and "function mainImage" not in "\n".join(sections):
            mi = _extract_function_snippet(self.code, "mainImage", limit=1400)
            if mi: sections.append("[context: mainImage]\n" + mi)
        for fn in self.key_functions[:3]:
            key = f"function:{fn}"
            if key in added_titles: continue
            snip = _extract_function_snippet(self.code, fn, limit=1200)
            if snip: sections.append(f"[context: key_function {fn}]\n" + snip)
        out = "\n\n".join(s for s in sections if s.strip()).strip()
        return _clip(out or _clip(self.code or "", max_chars), max_chars)

    def to_similar_payload(self) -> dict:
        ctx = self.build_reference_context()
        return {
            "shader_id": self.shader_id,
            "name": self.name,
            "distance": round(max(0.0, 1.0 - self.fused_score), 4),
            "tags_topic": self.tags_topic,
            "code_excerpt": _clip(self.code or ctx, 1200),
            "algorithm_summary": self.algorithm_summary,
            "matched_chunks": self.matched_chunks,
            "reference_context": ctx,
        }


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit: return text
    return text[:limit].rstrip() + "\n// ..."

def _extract_function_snippet(code: str, func_name: str, limit: int = 1200) -> str:
    """从完整 GLSL 中提取指定函数定义。"""
    if not code or not func_name: return ""
    import re
    pattern = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s+" + re.escape(func_name) + r"\s*\([^;{}]*\)\s*\{")
    m = pattern.search(code)
    if not m: return ""
    brace_start = code.find("{", m.start())
    if brace_start < 0: return ""
    depth = 0; i = brace_start
    while i < len(code):
        ch = code[i]
        if ch == "{": depth += 1
        elif ch == "}": depth -= 1
        if depth == 0: return _clip(code[m.start(): i + 1], limit)
        i += 1
    return _clip(code[m.start():], limit)

def _add_chunk(entry, h, score):
    """把子块命中并入 parent 聚合记录。"""
    title = h.get("title") or h.get("kind") or "chunk"
    kind = h.get("kind") or "chunk"
    key = f"{kind}:{title}"
    text = h.get("document") or ""
    chunks = entry.setdefault("chunks", {})
    old = chunks.get(key)
    if old is None or score > float(old.get("score") or 0.0):
        chunks[key] = {"kind": kind, "title": title, "text": text, "score": float(score)}

def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 if v > 0 else 0.0 for v in values]
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever:
    """向量 + BM25 + 过滤 + 重排的统一检索入口。"""

    def __init__(
        self,
        vector_store: Any,
        keyword_store: Any = None,
        parent_store: Any = None,
        reranker: Any = None,
    ) -> None:
        self.vstore = vector_store
        self.kstore = keyword_store
        self.pstore = parent_store
        self.reranker = reranker
        cfg = getattr(settings, "retrieval", None)
        self.w_vec = getattr(cfg, "w_vector", 0.50)
        self.w_bm25 = getattr(cfg, "w_bm25", 0.25)
        self.w_tag = getattr(cfg, "w_tag", 0.15)
        self.w_quality = getattr(cfg, "w_quality", 0.10)
        self.recall_k = getattr(cfg, "recall_k", 20)
        self.min_score = getattr(cfg, "min_score", 0.15)
        self.use_rerank = getattr(cfg, "use_rerank", True)

    # ---------- 主入口 ----------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        want_tags: list[str] | None = None,
    ) -> list[RetrievalHit]:
        """检索并返回 shader 级结果。want_tags 用于标签匹配度加分。"""
        if not query or not query.strip():
            return []

        # 子块级混合检索可用时走父子分块；否则降级 shader 级
        if self._chunks_available():
            return self._retrieve_hybrid(query, top_k, want_tags or [])
        return self._retrieve_legacy(query, top_k, want_tags or [])

    def _chunks_available(self) -> bool:
        try:
            return (
                self.vstore is not None
                and self.pstore is not None
                and self.vstore.chunk_count() > 0
            )
        except Exception:
            return False

    # ---------- 父子分块混合检索 ----------
    def _retrieve_hybrid(
        self, query: str, top_k: int, want_tags: list[str]
    ) -> list[RetrievalHit]:
        # 1) 向量召回（子块）
        vec_hits = self.vstore.query_chunks(query, top_k=self.recall_k)
        # 2) 关键词召回（子块）
        kw_hits = self.kstore.search(query, top_k=self.recall_k) if self.kstore else []

        # 3) 聚合到 parent_id：取该 parent 下最好的子块分
        agg: dict[str, dict[str, Any]] = {}
        for h in vec_hits:
            pid = h.get("parent_id") or ""
            if not pid:
                continue
            rel = 1.0 - float(h.get("distance") or 1.0)
            entry = agg.setdefault(pid, {"vec": 0.0, "bm25": 0.0, "chunks": {}})
            entry["vec"] = max(entry["vec"], rel)
            _add_chunk(entry, h, rel)

        bm25_raw: dict[str, float] = {}
        for h in kw_hits:
            pid = h.get("parent_id")
            if not pid:
                continue
            s = float(h.get("score") or 0.0)
            bm25_raw[pid] = max(bm25_raw.get(pid, 0.0), s)
        # 对 BM25 原始分做本次 query 内归一化
        if bm25_raw:
            pids = list(bm25_raw.keys())
            norm = _minmax([bm25_raw[p] for p in pids])
            bm25_norm_map = dict(zip(pids, norm))
        else:
            bm25_norm_map = {}
        for h in kw_hits:
            pid = h.get("parent_id") or ""
            if not pid:
                continue
            bm25_s = bm25_norm_map.get(pid, 0.0)
            entry = agg.setdefault(pid, {"vec": 0.0, "bm25": 0.0, "chunks": {}})
            entry["bm25"] = max(entry["bm25"], bm25_s)
            _add_chunk(entry, h, bm25_s)

        if not agg:
            return []

        # 4) 取父文档元数据，计算标签匹配度与质量分，融合
        parents = self.pstore.get_many(list(agg.keys()))
        candidates: list[dict[str, Any]] = []
        want = set(want_tags)
        for pid, sub in agg.items():
            p = parents.get(pid)
            if p is None:
                continue
            tags = p.get("tags_topic") or []
            tag_match = (len(want & set(tags)) / len(want)) if want else 0.0
            quality = float(p.get("quality_score") or 0.0)
            fused = (
                self.w_vec * sub["vec"]
                + self.w_bm25 * sub["bm25"]
                + self.w_tag * tag_match
                + self.w_quality * quality
            )
            matched_chunk_texts = sorted(sub.get("chunks", {}).values(), key=lambda x: -float(x.get("score", 0.0)))
            matched_chunk_titles = [c.get("title", "") for c in matched_chunk_texts]
            candidates.append(
                {
                    "shader_id": pid,
                    "name": p.get("name") or pid,
                    "vec_rel": sub["vec"],
                    "bm25_norm": sub["bm25"],
                    "tag_match": tag_match,
                    "quality": quality,
                    "fused_score": fused,
                    "tags_topic": tags,
                    "code": p.get("code_image") or "",
                    "text": p.get("algorithm_summary") or p.get("code_image", "")[:600],
                    "matched_chunks": matched_chunk_titles,
                    "matched_chunk_texts": matched_chunk_texts,
                    "algorithm_summary": p.get("algorithm_summary") or "",
                    "key_functions": p.get("key_functions") or [],
                    "visual_features": p.get("visual_features") or [],
                    "source_url": p.get("source_url") or "",
                    "license": p.get("license") or "",
                }
            )

        # 5) 可选重排
        if self.use_rerank and self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates)
            # 重排分归一化后并入融合分（七三开），保留融合分的可解释性
            rr = _minmax([c.get("rerank_score", 0.0) for c in candidates])
            for c, r in zip(candidates, rr):
                c["fused_score"] = 0.3 * c["fused_score"] + 0.7 * r
        candidates.sort(key=lambda c: c["fused_score"], reverse=True)

        # 6) 阈值过滤 + 截断
        hits: list[RetrievalHit] = []
        for c in candidates:
            if c["fused_score"] < self.min_score:
                continue
            hits.append(
                RetrievalHit(
                    shader_id=c["shader_id"],
                    name=c["name"],
                    fused_score=round(c["fused_score"], 4),
                    vec_rel=round(c["vec_rel"], 4),
                    bm25_norm=round(c["bm25_norm"], 4),
                    tag_match=round(c["tag_match"], 4),
                    quality=round(c["quality"], 4),
                    tags_topic=c["tags_topic"],
                    code=c["code"],
                    matched_chunks=c["matched_chunks"],
                    algorithm_summary=c.get("algorithm_summary", ""),
                    key_functions=c.get("key_functions", []),
                    visual_features=c.get("visual_features", []),
                    source_url=c.get("source_url", ""),
                    license=c.get("license", ""),
                    matched_chunk_texts=c.get("matched_chunk_texts", []),
                )
            )
            if len(hits) >= top_k:
                break
        logger.info(
            f"[retriever] hybrid query -> {len(hits)} hits "
            f"(vec={len(vec_hits)}, bm25={len(kw_hits)}, agg={len(agg)})"
        )
        return hits

    # ---------- 旧库降级：shader 级向量检索 ----------
    def _retrieve_legacy(
        self, query: str, top_k: int, want_tags: list[str]
    ) -> list[RetrievalHit]:
        if self.vstore is None:
            return []
        raw = self.vstore.query_by_text(query, top_k=top_k)
        want = set(want_tags)
        hits: list[RetrievalHit] = []
        for h in raw:
            md = h.get("metadata") or {}
            tags = [t for t in (md.get("tags_topic") or "").split(",") if t]
            rel = 1.0 - float(h.get("distance") or 1.0)
            tag_match = (len(want & set(tags)) / len(want)) if want else 0.0
            fused = self.w_vec * rel + self.w_tag * tag_match
            hits.append(
                RetrievalHit(
                    shader_id=h.get("shader_id", ""),
                    name=md.get("name", ""),
                    fused_score=round(fused, 4),
                    vec_rel=round(rel, 4),
                    bm25_norm=0.0,
                    tag_match=round(tag_match, 4),
                    quality=float(md.get("quality_score") or 0.0),
                    tags_topic=tags,
                    code=(h.get("document") or ""),
                    matched_chunks=[],
                )
            )
        return hits
