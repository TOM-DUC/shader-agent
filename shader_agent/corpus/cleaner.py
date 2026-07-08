"""数据清洗。

职责：
1. 抽取 Image pass 与可选的 Common pass 作为 code_image / code_common；
2. 过滤外部资源依赖（texture / cubemap / mic / keyboard / buffer）的 shader
   —— 渲染后端只支持单 pass、无外部输入；
3. 长度过滤：太短（< min_code_chars）/ 太长（> max_code_chars）的剔除；
4. 点赞数过滤：likes >= min_likes（seed 默认很高，不会被剔）；
5. 去重：基于 (image, common) 代码的 sha256；
6. 落盘 clean/{id}.json，便于 diff 与复跑。
"""
from __future__ import annotations

from pathlib import Path

from shader_agent.config.settings import settings
from shader_agent.corpus.models import ShaderRecord
from shader_agent.utils.logger import logger


# 视为"需外部资源"的 input ctype
_EXTERNAL_CTYPES = {
    "texture", "cubemap", "volume", "video", "webcam",
    "music", "musicstream", "mic", "keyboard", "buffer",
}

# 可信来源（豁免 min_likes / 长度过滤；仍走外部资源与去重检查）
# - seed: 内嵌种子
# - local: 用户从 data/external_shaders/ 主动导入
# - shaders21k: 离线数据集，无 likes 信息但代码量足够
# - isf: ISF 格式标准，无 likes 信息
_TRUSTED_SOURCES = {"seed", "local", "shaders21k", "isf"}


def _extract_passes(rec: ShaderRecord) -> tuple[str, str, bool]:
    """从多 pass 中抽取 Image / Common；同时判断是否存在外部资源。"""
    image_code = ""
    common_code = ""
    has_external = False
    for p in rec.passes:
        ptype = (p.type or "").lower()
        for inp in p.inputs:
            ctype = str(inp.get("ctype", "")).lower()
            if ctype in _EXTERNAL_CTYPES:
                has_external = True
                break
        if ptype == "image":
            image_code = p.code or ""
        elif ptype == "common":
            common_code = p.code or ""
    return image_code, common_code, has_external


def clean_records(
    records: list[ShaderRecord],
    *,
    min_likes: int | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
    drop_external_assets: bool = True,
    keep_external_as_reference: bool = False,
) -> list[ShaderRecord]:
    """对 fetcher 输出的记录做清洗与去重，返回 clean 列表。

    Args:
        keep_external_as_reference: True 时将依赖外部资源的 shader 标记为
            ``reference_only=True`` 并保留，而不是直接丢弃。
    """
    min_likes = min_likes if min_likes is not None else settings.corpus.min_likes
    min_chars = min_chars if min_chars is not None else settings.corpus.min_code_chars
    max_chars = max_chars if max_chars is not None else settings.corpus.max_code_chars

    cleaned: list[ShaderRecord] = []
    seen_hashes: set[str] = set()

    stats = {
        "input": len(records),
        "drop_no_image": 0,
        "drop_external": 0,
        "drop_likes": 0,
        "drop_length": 0,
        "drop_dup": 0,
        "kept_reference": 0,
    }

    for rec in records:
        # 兼容两种来源：
        # - Shadertoy API: records 有 passes（多 pass 需抽取）
        # - ISF / shaders21k / seed: records 直接填了 code_image
        if rec.passes and rec.passes[0].code:
            img, common, has_ext = _extract_passes(rec)
        else:
            img = rec.code_image or ""
            common = rec.code_common or ""
            has_ext = rec.has_external_assets

        if not img:
            stats["drop_no_image"] += 1
            continue

        # seed 记录可能没填 passes 但已直接放进 code_image，做一次回填
        if not rec.code_image:
            rec.code_image = img
        if not rec.code_common:
            rec.code_common = common
        rec.has_external_assets = has_ext

        if drop_external_assets and has_ext and rec.source != "seed":
            # 新策略：外部资源型保留为参考
            if keep_external_as_reference:
                rec.reference_only = True
                stats["kept_reference"] += 1
            else:
                stats["drop_external"] += 1
                continue

        # 点赞过滤：trusted source（seed / local）豁免
        if rec.likes < min_likes and rec.source not in _TRUSTED_SOURCES:
            stats["drop_likes"] += 1
            continue

        # 长度过滤（trusted source 源豁免，便于内嵌示例与本地导入的最小示例不被剔）
        n = len(rec.code_image)
        if rec.source not in _TRUSTED_SOURCES and (n < min_chars or n > max_chars):
            stats["drop_length"] += 1
            continue

        # 去重
        h = rec.compute_code_hash()
        rec.code_hash = h
        if h in seen_hashes:
            stats["drop_dup"] += 1
            continue
        seen_hashes.add(h)

        cleaned.append(rec)

    logger.info(f"[cleaner] stats: {stats} -> kept {len(cleaned)}")
    return cleaned


def balance_by_topic(
    records: list[ShaderRecord],
    max_per_topic: int,
) -> list[ShaderRecord]:
    """按主题配额封顶，削减尾部冗余。

    对每个主题（tags_topic 中任一标签），最多保留 ``max_per_topic`` 条。
    多标签记录在一个已满主题中仍可能因另一标签被保留。
    用于极端不均衡时的硬性干预。
    """
    from collections import Counter, defaultdict

    # 计算每个记录选哪个"主标签"（出现最早的那个）
    topic_groups: dict[str, list[tuple[int, ShaderRecord]]] = defaultdict(list)
    for idx, r in enumerate(records):
        if r.tags_topic:
            # 用第一个有效标签作为主标签（排除 uncategorized）
            primary = r.tags_topic[0]
            if primary == "uncategorized" and len(r.tags_topic) > 1:
                primary = r.tags_topic[1]
        else:
            primary = "uncategorized"
        topic_groups[primary].append((idx, r))

    kept: set[int] = set()
    topic_counts: Counter = Counter()
    for topic, items in topic_groups.items():
        # 按优先级排序：非参考型优先、质量分高优先
        items.sort(key=lambda x: (not x[1].reference_only, -x[1].quality_score))
        for idx, r in items:
            if topic_counts[topic] < max_per_topic:
                kept.add(idx)
                topic_counts[topic] += 1

    result = [r for idx, r in enumerate(records) if idx in kept]
    logger.info(
        f"[cleaner] balance_by_topic(max_per_topic={max_per_topic}): "
        f"{len(records)} -> {len(result)}"
    )
    return result


def save_clean(records: list[ShaderRecord], out_dir: Path | None = None) -> Path:
    """把清洗后的记录逐条落到 clean/{id}.json。"""
    out_dir = out_dir or settings.corpus_clean_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        rec.save_json(out_dir)
    logger.info(f"[cleaner] saved {len(records)} clean records to {out_dir}")
    return out_dir
