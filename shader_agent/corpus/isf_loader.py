"""ISF（Interactive Shader Format）导入器。

ISF 文件是以 ``/* { JSON 元数据 } */`` 开头的 ``.fs`` 文件，包含：
- CATEGORIES（数组，表示效果分类，如 ``["Blur"]``）
- INPUTS（数组，表示输入参数与外部贴图）
- GLSL 代码体

导入策略：
- 解析 JSON 元数据头，提取 CATEGORIES 与 INPUTS；
- 有 ``inputImage`` 输入的视为**滤镜**（``reference_only=True``），
  无外部贴图输入的视为**生成器**（``is_generator=True``）；
- CATEGORIES 通过 ``category_map.isf_categories_to_tags`` 映射到 v2 标签；
- 代码体直接作为 ``code_image``。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from shader_agent.corpus.models import ShaderRecord
from shader_agent.utils.logger import logger


# ISF 文件头的 JSON 元数据匹配：/* { ... } */  或  /*! ... */
_RE_ISF_HEADER = re.compile(
    r"/\*[\!\*]?\s*(\{.*?\})\s*\*/",
    re.DOTALL,
)


def _parse_isf_file(path: Path) -> ShaderRecord | None:
    """解析单个 .fs 文件，返回 ShaderRecord 或 None（失败时）。"""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # 抽取 JSON 元数据头
    m = _RE_ISF_HEADER.search(raw)
    if not m:
        logger.warning(f"[isf] 无 JSON 元数据头: {path}")
        return None
    header_text = m.group(1).strip()
    try:
        meta = json.loads(header_text)
    except json.JSONDecodeError as e:
        logger.warning(f"[isf] JSON 解析失败 {path.name}: {e}")
        return None

    # GLSL 代码体：去掉元数据头
    code_body = _RE_ISF_HEADER.sub("", raw, count=1).strip()

    # 处理 INPUTS：判断是否依赖外部贴图
    inputs: list[dict] = meta.get("INPUTS", []) or []
    has_input_image = any(
        inp.get("NAME") == "inputImage" or inp.get("TYPE") == "inputImage"
        for inp in inputs
    )

    # CATEGORIES 映射到 v2 标签
    categories: list[str] = meta.get("CATEGORIES", []) or []
    from shader_agent.corpus.category_map import isf_categories_to_tags
    tags = isf_categories_to_tags(categories)

    # CREDIT / 作者
    credit = meta.get("CREDIT", "") or ""

    # 生成 shader_id：基于代码 hash 截断
    code_hash = hashlib.sha256(code_body.encode()).hexdigest()[:12]
    shader_id = f"isf_{code_hash}"

    # name：从文件名去后缀，或 meta 中的 TITLE
    name = meta.get("TITLE", "").strip() or path.stem

    # 构建 ShaderRecord
    rec = ShaderRecord(
        shader_id=shader_id,
        name=name,
        username=credit or "ISF",
        source="isf",
        source_url="",
        tags_raw=categories,
        tags_topic=tags,
        code_image=code_body,
        code_hash=code_hash,
        has_external_assets=has_input_image,
        is_generator=not has_input_image,
        reference_only=has_input_image,
    )
    return rec


def load_isf_dir(isf_dir: Path) -> list[ShaderRecord]:
    """扫描 ISF 目录下所有 ``.fs`` 文件，导入为 ShaderRecord 列表。"""
    fs_files = sorted(isf_dir.glob("*.fs"))
    if not fs_files:
        logger.warning(f"[isf] 未在 {isf_dir} 找到 .fs 文件")
        return []

    records: list[ShaderRecord] = []
    for fp in fs_files:
        try:
            rec = _parse_isf_file(fp)
            if rec is not None:
                records.append(rec)
        except Exception as e:
            logger.warning(f"[isf] 跳过 {fp.name}: {e}")

    logger.info(f"[isf] 导入 {len(records)}/{len(fs_files)} 条 <- {isf_dir}")
    return records
