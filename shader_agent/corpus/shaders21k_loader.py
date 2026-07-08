"""shaders21k 数据集导入器。

数据来源
--------
shaders21k（*Procedural Image Programs for Representation Learning*, NeurIPS'22）
是一个包含约 2 万条 GLSL shader 的数据集。其源码托管在 Google Drive：
  https://drive.google.com/file/d/1kIiBdeW9CEIfRlYOYTuxfWvN036k3Iig/view
（文件名 ``all_codes.zip``，约 40 MB）

导入策略
--------
1. 从仓库内嵌的 ``image_generation/shaders/programs/`` 导入 twigl 示例（约 12 条）；
2. 从已下载的 ``shader_codes/`` 目录导入：
   - 扫描 ``.fragment`` / ``.json`` / ``.glsl`` / ``.frag`` / ``.txt`` 文件；
   - 优先尝试 JSON 解析（Shadertoy 格式）；
   - JSON 失败则回退为裸 GLSL（twigl / txt 格式）；
   - 按代码去重。

文件格式
--------
- Shadertoy JSON: 含 ``info`` / ``renderpass`` 字段，形如 ``{ "info":{ "id":"...",... }, "renderpass":[{...}] }``
- 裸 GLSL（twigl）: 纯代码文本，``precision highp float; void main(){...}``
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from shader_agent.corpus.models import ShaderRecord
from shader_agent.utils.logger import logger


# 扫描的文件后缀
_SHADER_EXTS = {".fragment", ".json", ".glsl", ".frag", ".txt"}

# twigl 示例子目录（相对于仓库根）
_TWIGL_PROGRAM_DIR = "image_generation/shaders/programs"


def _parse_shadertoy_json(data: dict) -> ShaderRecord | None:
    """从 Shadertoy JSON 格式的记录解析为 ShaderRecord。"""
    info = data.get("info") or {}
    sid = (info.get("id") or "").strip()
    if not sid:
        return None

    name = (info.get("name") or "").strip()
    username = (info.get("username") or "").strip()
    likes = int(info.get("likes", 0) or 0)
    tags_raw = info.get("tags") or []

    passes = data.get("renderpass") or []
    image_code = ""
    for p in passes:
        if (p.get("type") or "").lower() == "image":
            image_code = (p.get("code") or "").strip()
            break

    if not image_code:
        return None

    code_hash = hashlib.sha256(image_code.encode()).hexdigest()[:12]

    return ShaderRecord(
        shader_id=sid,
        name=name or sid,
        username=username,
        likes=likes,
        tags_raw=tags_raw,
        code_image=image_code,
        code_hash=code_hash,
        source="shaders21k",
        is_generator=True,
        reference_only=False,
    )


def _parse_raw_glsl(path: Path, code: str) -> ShaderRecord | None:
    """从裸 GLSL 文件解析为 ShaderRecord。"""
    code = code.strip()
    if len(code) < 50:
        return None

    # 用文件路径中最后一两段做 id
    parts = path.parts
    stem = path.stem
    # 尝试取相对 shader_codes/ 的剩余路径
    sid = stem

    code_hash = hashlib.sha256(code.encode()).hexdigest()[:12]
    # 全局唯一 id 用 hash
    sid = f"twigl_{code_hash}"

    return ShaderRecord(
        shader_id=sid,
        name=stem,
        source="shaders21k",
        tags_raw=[],
        code_image=code,
        code_hash=code_hash,
        is_generator=True,
        reference_only=False,
    )


def _file_to_record(path: Path) -> ShaderRecord | None:
    """尝试将单个文件解析为 ShaderRecord（先 JSON 后裸 GLSL）。"""
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return None

    # 先试 JSON
    if path.suffix in _SHADER_EXTS:
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "info" in data:
                rec = _parse_shadertoy_json(data)
                if rec is not None:
                    return rec
        except (json.JSONDecodeError, ValueError):
            pass

    # JSON 失败 → 裸 GLSL
    return _parse_raw_glsl(path, raw)


def load_shader_codes_dir(shader_codes_dir: Path) -> list[ShaderRecord]:
    """扫描已下载的 ``shader_codes/`` 目录，导入全部 shader。

    支持格式：``.fragment``（JSON 或 GLSL）、``.json``、``.glsl``、``.frag``、``.txt``。
    """
    if not shader_codes_dir.is_dir():
        logger.warning(f"[s21k] 目录不存在: {shader_codes_dir}")
        return []

    files = []
    for ext in _SHADER_EXTS:
        files.extend(shader_codes_dir.rglob(f"*{ext}"))

    if not files:
        logger.warning(f"[s21k] 未在 {shader_codes_dir} 找到 shader 文件")
        return []

    seen_hashes: set[str] = set()
    records: list[ShaderRecord] = []
    for fp in sorted(files):
        try:
            rec = _file_to_record(fp)
            if rec is None:
                continue
            # 去重
            if rec.code_hash in seen_hashes:
                continue
            seen_hashes.add(rec.code_hash)
            records.append(rec)
        except Exception as e:
            logger.debug(f"[s21k] 跳过 {fp.name}: {e}")

    logger.info(
        f"[s21k] 从 {shader_codes_dir} 导入 {len(records)} 条 "
        f"（扫描 {len(files)} 文件）"
    )
    return records


def load_twigl_examples(repo_root: Path) -> list[ShaderRecord]:
    """从 shaders21k 仓库内嵌的 programs/ 目录导入 twigl 示例。"""
    prog_dir = repo_root / _TWIGL_PROGRAM_DIR
    if not prog_dir.is_dir():
        logger.warning(f"[s21k] twigl programs 目录不存在: {prog_dir}")
        return []

    records: list[ShaderRecord] = []
    import glob as _glob
    for fp in sorted(prog_dir.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix in _SHADER_EXTS or fp.suffix in {".txt", ""}:
            try:
                rec = _parse_raw_glsl(fp, fp.read_text(encoding="utf-8", errors="ignore"))
                if rec is not None:
                    records.append(rec)
            except Exception:
                pass

    logger.info(f"[s21k] 从 {prog_dir} 导入 {len(records)} 条 twigl 示例")
    return records


def load_shaders21k(
    repo_root: Path | None = None,
    download_dir: Path | None = None,
) -> list[ShaderRecord]:
    """主入口：合并 twigl 示例 + 已下载数据，按代码去重。

    Args:
        repo_root: shaders21k 仓库根目录（抽取内嵌 twigl 示例）。
        download_dir: 已下载解压的 ``shader_codes/`` 目录。

    Returns:
        去重后的 ShaderRecord 列表。
    """
    seen: set[str] = set()
    records: list[ShaderRecord] = []

    def _add(recs: list[ShaderRecord]) -> None:
        for r in recs:
            if r.code_hash and r.code_hash not in seen:
                seen.add(r.code_hash)
                records.append(r)

    if repo_root is not None:
        _add(load_twigl_examples(repo_root))

    if download_dir is not None:
        _add(load_shader_codes_dir(download_dir))

    logger.info(f"[s21k] 合计导入 {len(records)} 条（去重后）")
    return records
