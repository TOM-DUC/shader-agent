"""父文档存储（SQLite）。

父子分块检索的下半段：子块（向量 / BM25）命中后，需要按 ``parent_id``（即
shader_id）取回完整 shader——完整代码、来源、许可证、质量分、算法摘要、关键函数。

用一张 SQLite 表存这些结构化字段，避免每次检索都去读磁盘上的 clean/*.json。
表与向量库、关键词索引解耦，可独立重建。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from shader_agent.config.settings import settings
from shader_agent.corpus.models import ShaderRecord
from shader_agent.utils.logger import logger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parent_docs (
    shader_id        TEXT PRIMARY KEY,
    name             TEXT,
    username         TEXT,
    source           TEXT,
    source_url       TEXT,
    license          TEXT,
    tags_topic       TEXT,
    categories       TEXT,
    key_functions    TEXT,
    visual_features  TEXT,
    algorithm_summary TEXT,
    compile_ok       INTEGER,
    render_ok        INTEGER,
    quality_score    REAL,
    code_image       TEXT,
    code_common      TEXT,
    is_generator     INTEGER DEFAULT 1,
    reference_only   INTEGER DEFAULT 0,
    indexed_at       TEXT
);
"""

# v1 → v2 迁移：旧表可能缺少三个新列
_MIGRATE_V2_COLUMNS = [
    ("categories", "TEXT"),
    ("is_generator", "INTEGER DEFAULT 1"),
    ("reference_only", "INTEGER DEFAULT 0"),
]


class ParentDocumentStore:
    """shader_id -> 完整记录 的 SQLite 存储。"""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (settings.project_root / "data" / "parent_docs.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate_v2()
        self._conn.commit()

    def _migrate_v2(self) -> None:
        """幂等地给旧表补充 v2 新列（无痛迁移）。"""
        cursor = self._conn.execute("PRAGMA table_info(parent_docs)")
        existing = {row[1] for row in cursor.fetchall()}
        for col_name, col_type in _MIGRATE_V2_COLUMNS:
            if col_name not in existing:
                self._conn.execute(
                    f"ALTER TABLE parent_docs ADD COLUMN {col_name} {col_type}"
                )
                logger.info(f"[parent] 迁移 v2: 新增列 {col_name} {col_type}")

    # ---------- 写入 ----------
    def upsert(self, records: list[ShaderRecord]) -> int:
        rows = []
        for r in records:
            rows.append(
                (
                    r.shader_id, r.name, r.username, r.source, r.source_url,
                    r.license, ",".join(r.tags_topic),
                    ",".join(r.categories),
                    ",".join(r.key_functions),
                    ",".join(r.visual_features), r.algorithm_summary,
                    int(r.compile_ok), int(r.render_ok), float(r.quality_score),
                    r.code_image, r.code_common,
                    int(r.is_generator), int(r.reference_only),
                    r.indexed_at,
                )
            )
        self._conn.executemany(
            """
            INSERT INTO parent_docs (
                shader_id, name, username, source, source_url, license,
                tags_topic, categories, key_functions, visual_features,
                algorithm_summary, compile_ok, render_ok, quality_score,
                code_image, code_common, is_generator, reference_only, indexed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(shader_id) DO UPDATE SET
                name=excluded.name, username=excluded.username,
                source=excluded.source, source_url=excluded.source_url,
                license=excluded.license, tags_topic=excluded.tags_topic,
                categories=excluded.categories,
                key_functions=excluded.key_functions,
                visual_features=excluded.visual_features,
                algorithm_summary=excluded.algorithm_summary,
                compile_ok=excluded.compile_ok, render_ok=excluded.render_ok,
                quality_score=excluded.quality_score,
                code_image=excluded.code_image, code_common=excluded.code_common,
                is_generator=excluded.is_generator,
                reference_only=excluded.reference_only,
                indexed_at=excluded.indexed_at
            """,
            rows,
        )
        self._conn.commit()
        logger.info(f"[parent] upserted {len(rows)} parent docs -> {self.db_path}")
        return len(rows)

    # ---------- 读取 ----------
    def get(self, shader_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT * FROM parent_docs WHERE shader_id = ?", (shader_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_many(self, shader_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not shader_ids:
            return {}
        placeholders = ",".join("?" * len(shader_ids))
        cur = self._conn.execute(
            f"SELECT * FROM parent_docs WHERE shader_id IN ({placeholders})",
            shader_ids,
        )
        return {row["shader_id"]: self._row_to_dict(row) for row in cur.fetchall()}

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["tags_topic"] = [t for t in (d.get("tags_topic") or "").split(",") if t]
        d["categories"] = [c for c in (d.get("categories") or "").split(",") if c]
        d["key_functions"] = [t for t in (d.get("key_functions") or "").split(",") if t]
        d["visual_features"] = [t for t in (d.get("visual_features") or "").split(",") if t]
        d["compile_ok"] = bool(d.get("compile_ok"))
        d["render_ok"] = bool(d.get("render_ok"))
        d["is_generator"] = bool(d.get("is_generator", True))
        d["reference_only"] = bool(d.get("reference_only", False))
        return d

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) AS n FROM parent_docs")
        return int(cur.fetchone()["n"])

    def reset(self) -> None:
        self._conn.execute("DELETE FROM parent_docs")
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
