"""Shadertoy 数据采集。

依赖官方 API（需在 https://www.shadertoy.com/myapps 申请 key）：
  - GET /api/v1/shaders/query/{query}?sort=popular&from=0&num=N&key=KEY
      搜索 shader id
  - GET /api/v1/shaders/{shader_id}?key=KEY
      获取单 shader 详情

若 SHADERTOY_API_KEY 为空，会跳过远程拉取，只输出 seed shaders；
此时流水线后续步骤照常工作，方便冒烟测试与 demo。

落盘策略：每个 shader_id 一个 JSON 文件存在 data/shadertoy_corpus/raw/，
存在则跳过 → 支持断点续传。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from shader_agent.config.settings import settings
from shader_agent.corpus.models import RenderPass, ShaderRecord
from shader_agent.corpus.seed_shaders import get_seed_shaders
from shader_agent.utils.logger import logger


class ShadertoyFetcher:
    """Shadertoy API 拉取器。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        request_interval: float | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.shadertoy_api_key
        self.base_url = (base_url or settings.corpus.shadertoy_api_base).rstrip("/")
        self.request_interval = (
            request_interval
            if request_interval is not None
            else settings.corpus.request_interval_seconds
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "shader-agent/0.2"})

    # ---------- 远程调用 ----------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def search_ids(self, query: str, num: int) -> list[str]:
        """按关键词搜索，返回 shader id 列表。"""
        if not self.api_key:
            return []
        url = f"{self.base_url}/shaders/query/{query}"
        params = {"sort": "popular", "from": 0, "num": num, "key": self.api_key}
        try:
            data = self._get_json(url, params)
        except Exception as e:
            logger.warning(f"[fetcher] search '{query}' failed: {e}")
            return []
        ids = data.get("Results") or []
        logger.info(f"[fetcher] query='{query}' -> {len(ids)} ids")
        return ids

    def fetch_one(self, shader_id: str) -> dict[str, Any] | None:
        """获取单个 shader 的完整 JSON。"""
        if not self.api_key:
            return None
        url = f"{self.base_url}/shaders/{shader_id}"
        params = {"key": self.api_key}
        try:
            data = self._get_json(url, params)
        except Exception as e:
            logger.warning(f"[fetcher] fetch {shader_id} failed: {e}")
            return None
        if "Shader" not in data:
            logger.warning(f"[fetcher] {shader_id} response missing 'Shader'")
            return None
        return data["Shader"]

    # ---------- 转换 ----------
    @staticmethod
    def to_record(raw: dict[str, Any]) -> ShaderRecord | None:
        """把 Shadertoy 返回的原始 JSON 转成 ShaderRecord。"""
        try:
            info = raw.get("info") or {}
            passes_raw = raw.get("renderpass") or []
            passes: list[RenderPass] = []
            for p in passes_raw:
                passes.append(
                    RenderPass(
                        name=p.get("name", ""),
                        type=p.get("type", ""),
                        code=p.get("code", ""),
                        inputs=p.get("inputs", []) or [],
                        outputs=p.get("outputs", []) or [],
                    )
                )
            return ShaderRecord(
                shader_id=info.get("id", ""),
                name=info.get("name", ""),
                username=info.get("username", ""),
                description=info.get("description", "") or "",
                likes=int(info.get("likes", 0) or 0),
                viewed=int(info.get("viewed", 0) or 0),
                tags_raw=info.get("tags", []) or [],
                passes=passes,
                source="shadertoy",
            )
        except Exception as e:
            logger.warning(f"[fetcher] to_record failed: {e}")
            return None

    # ---------- 入口：批量采集 ----------
    def collect(
        self,
        out_dir: Path | None = None,
        queries: list[str] | None = None,
        per_query: int | None = None,
        max_total: int | None = None,
        include_seed: bool = True,
    ) -> list[ShaderRecord]:
        """按多 query 搜索 + 详情拉取，落盘 raw JSON 并返回 ShaderRecord 列表。

        - include_seed=True 时，最后并入内嵌种子数据。
        - 已经存在的 raw/{id}.json 会被复用，不重复请求。
        """
        out_dir = out_dir or settings.corpus_raw_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        queries = queries or settings.corpus.search_queries
        per_query = per_query or settings.corpus.per_query_limit
        max_total = max_total or settings.corpus.max_shaders

        # 1) 收集候选 id（去重）
        all_ids: list[str] = []
        seen: set[str] = set()
        if self.api_key:
            for q in queries:
                for sid in self.search_ids(q, per_query):
                    if sid not in seen:
                        seen.add(sid)
                        all_ids.append(sid)
                time.sleep(self.request_interval)
            all_ids = all_ids[: max_total or len(all_ids)]
            logger.info(f"[fetcher] unique ids collected: {len(all_ids)}")
        else:
            logger.warning(
                "[fetcher] SHADERTOY_API_KEY 未配置 → 跳过远程拉取，仅使用 seed shaders。"
                " 申请地址：https://www.shadertoy.com/myapps"
            )

        # 2) 详情拉取（或读已有 cache）
        records: list[ShaderRecord] = []
        for sid in all_ids:
            cache = out_dir / f"{sid}.json"
            if cache.exists():
                try:
                    raw = json.loads(cache.read_text(encoding="utf-8"))
                except Exception:
                    raw = None
            else:
                raw = self.fetch_one(sid)
                if raw is not None:
                    cache.write_text(
                        json.dumps(raw, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                time.sleep(self.request_interval)
            if raw is None:
                continue
            rec = self.to_record(raw)
            if rec is not None:
                records.append(rec)

        logger.info(f"[fetcher] records from API: {len(records)}")

        # 3) 合并 seed
        if include_seed:
            seeds = get_seed_shaders()
            records.extend(seeds)
            logger.info(f"[fetcher] seed appended: {len(seeds)}")

        return records
