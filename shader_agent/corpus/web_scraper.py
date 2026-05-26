"""阶段二补齐：Shadertoy 公开端点降级抓取。

适用前提
=========
当用户**没有官方 API key**（账户等级未达申请门槛、或政策限制）时，
本模块用 Shadertoy 网站自身用来加载 shader 的 ``POST /shadertoy`` 端点抓取数据。
该端点公开、无需 key，但是 Shadertoy 自家前端用的接口，**请谨慎使用并尊重 TOS**：

1. 严格控制速率（默认 ``min_interval=1.5s``，比 API 路径慢得多）；
2. 只抓**用户主动给出**的 shader id 列表（不要爬全站）；
3. 不要在 CI / 服务器后台无限循环；
4. 抓回来的 shader 默认是 CC-BY-NC-SA-3.0（Shadertoy 默认 license）—
   仅限学习与非商用；若打算公开二次分发产物，请逐条 attribution；
5. 若 Shadertoy 调整端点或返回 403，请改用本地 seed + local_loader 路径，
   或申请正式 API key。

接口
-----
.. code-block:: python

    scraper = ShadertoyWebScraper()
    rec = scraper.fetch_by_id("XlSSRV")
    recs = scraper.fetch_many(["XlSSRV", "MdX3Rr"])
    recs = scraper.fetch_from_urls([
        "https://www.shadertoy.com/view/XlSSRV",
        "https://www.shadertoy.com/view/MdX3Rr",
    ])

实现笔记
---------
- 我们模拟 Shadertoy 前端发起的请求：
    POST https://www.shadertoy.com/shadertoy
    Headers: Referer=https://www.shadertoy.com/view/{id}
             User-Agent=<浏览器风格>
             Origin=https://www.shadertoy.com
    Body (form-urlencoded): s={"shaders":["{id}"]}&nt=1&nl=1&np=1
- 返回 JSON 是一个数组 [{"info":{...}, "renderpass":[...]}]；
- 与官方 API 返回的 ``{"Shader": {...}}`` 结构略有不同（无 Shader 包装层），
  本模块负责适配。
- 失败优先 200/422/403/429 都返回 None，不抛异常上层。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from shader_agent.config.settings import settings
from shader_agent.corpus.models import RenderPass, ShaderRecord
from shader_agent.utils.logger import logger


SHADERTOY_VIEW_RE = re.compile(
    r"https?://(?:www\.)?shadertoy\.com/view/([A-Za-z0-9]{6,7})",
    re.IGNORECASE,
)


def extract_shader_ids(text: str) -> list[str]:
    """从一段文本（URL 列表 / 自由文本）中抽取所有 Shadertoy shader id。

    支持的输入格式：
      - "https://www.shadertoy.com/view/XlSSRV"
      - 多行混杂的 URL
      - 已经是裸 id 的 6~7 位字母数字串（一行一个）
    """
    ids: list[str] = []
    seen: set[str] = set()
    for m in SHADERTOY_VIEW_RE.finditer(text):
        sid = m.group(1)
        if sid not in seen:
            seen.add(sid)
            ids.append(sid)
    # 兜底：每行如果是 6-7 位 [A-Za-z0-9]，也算
    for line in text.splitlines():
        line = line.strip()
        if 6 <= len(line) <= 7 and line.isalnum() and line not in seen:
            seen.add(line)
            ids.append(line)
    return ids


class ShadertoyWebScraper:
    """Shadertoy 公开端点降级抓取器。

    与 ``ShadertoyFetcher`` 的关系：API key 在手时优先用 fetcher（更稳、TOS 友好）；
    无 key 时回落到本类。
    """

    ENDPOINT = "https://www.shadertoy.com/shadertoy"
    BASE = "https://www.shadertoy.com"

    # 浏览器风格 UA；Shadertoy 后端会拒绝过于明显的脚本 UA
    DEFAULT_UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )

    def __init__(
        self,
        *,
        min_interval: float = 1.5,
        timeout_seconds: int = 30,
        user_agent: str | None = None,
        cache_dir: Path | None = None,
        max_retries: int = 2,
    ) -> None:
        self.min_interval = min_interval
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent or self.DEFAULT_UA,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": self.BASE,
        })
        self._last_call_at: float = 0.0

        # 落盘缓存，命中则不再请求（与 fetcher.collect 的 raw/ 缓存互不冲突）
        self.cache_dir = cache_dir or (settings.corpus_raw_dir / "scraped")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 速率限制 ----------
    def _throttle(self) -> None:
        now = time.perf_counter()
        wait = self._last_call_at + self.min_interval - now
        if wait > 0:
            time.sleep(wait)
        self._last_call_at = time.perf_counter()

    # ---------- 单条抓取 ----------
    def _cache_path(self, shader_id: str) -> Path:
        return self.cache_dir / f"{shader_id}.json"

    def fetch_by_id(self, shader_id: str, *, use_cache: bool = True) -> ShaderRecord | None:
        """抓取单个 shader。失败返回 None。"""
        shader_id = (shader_id or "").strip()
        if not shader_id:
            return None

        cached = self._cache_path(shader_id)
        if use_cache and cached.exists():
            try:
                raw = json.loads(cached.read_text(encoding="utf-8"))
                logger.debug(f"[web_scraper] cache hit {shader_id}")
                return self.to_record(raw)
            except Exception:
                pass  # 缓存坏了，重新抓

        raw = self._do_fetch(shader_id)
        if raw is None:
            return None
        try:
            cached.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[web_scraper] 缓存写入失败: {e}")
        return self.to_record(raw)

    def _do_fetch(self, shader_id: str) -> dict[str, Any] | None:
        """实际发请求；返回原始 shader dict（已去掉 list 外壳）或 None。"""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            headers = {
                # Referer 必须，端点要靠它确认是从前端跳来的
                "Referer": f"{self.BASE}/view/{shader_id}",
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            body = {
                "s": json.dumps({"shaders": [shader_id]}),
                "nt": "1",
                "nl": "1",
                "np": "1",
            }
            try:
                r = self.session.post(
                    self.ENDPOINT, data=body, headers=headers,
                    timeout=self.timeout_seconds,
                )
            except Exception as e:
                logger.warning(
                    f"[web_scraper] {shader_id} attempt {attempt+1} 网络异常: {e}"
                )
                continue
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception as e:
                    logger.warning(f"[web_scraper] {shader_id} JSON 解析失败: {e}")
                    return None
                # 返回是 list[{...}]，正常情况下取第一个
                if isinstance(data, list) and data:
                    return data[0]
                if isinstance(data, dict):  # 偶尔会包成 {"Shader": {...}}
                    return data.get("Shader") or data
                logger.warning(f"[web_scraper] {shader_id} 返回意外结构: {type(data)}")
                return None
            if r.status_code in (429, 503):
                # 限流，加倍 backoff
                backoff = self.min_interval * (2 ** attempt) * 2
                logger.warning(
                    f"[web_scraper] {shader_id} {r.status_code} rate-limited, "
                    f"backoff {backoff:.1f}s"
                )
                time.sleep(backoff)
                continue
            logger.warning(
                f"[web_scraper] {shader_id} HTTP {r.status_code}: "
                f"{(r.text or '')[:160]}"
            )
            return None
        return None

    # ---------- 批量入口 ----------
    def fetch_many(
        self,
        shader_ids: list[str],
        *,
        use_cache: bool = True,
    ) -> list[ShaderRecord]:
        out: list[ShaderRecord] = []
        for sid in shader_ids:
            rec = self.fetch_by_id(sid, use_cache=use_cache)
            if rec is not None:
                out.append(rec)
        logger.info(f"[web_scraper] {len(out)}/{len(shader_ids)} 成功")
        return out

    def fetch_from_urls(
        self,
        urls: list[str] | str,
        *,
        use_cache: bool = True,
    ) -> list[ShaderRecord]:
        """支持传 URL 列表或一大段文本，自动抽 id。"""
        if isinstance(urls, list):
            joined = "\n".join(urls)
        else:
            joined = urls
        ids = extract_shader_ids(joined)
        if not ids:
            logger.warning("[web_scraper] 输入里没找到任何 shader id")
            return []
        return self.fetch_many(ids, use_cache=use_cache)

    # ---------- 适配层：scraped raw → ShaderRecord ----------
    @staticmethod
    def to_record(raw: dict[str, Any]) -> ShaderRecord | None:
        """把端点返回的 dict 转为 ShaderRecord。结构与官方 API 类似但缺 Shader 外壳。"""
        if not isinstance(raw, dict):
            return None
        info = raw.get("info") or {}
        passes_raw = raw.get("renderpass") or []
        passes: list[RenderPass] = []
        for p in passes_raw:
            passes.append(RenderPass(
                name=p.get("name", ""),
                type=p.get("type", ""),
                code=p.get("code", ""),
                inputs=p.get("inputs", []) or [],
                outputs=p.get("outputs", []) or [],
            ))
        try:
            return ShaderRecord(
                shader_id=info.get("id", ""),
                name=info.get("name", ""),
                username=info.get("username", ""),
                description=info.get("description", "") or "",
                likes=int(info.get("likes", 0) or 0),
                viewed=int(info.get("viewed", 0) or 0),
                tags_raw=info.get("tags", []) or [],
                passes=passes,
                source="shadertoy_scraped",
            )
        except Exception as e:
            logger.warning(f"[web_scraper] to_record 失败: {e}")
            return None
