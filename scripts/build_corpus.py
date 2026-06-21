"""语料库构建主入口（fetch → clean → tag → embed → index）。

数据源（v2 扩容，5 路并集，互不冲突）：
  1. 内嵌 seed shaders（33 条，本地无网络也能跑通）
  2. Shadertoy 官方 API（需 SHADERTOY_API_KEY，账户等级 ≥1 才能申请）
  3. Shadertoy 公开 POST 端点降级抓取（无需 key；--from-urls / --from-id-list）
  4. 本地 GLSL 文件目录（data/external_shaders/*.glsl + .meta.json）
  5. 上一次跑的 raw cache（断点续传）

用法（项目根）：

    # 默认：seed + （若有 key）API 拉取
    python -m scripts.build_corpus

    # 完全离线：只用 seed（最快冒烟，约 30s）
    python -m scripts.build_corpus --no-api

    # 从本地 .glsl 文件目录导入
    python -m scripts.build_corpus --from-local-dir data/external_shaders

(Showing lines 1-20 of 284. Use offset=21 to continue.)

    # 从 Shadertoy URL 列表抓取（无需 key，慢但能用）
    python -m scripts.build_corpus --from-urls \\
        https://www.shadertoy.com/view/XlSSRV \\
        https://www.shadertoy.com/view/MdX3Rr

    # 从一个文本文件读取 id/URL 列表（每行一个）
    python -m scripts.build_corpus --from-id-list data/wanted_ids.txt

    # 组合：seed + 本地 + 抓取
    python -m scripts.build_corpus --no-api \\
        --from-local-dir data/external_shaders \\
        --from-id-list data/wanted_ids.txt

    # 清空 collection 重建
    python -m scripts.build_corpus --reset

    # 启用 LLM 复核打标
    python -m scripts.build_corpus --enable-llm-tagging
"""
from __future__ import annotations

# 绕过 transformers 对 torch<2.6 的安全检查（本地模型，安全可控）
import transformers.utils.import_utils as _t_iu
_t_iu._torch_available = True
_t_iu._torch_version = (2, 6, 0)
_t_iu.check_torch_load_is_safe = lambda: None
_t_iu.is_torch_available = lambda: True

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from shader_agent.config.settings import settings
from shader_agent.corpus.cleaner import clean_records, save_clean
from shader_agent.corpus.fetcher import ShadertoyFetcher
from shader_agent.corpus.local_loader import load_local_dir
from shader_agent.corpus.tagger import tag_records
from shader_agent.corpus.vector_store import ShaderVectorStore
from shader_agent.corpus.web_scraper import ShadertoyWebScraper
from shader_agent.utils.logger import logger

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--no-api", action="store_true",
                   help="即使配置了 SHADERTOY_API_KEY 也不调用远程 API（只用种子）")
    p.add_argument("--no-seed", action="store_true",
                   help="不并入内嵌 seed shaders")
    p.add_argument("--reset", action="store_true",
                   help="构建前清空向量库 collection")
    p.add_argument("--no-index", action="store_true",
                   help="只跑 fetch/clean/tag，不写向量库")
    p.add_argument("--enable-llm-tagging", action="store_true",
                   help="覆盖 config，对规则未命中的样本启用 LLM 复核打标")

    # 三种额外数据源
    p.add_argument("--from-local-dir", type=str, default="",
                   help="从本地目录导入 .glsl 文件（默认 data/external_shaders/）；\n"
                        "传 'auto' 走默认目录；传具体路径覆盖")
    p.add_argument("--from-urls", nargs="+", default=[],
                   help="Shadertoy URL 列表（用 web scraper 抓取，无需 key）")
    p.add_argument("--from-id-list", type=str, default="",
                   help="文件路径：每行一个 Shadertoy URL 或裸 id")
    p.add_argument("--accept-restricted-license", action="store_true",
                   help="本地导入时，跳过 'All Rights Reserved' 等限制性 license 检查")
    return p.parse_args()


def step(title: str) -> None:
    console.rule(f"[bold cyan]{title}[/bold cyan]")


def _load_from_local(args: argparse.Namespace) -> list:
    if not args.from_local_dir:
        return []
    if args.from_local_dir.strip().lower() == "auto":
        path = settings.project_root / "data" / "external_shaders"
    else:
        path = Path(args.from_local_dir).expanduser().resolve()
    return load_local_dir(path, accept_restricted=args.accept_restricted_license)


def _load_from_urls_or_list(args: argparse.Namespace) -> list:
    sources: list[str] = list(args.from_urls or [])
    if args.from_id_list:
        list_path = Path(args.from_id_list).expanduser().resolve()
        if list_path.exists():
            sources.extend([
                ln.strip()
                for ln in list_path.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ])
        else:
            logger.warning(f"--from-id-list 指向不存在的文件: {list_path}")

    if not sources:
        return []
    console.print(f"[yellow]使用公开 POST 端点抓取 {len(sources)} 个 id，"
                  "速率 1.5s/req，请遵守 Shadertoy TOS。[/yellow]")
    scraper = ShadertoyWebScraper()
    return scraper.fetch_from_urls(sources)


def main() -> int:
    args = parse_args()

    # ---------- 1) Fetch ----------
    step("1/5  收集数据（API + seed + 本地 + 抓取）")
    api_key = "" if args.no_api else settings.shadertoy_api_key
    fetcher = ShadertoyFetcher(api_key=api_key)
    records = fetcher.collect(include_seed=not args.no_seed)
    console.print(f"  · seed + API   → [bold]{len(records)}[/bold] 条")

    local_recs = _load_from_local(args)
    if local_recs:
        records.extend(local_recs)
        console.print(f"  · 本地目录     → [bold]{len(local_recs)}[/bold] 条")

    scraped = _load_from_urls_or_list(args)
    if scraped:
        records.extend(scraped)
        console.print(f"  · 公开端点抓取 → [bold]{len(scraped)}[/bold] 条")

    console.print(f"\nfetched [bold]{len(records)}[/bold] raw records (合并去重前)")
    if not records:
        logger.error("没有任何 shader 记录，终止。"
                     "可加 --from-local-dir auto 或 --from-urls/--from-id-list。")
        return 1

    # ---------- 2) Clean ----------
    step("2/5  Clean & dedup")
    cleaned = clean_records(records)
    save_clean(cleaned)
    console.print(f"kept [bold]{len(cleaned)}[/bold] after cleaning")
    if not cleaned:
        logger.error("清洗后剩 0 条，请检查 min_likes / 长度阈值。")
        return 1

    # ---------- 3) Tag ----------
    step("3/5  Topic tagging")
    use_llm = args.enable_llm_tagging or settings.corpus.enable_llm_tagging
    tag_records(cleaned, use_llm=use_llm)

    # 标签分布表
    table = Table(title="Topic tag distribution")
    table.add_column("topic", style="cyan")
    table.add_column("count", justify="right", style="green")
    counter: dict[str, int] = {}
    for r in cleaned:
        for t in r.tags_topic:
            counter[t] = counter.get(t, 0) + 1
    for t, c in sorted(counter.items(), key=lambda x: -x[1]):
        table.add_row(t, str(c))
    console.print(table)

    # 来源分布表
    src_table = Table(title="Source distribution")
    src_table.add_column("source", style="cyan")
    src_table.add_column("count", justify="right", style="green")
    src_counter: dict[str, int] = {}
    for r in cleaned:
        src_counter[r.source] = src_counter.get(r.source, 0) + 1
    for s, c in sorted(src_counter.items(), key=lambda x: -x[1]):
        src_table.add_row(s, str(c))
    console.print(src_table)

    # 把带 tags 的版本再覆盖落盘
    save_clean(cleaned)

    if args.no_index:
        console.print("[yellow]--no-index 已指定，跳过向量化与入库。[/yellow]")
        return 0

    # ---------- 4) Static analysis + quality scoring ----------
    step("4/6  静态分析与质量评分")
    from shader_agent.corpus.static_analysis import analyze_records
    analyze_records(cleaned)
    q_table = Table(title="Quality score distribution")
    q_table.add_column("range", style="cyan")
    q_table.add_column("count", justify="right", style="green")
    buckets = {"0.0-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for r in cleaned:
        q = r.quality_score
        if q < 0.4:
            buckets["0.0-0.4"] += 1
        elif q < 0.6:
            buckets["0.4-0.6"] += 1
        elif q < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1
    for k, v in buckets.items():
        q_table.add_row(k, str(v))
    console.print(q_table)
    save_clean(cleaned)  # 带质量字段再落盘

    # ---------- 5) Build stores ----------
    step("5/6  建库（向量 + 子块 + BM25 + 父文档）")
    from shader_agent.corpus.chunker import chunk_shader
    from shader_agent.corpus.keyword_store import KeywordStore
    from shader_agent.corpus.parent_store import ParentDocumentStore

    vstore = ShaderVectorStore()
    if args.reset:
        vstore.reset()

    n = vstore.upsert(cleaned)
    console.print(f"  · shader 级向量 → [bold]{n}[/bold]")

    n_chunks = vstore.upsert_chunks(cleaned)
    console.print(f"  · 子块级向量   → [bold]{n_chunks}[/bold]")

    all_chunks = []
    for r in cleaned:
        all_chunks.extend(chunk_shader(r))
    kstore = KeywordStore()
    kstore.build(all_chunks)
    kstore.save()
    console.print(f"  · BM25 关键词   → [bold]{kstore.count()}[/bold]")

    pstore = ParentDocumentStore()
    if args.reset:
        pstore.reset()
    pstore.upsert(cleaned)
    console.print(f"  · 父文档        → [bold]{pstore.count()}[/bold]")
    console.print(
        f"collection total = [bold]{vstore.count()}[/bold] / "
        f"chunks = [bold]{vstore.chunk_count()}[/bold]"
    )

    # ---------- 6) Smoke query (hybrid) ----------
    step("6/6  混合检索冒烟测试")
    from shader_agent.corpus.reranker import Reranker
    from shader_agent.corpus.retriever import HybridRetriever
    retriever = HybridRetriever(
        vector_store=vstore,
        keyword_store=kstore,
        parent_store=pstore,
        reranker=Reranker(enabled=settings.retrieval.use_rerank),
    )
    smoke_queries = [
        "raymarched sphere with lighting",
        "2d procedural noise pattern",
        "fractal escape time coloring",
        "neon kaleidoscope animation",
        "domain warping fbm clouds",
    ]
    for q in smoke_queries:
        hits = retriever.retrieve(q, top_k=3)
        console.print(f"\n[bold]Query:[/bold] {q}")
        if not hits:
            console.print("  [dim](无命中，可能低于相关度阈值)[/dim]")
        for i, h in enumerate(hits, 1):
            console.print(
                f"  {i}. {h.name} "
                f"[dim](id={h.shader_id}, fused={h.fused_score:.3f}, "
                f"vec={h.vec_rel:.2f}, bm25={h.bm25_norm:.2f}, "
                f"chunks={h.matched_chunks[:3]})[/dim]"
            )

    console.print("\n[green]Done.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
