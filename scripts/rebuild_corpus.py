"""语料库重建主入口（v2：旧库 + ISF + shaders21k → 重打标 → 重评分 → 重建库）。

解决两件事：
  1. 数据过少：并入 ISF（327）与 shaders21k 内嵌示例，规模 ~翻倍；
  2. 分布不均衡：用 v2 tagger（分类映射优先、去掉 2d-pattern 兜底）重打标，
     用格式自适应的 v2 质量评分重评分，可选主题配额封顶。

数据源
------
  --from-clean      DIR   旧 clean/*.json（默认 settings 的 clean 目录）
  --from-isf        DIR   ISF 目录（含 *.fs），如 plan 解压后的 .../ISF
  --from-s21k-repo  DIR   shaders21k-main 仓库根（抽取内嵌 twigl 示例）
  --from-s21k-data  DIR   已下载的 shaders21k Shadertoy JSON 目录（可选）

用法
----
  # 只看重建后的分布，不建库、不依赖 chromadb/torch（离线、秒级）
  python -m scripts.rebuild_corpus --from-clean data/clean \\
      --from-isf data/plan/ISF-Files-master/ISF \\
      --from-s21k-repo data/plan/shaders21k-main \\
      --out data/clean_rebuilt --dry-run

  # 全量重建（写向量库 + 子块 + BM25 + 父文档），需安装 requirements
  python -m scripts.rebuild_corpus --from-clean data/clean \\
      --from-isf data/plan/ISF-Files-master/ISF \\
      --from-s21k-repo data/plan/shaders21k-main \\
      --out data/clean_rebuilt --reset

  # 极端不均衡时启用主题配额封顶
  python -m scripts.rebuild_corpus ... --max-per-topic 150
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--from-clean", type=str, default="")
    p.add_argument("--from-isf", type=str, default="")
    p.add_argument("--from-s21k-repo", type=str, default="")
    p.add_argument("--from-s21k-data", type=str, default="")
    p.add_argument("--out", type=str, default="data/clean_rebuilt",
                   help="重建后的 clean JSON 落盘目录")
    p.add_argument("--max-per-topic", type=int, default=0,
                   help=">0 时启用主题配额封顶（默认 0=不封顶，靠 tagger 自然均衡）")
    p.add_argument("--enable-llm-tagging", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="只重打标/重评分/落盘并打印分布，不建向量库（离线可跑）")
    p.add_argument("--reset", action="store_true",
                   help="建库前清空 collection 与父文档表")
    p.add_argument("--no-index", action="store_true",
                   help="同 --dry-run 但仍落盘（保留以兼容）")
    return p.parse_args()


def _load_clean_dir(d: Path) -> list:
    from shader_agent.corpus.models import ShaderRecord
    from shader_agent.utils.logger import logger
    recs = []
    for f in sorted(Path(d).glob("*.json")):
        try:
            recs.append(ShaderRecord.load_json(f))
        except Exception as e:
            logger.warning(f"[rebuild] 跳过 {f.name}: {e}")
    logger.info(f"[rebuild] 旧 clean 载入 {len(recs)} 条 <- {d}")
    return recs


def collect_records(args: argparse.Namespace) -> list:
    from shader_agent.utils.logger import logger
    records = []

    if args.from_clean:
        records.extend(_load_clean_dir(Path(args.from_clean)))

    if args.from_isf:
        from shader_agent.corpus.isf_loader import load_isf_dir
        records.extend(load_isf_dir(Path(args.from_isf)))

    if args.from_s21k_repo or args.from_s21k_data:
        from shader_agent.corpus.shaders21k_loader import load_shaders21k
        records.extend(load_shaders21k(
            repo_root=Path(args.from_s21k_repo) if args.from_s21k_repo else None,
            download_dir=Path(args.from_s21k_data) if args.from_s21k_data else None,
        ))

    logger.info(f"[rebuild] 合并原始记录 {len(records)} 条（去重前）")
    return records


def main() -> int:
    args = parse_args()
    from shader_agent.corpus.cleaner import balance_by_topic, clean_records, save_clean
    from shader_agent.corpus.static_analysis import analyze_records
    from shader_agent.corpus.tagger import tag_records
    from shader_agent.config.settings import settings
    from scripts.analyze_distribution import print_summary, summarize

    out_dir = Path(args.out)

    # ---- 重建前快照（如果有旧 clean）----
    if args.from_clean:
        before = [r.model_dump() for r in _load_clean_dir(Path(args.from_clean))]
        print_summary("BEFORE (旧 clean)", summarize(before))

    # ---- 1) 收集 ----
    raw = collect_records(args)
    if not raw:
        print("没有任何记录。请检查 --from-* 路径。")
        return 1

    # ---- 2) 清洗去重（保留外部资源型为参考）----
    cleaned = clean_records(raw, keep_external_as_reference=True)

    # ---- 3) 重打标（v2 tagger）----
    use_llm = args.enable_llm_tagging or settings.corpus.enable_llm_tagging
    tag_records(cleaned, use_llm=use_llm)

    # ---- 4) 静态分析 + 重评分（v2 质量）----
    analyze_records(cleaned)

    # ---- 5) 可选主题配额封顶 ----
    if args.max_per_topic and args.max_per_topic > 0:
        cleaned = balance_by_topic(cleaned, max_per_topic=args.max_per_topic)

    # ---- 6) 落盘 ----
    save_clean(cleaned, out_dir)
    after = [r.model_dump() for r in cleaned]
    print_summary(f"AFTER (重建 -> {out_dir})", summarize(after))

    if args.dry_run or args.no_index:
        print("\n[dry-run] 已落盘重建 clean，未建向量库。"
              "安装 requirements 后去掉 --dry-run 即可全量建库。")
        return 0

    # ---- 7) 建库（需要 chromadb / sentence-transformers）----
    print("\n开始建库（向量 + 子块 + BM25 + 父文档）...")
    from shader_agent.corpus.chunker import chunk_shader
    from shader_agent.corpus.keyword_store import KeywordStore
    from shader_agent.corpus.parent_store import ParentDocumentStore
    from shader_agent.corpus.vector_store import ShaderVectorStore

    vstore = ShaderVectorStore()
    if args.reset:
        vstore.reset()
    # v2：不再做"shader 级"重复嵌入（hybrid 路径只用子块），仅建子块级，省一半算力
    n_chunks = vstore.upsert_chunks(cleaned)
    print(f"  · 子块级向量 → {n_chunks}")

    all_chunks = []
    for r in cleaned:
        all_chunks.extend(chunk_shader(r))
    kstore = KeywordStore()
    kstore.build(all_chunks)
    kstore.save()
    print(f"  · BM25 关键词 → {kstore.count()}")

    pstore = ParentDocumentStore()
    if args.reset:
        pstore.reset()
    pstore.upsert(cleaned)
    print(f"  · 父文档 → {pstore.count()}")
    print(f"chunks total = {vstore.chunk_count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
