"""语料库分布分析（纯标准库，无需 chromadb / torch）。

用途：在重建前后快速比对主题/来源/质量/生成器占比，验证是否均衡。

    python -m scripts.analyze_distribution                      # 默认看 settings 的 clean 目录
    python -m scripts.analyze_distribution --dir data/clean_rebuilt
    python -m scripts.analyze_distribution --dir A --compare B  # 并排对比两个目录
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load_dir(d: Path) -> list[dict]:
    out = []
    for f in sorted(Path(d).glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


def gini(counts: list[int]) -> float:
    """主题计数的基尼系数（0=完全均衡，1=完全集中），衡量不均衡程度。"""
    if not counts:
        return 0.0
    xs = sorted(counts)
    n = len(xs)
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    total = sum(xs)
    if total == 0:
        return 0.0
    return (2 * cum) / (n * total) - (n + 1) / n


def summarize(docs: list[dict]) -> dict:
    topics = Counter()
    sources = Counter()
    n = len(docs)
    compile_ok = render_ok = generators = reference_only = 0
    qsum = 0.0
    multi_tag = 0
    for d in docs:
        tt = d.get("tags_topic", []) or []
        topics.update(tt)
        if len([t for t in tt if t != "uncategorized"]) >= 2:
            multi_tag += 1
        sources[d.get("source", "?")] += 1
        compile_ok += 1 if d.get("compile_ok") else 0
        render_ok += 1 if d.get("render_ok") else 0
        generators += 1 if d.get("is_generator", True) else 0
        reference_only += 1 if d.get("reference_only", False) else 0
        qsum += float(d.get("quality_score", 0.0) or 0.0)
    top_counts = [c for _, c in topics.most_common()]
    return {
        "n": n,
        "topics": topics,
        "sources": sources,
        "compile_ok": compile_ok,
        "render_ok": render_ok,
        "generators": generators,
        "reference_only": reference_only,
        "avg_quality": (qsum / n) if n else 0.0,
        "multi_tag": multi_tag,
        "top_share": (top_counts[0] / sum(top_counts)) if top_counts else 0.0,
        "gini": gini(top_counts),
    }


def print_summary(title: str, s: dict) -> None:
    print(f"\n================ {title} ================")
    print(f"records            : {s['n']}")
    print(f"avg quality_score  : {s['avg_quality']:.3f}")
    print(f"compile_ok         : {s['compile_ok']}/{s['n']} "
          f"({100*s['compile_ok']/max(s['n'],1):.0f}%)")
    print(f"render_ok          : {s['render_ok']}/{s['n']}")
    print(f"generators / refs  : {s['generators']} / {s['reference_only']}")
    print(f"multi-tag records  : {s['multi_tag']}/{s['n']} "
          f"({100*s['multi_tag']/max(s['n'],1):.0f}%)")
    print(f"largest-topic share: {100*s['top_share']:.1f}%  "
          f"(越低越均衡)")
    print(f"topic Gini         : {s['gini']:.3f}  (0=完全均衡, 1=完全集中)")
    print("\nsource distribution:")
    for k, v in s["sources"].most_common():
        print(f"  {k:14s} {v}")
    print("\ntopic distribution:")
    total_tags = sum(s["topics"].values()) or 1
    for k, v in s["topics"].most_common():
        bar = "#" * max(1, int(40 * v / max(s['topics'].values())))
        print(f"  {k:16s} {v:4d}  {100*v/total_tags:5.1f}%  {bar}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="")
    ap.add_argument("--compare", type=str, default="")
    args = ap.parse_args()

    if args.dir:
        d = Path(args.dir)
    else:
        try:
            from shader_agent.config.settings import settings
            d = settings.corpus_clean_dir
        except Exception:
            d = Path("data/clean")

    docs = load_dir(d)
    print_summary(str(d), summarize(docs))
    if args.compare:
        docs2 = load_dir(Path(args.compare))
        print_summary(args.compare, summarize(docs2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
