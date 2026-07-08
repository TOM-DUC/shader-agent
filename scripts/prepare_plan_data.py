"""把 plan/*.zip 解压到 data/plan/，供 rebuild_corpus 使用。

    python -m scripts.prepare_plan_data

解压后目录：
    data/plan/ISF-Files-master/ISF/*.fs
    data/plan/shaders21k-main/...
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from shader_agent.config.settings import PROJECT_ROOT


def main() -> int:
    plan_dir = PROJECT_ROOT / "plan"
    out = PROJECT_ROOT / "data" / "plan"
    out.mkdir(parents=True, exist_ok=True)
    zips = sorted(plan_dir.glob("*.zip"))
    if not zips:
        print(f"未在 {plan_dir} 找到 zip。")
        return 1
    for z in zips:
        target = out / z.stem
        print(f"解压 {z.name} -> {target}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(target)
    # 给出 ISF 的真实路径提示
    isf_candidates = list(out.rglob("ISF/*.fs"))
    if isf_candidates:
        print(f"\nISF 目录: {isf_candidates[0].parent}")
    print("完成。现在可运行 scripts.rebuild_corpus。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
