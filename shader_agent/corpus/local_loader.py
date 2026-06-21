"""从本地目录批量导入 GLSL 文件作为 ShaderRecord。

用途：
1. 用户没有 Shadertoy API key、抓取又不稳定时，可以手工去
   github.com/ 一些开放 shader 集合（如 The Book of Shaders / glsl-sandbox-mirror /
   shader-park-presets 等，遵守各自 LICENSE）下载 .glsl 文件，
   放到 `data/external_shaders/` 下，本模块一行命令导入。
2. 同名 sidecar `.meta.json` 提供 name / description / tags_raw / author，可选。
3. 自动剥 Shadertoy `// Created by ...` 头注释里的元信息回填到 sidecar。

文件命名约定：
    data/external_shaders/
    ├── my_first.glsl
    ├── my_first.meta.json      (可选: {"name":"...","description":"...","tags_raw":[...],"author":"..."})
    ├── another.frag            (.glsl/.frag/.glslfs 均接受)
    └── another.meta.json

License 合规检查（最小可行）：
- sidecar 里若声明 `license`，会原样写入 ShaderRecord.description 末尾，便于审查；
- 若主代码顶部 30 行内匹配 "All Rights Reserved" / "Proprietary" 等关键词，
  默认打 warning 并跳过；通过 `accept_restricted=True` 强制导入。

不做的事：
- 不下载、不联网、不改写 GLSL 内容；
- 不做去重（去重交给 cleaner.clean_records 统一处理）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from shader_agent.corpus.models import RenderPass, ShaderRecord
from shader_agent.utils.logger import logger


# 支持的 GLSL 后缀
_GLSL_EXTS = {".glsl", ".frag", ".glslfs", ".fs", ".fsh"}

# 限制性 license 关键词（保守扫描，避免误导入）
_RESTRICTIVE_LICENSE_RE = re.compile(
    r"all\s*rights\s*reserved|proprietary|do\s*not\s*redistribute|"
    r"no\s*permission|confidential",
    re.IGNORECASE,
)

# Shadertoy 顶部典型注释：// Created by <author> in <year>
_SHADERTOY_HEADER_RE = re.compile(
    r"//\s*(?:Created\s+by|Author[:]?\s*|@author\s+)([^\n]+)",
    re.IGNORECASE,
)


def _read_sidecar(meta_path: Path) -> dict:
    """读取 .meta.json sidecar。文件不存在 → 空 dict。"""
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[local_loader] 解析 {meta_path.name} 失败: {e}")
        return {}


def _make_id(path: Path, external_root: Path | None = None) -> str:
    """以稳定且文件系统友好的方式生成 shader_id。
    用 external_shaders 下的相对路径前缀防冲突。
    例：data/external_shaders/PixelFlow/data/shader.frag → local_pixelflow_data_shader
    """
    if external_root is not None:
        try:
            rel = path.relative_to(external_root)
        except ValueError:
            rel = path
        parts = list(rel.parent.parts) + [rel.stem]
    else:
        parts = [path.stem]
    clean = [re.sub(r"[^a-zA-Z0-9]+", "_", p).strip("_").lower() for p in parts]
    clean = [c for c in clean if c]
    return f"local_{'_'.join(clean) or 'unnamed'}"


def _has_restrictive_license(code: str) -> bool:
    head = "\n".join(code.splitlines()[:30])
    return bool(_RESTRICTIVE_LICENSE_RE.search(head))


def _extract_author_from_header(code: str) -> str:
    m = _SHADERTOY_HEADER_RE.search("\n".join(code.splitlines()[:20]))
    if m:
        return m.group(1).strip().rstrip(".")
    return ""


def load_local_shader(path: Path, *, accept_restricted: bool = False, external_root: Path | None = None) -> ShaderRecord | None:
    """加载单个 .glsl 文件 + 可选 sidecar，返回 ShaderRecord。

    Args:
        path: GLSL 文件路径。
        accept_restricted: 若为 True，含 "All Rights Reserved" 字样的也照单全收（自负其责）。

    Returns:
        ShaderRecord 或 None（解析失败 / 含限制性 license 且未强制接受）。
    """
    try:
        code = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"[local_loader] 读取 {path} 失败: {e}")
        return None

    if not code.strip():
        return None

    if _has_restrictive_license(code) and not accept_restricted:
        logger.warning(
            f"[local_loader] 跳过 {path.name}: 顶部注释含限制性 license 关键词；"
            "若确认可商用/二次分发，传 accept_restricted=True"
        )
        return None

    meta = _read_sidecar(path.with_suffix(".meta.json"))
    shader_id = str(meta.get("shader_id") or _make_id(path, external_root))
    name = str(meta.get("name") or path.stem.replace("_", " ").title())
    description = str(meta.get("description") or "")
    author = str(meta.get("author") or _extract_author_from_header(code) or "local")
    tags_raw = list(meta.get("tags_raw") or [])
    license_text = str(meta.get("license") or "")

    # 把 license 摘要追加到 description 末尾，便于事后审查
    if license_text:
        description = (description + f"\n[license] {license_text}").strip()

    rec = ShaderRecord(
        shader_id=shader_id,
        name=name,
        username=author,
        description=description,
        likes=int(meta.get("likes") or 100),  # 默认 100 给个高分，避免 cleaner 过滤
        viewed=int(meta.get("viewed") or 0),
        tags_raw=tags_raw,
        source="local",
        code_image=code,
        passes=[RenderPass(name="Image", type="image", code=code)],
    )
    return rec


def load_local_dir(
    src_dir: Path,
    *,
    accept_restricted: bool = False,
    recursive: bool = True,
) -> list[ShaderRecord]:
    """扫描目录里所有 GLSL 文件并导入。

    Args:
        src_dir: 源目录。
        accept_restricted: 是否接受限制性 license 文件。
        recursive: 是否递归子目录。

    Returns:
        ShaderRecord 列表（可能为空）。
    """
    src_dir = Path(src_dir)
    if not src_dir.exists():
        logger.warning(f"[local_loader] 目录不存在: {src_dir}")
        return []

    iterator = src_dir.rglob("*") if recursive else src_dir.iterdir()
    files = sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in _GLSL_EXTS
    )

    out: list[ShaderRecord] = []
    for f in files:
        rec = load_local_shader(f, accept_restricted=accept_restricted, external_root=src_dir)
        if rec is not None:
            out.append(rec)
    logger.info(f"[local_loader] {src_dir}: 扫描 {len(files)} 文件 → 导入 {len(out)} 条")
    return out


def default_local_dir(project_root: Path) -> Path:
    """约定俗成的本地 shader 目录：data/external_shaders/"""
    return project_root / "data" / "external_shaders"
