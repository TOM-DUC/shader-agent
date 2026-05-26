"""徽章 / CSS 小工具，让 UI 信息密度更高。"""
from __future__ import annotations


CUSTOM_CSS = """
.shader-status-bar { font-size: 12px; color: #666; }
.shader-status-bar code { background: #f3f3f5; padding: 1px 4px; border-radius: 3px; }
.shader-badge { display: inline-block; padding: 2px 8px; margin-right: 6px;
                border-radius: 10px; font-size: 12px; font-weight: 600; }
.shader-badge-ok    { background: #e6f7ec; color: #1c7c3e; }
.shader-badge-fail  { background: #fdeaea; color: #b3261e; }
.shader-badge-warn  { background: #fff7e0; color: #876500; }
.shader-badge-info  { background: #eef2ff; color: #283cb6; }
.shader-diag       { font-family: monospace; font-size: 11px;
                     background: #fafafa; padding: 6px 10px; border-radius: 4px;
                     border: 1px solid #eee; color: #555; }
.shader-error-block { background: #fdeaea; border-left: 3px solid #b3261e;
                      padding: 8px 12px; font-family: monospace; font-size: 12px;
                      white-space: pre-wrap; color: #5b1c17; border-radius: 4px; }
"""


def badge(text: str, kind: str = "info") -> str:
    """生成一个 HTML 徽章。kind ∈ {ok, fail, warn, info}。"""
    return f'<span class="shader-badge shader-badge-{kind}">{text}</span>'


def status_html(
    *,
    backend_label: str,
    vstore_label: str,
    elapsed_ms: float,
    iterations: int = 0,
    compile_ok: bool | None = None,
) -> str:
    """组合一行状态徽章 HTML。"""
    parts: list[str] = []
    parts.append(badge(f"渲染后端: {backend_label}", "info"))
    parts.append(badge(f"向量库: {vstore_label}", "info"))
    if compile_ok is True:
        parts.append(badge("编译 OK", "ok"))
    elif compile_ok is False:
        parts.append(badge("编译失败", "fail"))
    if iterations:
        parts.append(badge(f"迭代 {iterations} 轮", "warn" if iterations > 1 else "info"))
    parts.append(badge(f"耗时 {elapsed_ms/1000:.2f}s", "info"))
    return '<div class="shader-status-bar">' + " ".join(parts) + "</div>"


def error_block(msg: str) -> str:
    """红色错误块。"""
    if not msg:
        return ""
    return f'<div class="shader-error-block">{msg}</div>'


def diagnostics_html(items: list[str]) -> str:
    if not items:
        return ""
    body = "<br>".join(f"· {s}" for s in items if s)
    return f'<div class="shader-diag">{body}</div>'


def running_html(msg: str = "正在处理，请稍候…") -> str:
    """运行中状态条（配合 Gradio 流式回调，先给用户即时反馈）。"""
    return (
        '<div class="shader-status-bar">'
        + badge("⏳ " + msg, "info")
        + "</div>"
    )
