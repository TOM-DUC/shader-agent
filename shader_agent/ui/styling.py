"""UI 视觉设计层（taste-skill 重设计版 · 修订 v2）。

修订要点（相对上一版）
======================
1. **Hero**：从"全左对齐 + 横向 trust strip"改为左文右标二栏栅格。横排的
   Analyzer/Generator/Remixer tag 条违反 taste-skill 的 hero 文本元素铁则，
   现挪入右栏作为带序号的能力清单，整页节奏不再单调。
2. **Section 标题**：之前所有 eyebrow 都靠左，整页"标题全部偏左"。改成
   "细线 — 居中标签 — 细线"的规则线形式，加入水平方向的视觉变化。
3. **锁定布局比例**：右侧预览容器与状态条给定固定 min-height；状态条初始
   渲染"待命"芯片而非空 HTML，让运行前/运行后骨架一致。
4. **锁定按钮尺寸**：主按钮 `.ts-action-btn` 固定宽高，不再被列宽流动牵动。
5. **修下拉被遮挡**：Accordion / Tab / Row / Column / Form / Block 等所有
   可能的祖先容器全部 `overflow: visible`，Dropdown 菜单 z-index 提到 9999，
   焦点态容器自身 z-index 上调，避免与下方控件叠层冲突。
6. **修圆角边角字裁切**：CodeMirror、textarea、input 一律加大水平 padding；
   hero 与 card 内边距加大，所有字符都离开圆角影响区。

其余设计令牌（暖中性纸感底色 + 单一赤陶强调色 + 三组字体）沿用第一版。
所有公共函数（hero_html / section_title / status_html / running_html /
error_block / diagnostics_html / preview_placeholder / badge）签名保持
不变，新增一个 idle_status_html() 用于初始化"待命"状态条。
"""
from __future__ import annotations

import gradio as gr


# =====================================================================
# 主题
# =====================================================================

def build_theme() -> "gr.themes.Base":
    """Editorial 主题：暖中性 + 赤陶强调 + 有性格的字体。

    用 gr.themes.Base 而非 Soft，避免 Soft 自带的圆润蓝紫感；
    主色用 stone 中性灰打底，强调色靠 CSS 注入（Gradio 调色板里没有赤陶）。
    """
    return gr.themes.Base(
        primary_hue=gr.themes.colors.stone,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_lg,
        text_size=gr.themes.sizes.text_md,
        font=[gr.themes.GoogleFont("Spline Sans"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#FBF9F5",
        body_background_fill_dark="#1A1714",
        body_text_color="#2A2521",
        background_fill_primary="#FFFFFF",
        background_fill_secondary="#F4F1EA",
        block_background_fill="#FFFFFF",
        block_border_width="1px",
        block_border_color="#E7E1D6",
        block_radius="16px",
        block_shadow="0 1px 2px rgba(72,54,38,.04), 0 8px 24px -16px rgba(72,54,38,.18)",
        block_label_text_color="#8A7E6E",
        block_label_text_weight="600",
        block_title_text_color="#2A2521",
        block_title_text_weight="600",
        button_primary_background_fill="#C45A3B",
        button_primary_background_fill_hover="#A8472C",
        button_primary_text_color="#FFFFFF",
        button_primary_shadow="0 2px 8px -2px rgba(196,90,59,.5)",
        button_secondary_background_fill="#F4F1EA",
        button_secondary_background_fill_hover="#EBE6DB",
        button_secondary_text_color="#3A332C",
        button_large_radius="999px",
        button_small_radius="999px",
        input_background_fill="#FCFBF8",
        input_border_color="#E2DCD0",
        input_border_color_focus="#C45A3B",
        slider_color="#C45A3B",
    )


# =====================================================================
# CSS：设计令牌 + 组件样式
# =====================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Spline+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --ts-paper:    #FBF9F5;
  --ts-card:     #FFFFFF;
  --ts-ink:      #2A2521;
  --ts-ink-soft: #6B6155;
  --ts-faint:    #8A7E6E;
  --ts-line:     #E7E1D6;
  --ts-line-2:   #EFEBE2;
  --ts-accent:   #C45A3B;     /* 赤陶，唯一强调色 */
  --ts-accent-d: #A8472C;
  --ts-sage:     #5C6B5A;     /* 仅用于"成功"语义 */
  --ts-clay:     #B08968;
  --ts-shadow:   72,54,38;    /* 阴影色相（暖棕），不是死黑 */
  --ts-ease:     cubic-bezier(0.32, 0.72, 0, 1);
  --font-display:'Space Grotesk', system-ui, sans-serif;

  /* 布局常量：右列与状态条最小尺寸，锁定运行前/后骨架一致 */
  --ts-preview-h: 380px;
  --ts-col-min:   520px;
}

/* 全局字体兜底 */
.gradio-container { font-family: 'Spline Sans', system-ui, sans-serif !important; }

/* 细微纸纹噪点：固定层，不拦截事件，避免数字平面感 */
.gradio-container::before {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  opacity: .025;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* =====================================================
   Hero · 左文右标二栏栅格（替代原 trust strip）
   ===================================================== */
.ts-hero {
  position: relative;
  background: var(--ts-card);
  border: 1px solid var(--ts-line);
  border-radius: 22px;
  padding: 40px 48px;
  margin-bottom: 14px;
  box-shadow: 0 1px 2px rgba(var(--ts-shadow),.04), 0 18px 40px -28px rgba(var(--ts-shadow),.4);
  overflow: hidden;
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(300px, 1fr);
  gap: 56px;
  align-items: center;
}
/* 右上一抹极淡赤陶径向光，替代彩虹渐变 */
.ts-hero::after {
  content: ""; position: absolute; right: -140px; top: -160px;
  width: 380px; height: 380px; border-radius: 50%;
  background: radial-gradient(circle, rgba(196,90,59,.10), transparent 70%);
  pointer-events: none;
}

.ts-hero-left { min-width: 0; }
.ts-eyebrow {
  font-family: var(--font-display);
  font-size: 11px; letter-spacing: .24em; text-transform: uppercase;
  font-weight: 600; color: var(--ts-accent); margin: 0 0 14px;
}
.ts-hero h1 {
  font-family: var(--font-display); font-weight: 700;
  font-size: 38px; line-height: 1.12; letter-spacing: -.02em;
  color: var(--ts-ink); margin: 0; text-wrap: balance;
}
.ts-hero h1 .ts-mark { color: var(--ts-accent); }
.ts-hero-left p {
  margin: 0; max-width: 50ch; color: var(--ts-ink-soft);
  font-size: 15px; line-height: 1.7;
}

/* 右栏：能力清单（替代横向 tag 条） */
.ts-hero-right {
  border-left: 1px solid var(--ts-line);
  padding-left: 32px;
  display: flex; flex-direction: column; gap: 14px;
  position: relative; z-index: 1;
}
.ts-cap {
  display: grid; grid-template-columns: 28px 1fr; gap: 14px;
  align-items: baseline;
}
.ts-cap-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; color: var(--ts-accent);
  font-variant-numeric: tabular-nums; letter-spacing: .04em;
}
.ts-cap-body {
  font-size: 13.5px; color: var(--ts-ink); line-height: 1.55;
}
.ts-cap-body b {
  font-family: var(--font-display); font-weight: 600;
  letter-spacing: -.005em; margin-right: 6px;
}
.ts-cap-body .ts-mute {
  color: var(--ts-faint); font-size: 12.5px;
}

/* 窄屏：hero 改为单列 */
@media (max-width: 900px) {
  .ts-hero { grid-template-columns: 1fr; gap: 28px; padding: 32px 28px; }
  .ts-hero h1 { font-size: 32px; }
  .ts-hero-right { border-left: none; padding-left: 0; padding-top: 20px;
    border-top: 1px solid var(--ts-line); }
}

/* =====================================================
   Section 标题：细线 — 居中标签 — 细线
   解决"通篇标题全部偏左"
   ===================================================== */
.ts-section {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center; gap: 18px;
  margin: 2px 2px 14px;
}
.ts-section::before,
.ts-section::after {
  content: ""; display: block; height: 1px;
  background: var(--ts-line);
}
.ts-section .ts-section-label {
  font-family: var(--font-display);
  font-size: 11px; font-weight: 600;
  letter-spacing: .22em; text-transform: uppercase;
  color: var(--ts-faint);
  white-space: nowrap;
}
.ts-section .ts-section-label .ts-section-dot {
  display: inline-block; width: 5px; height: 5px;
  border-radius: 50%; background: var(--ts-accent);
  margin: 0 10px 2px 0; vertical-align: middle;
}

/* =====================================================
   状态徽章（chip）：方角、描边、单色，无彩虹
   ===================================================== */
.ts-status {
  display: flex; flex-wrap: wrap; gap: 7px; align-items: center;
  padding: 0; min-height: 8px;
}
.ts-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 2px 8px; border-radius: 8px;
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px; font-weight: 500;
  background: var(--ts-card); border: 1px solid var(--ts-line); color: var(--ts-ink-soft);
  font-variant-numeric: tabular-nums;
  line-height: 1;
}
.ts-chip .ts-led { width: 6px; height: 6px; border-radius: 50%; background: var(--ts-faint); }
.ts-chip-ok    { border-color: #CBD6C8; color: var(--ts-sage);    background: #F3F6F1; }
.ts-chip-ok    .ts-led { background: var(--ts-sage); }
.ts-chip-fail  { border-color: #E6C9BF; color: var(--ts-accent-d); background: #FBF0EC; }
.ts-chip-fail  .ts-led { background: var(--ts-accent); }
.ts-chip-accent{ border-color: #E6C9BF; color: var(--ts-accent-d); }
.ts-chip-accent .ts-led { background: var(--ts-accent); }
.ts-chip-idle  { color: var(--ts-faint); background: #F7F4ED; border-style: dashed; }
.ts-chip-idle  .ts-led { background: var(--ts-faint); }

@keyframes ts-breathe { 0%,100%{opacity:.45;} 50%{opacity:1;} }
.ts-chip-run { border-color: #E6C9BF; color: var(--ts-accent-d); background: #FBF0EC; }
.ts-chip-run .ts-led { background: var(--ts-accent); animation: ts-breathe 1.1s var(--ts-ease) infinite; }

/* =====================================================
   错误 / 诊断块
   ===================================================== */
.ts-error {
  background: #FBF0EC; border-left: 3px solid var(--ts-accent);
  padding: 12px 16px; font-family: 'JetBrains Mono', monospace; font-size: 12px;
  white-space: pre-wrap; color: #6E3322; border-radius: 10px; margin-top: 10px;
  line-height: 1.55;
}
.ts-diag {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  background: #F4F1EA; padding: 10px 14px; border-radius: 10px;
  border: 1px solid var(--ts-line-2); color: var(--ts-faint); margin-top: 8px;
  line-height: 1.6;
}

/* =====================================================
   预览容器：Double-Bezel + 固定 min-height（锁列高）
   ===================================================== */
.ts-preview {
  background: var(--ts-card); border: 1px solid var(--ts-line);
  border-radius: 18px; padding: 8px;
  box-shadow: inset 0 1px 1px rgba(255,255,255,.6),
              0 12px 30px -22px rgba(var(--ts-shadow),.5);
  min-height: var(--ts-preview-h);
}
.ts-preview iframe, .ts-preview canvas {
  border-radius: 12px !important; display: block;
}
.ts-preview-empty {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center; text-align: center;
  min-height: calc(var(--ts-preview-h) - 16px);
  border-radius: 12px; color: #E8E1D4;
  background: radial-gradient(120% 80% at 50% 0%, #2B2520 0%, #1C1814 60%, #161310 100%);
  border: 1px solid #2E2620;
  letter-spacing: .01em; padding: 24px 28px; line-height: 1.6;
  gap: 10px;
}
.ts-preview-empty .ts-ph-mark {
  color: var(--ts-clay); font-family: var(--font-display); font-weight: 600;
  font-size: 24px; letter-spacing: .02em;
}
.ts-preview-empty .ts-ph-msg {
  font-size: 13.5px; max-width: 32ch;
}
.ts-preview-empty .ts-ph-hint {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: #8B7F70; letter-spacing: .06em; margin-top: 4px;
}

/* =====================================================
   主操作按钮：锁定尺寸，不随列宽流动
   ===================================================== */
.gradio-container .ts-action-btn,
.gradio-container .ts-action-btn button {
  min-width: 200px !important;
  max-width: 260px !important;
  height: 48px !important;
  flex: 0 0 auto !important;
  align-self: flex-start !important;
  letter-spacing: .02em;
  font-weight: 600 !important;
  white-space: nowrap !important;
}
/* 物理按压：仅 transform，不改尺寸 */
.gradio-container button.primary,
.gradio-container button.lg {
  transition: transform .22s var(--ts-ease),
              background .22s var(--ts-ease),
              box-shadow .22s var(--ts-ease) !important;
}
.gradio-container button.primary:active,
.gradio-container button.lg:active { transform: scale(.97); }

/* =====================================================
   Tab：下划线式，去掉方块底色
   ===================================================== */
.tabs > .tab-nav { border-bottom: 1px solid var(--ts-line) !important; gap: 4px; }
.tabs > .tab-nav button {
  font-family: var(--font-display) !important; font-weight: 600 !important;
  font-size: 14px !important; color: var(--ts-faint) !important;
  background: transparent !important; border: none !important;
  border-bottom: 2px solid transparent !important; border-radius: 0 !important;
  padding: 11px 18px !important;
  transition: color .25s var(--ts-ease), border-color .25s var(--ts-ease) !important;
}
.tabs > .tab-nav button.selected {
  color: var(--ts-ink) !important; border-bottom-color: var(--ts-accent) !important;
}

/* 缩小 tab 标题与内容的间距 */
.gradio-container .tabitem {
  padding-top: 2px !important;
}

/* =====================================================
   修复圆角文本框边角字裁切
   关键：CodeMirror、textarea、input 一律加大水平 padding
   ===================================================== */
.gradio-container .cm-editor {
  font-size: 13px !important; border-radius: 12px !important;
}
.gradio-container .cm-editor .cm-scroller { padding: 10px 4px !important; }
.gradio-container .cm-editor .cm-line { padding-left: 8px !important; padding-right: 8px !important; }
.gradio-container .cm-editor .cm-gutters { padding-left: 4px !important; }

.gradio-container textarea {
  padding: 12px 14px !important;
  border-radius: 12px !important;
  line-height: 1.6 !important;
}
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container input:not([type]) {
  padding: 10px 14px !important;
  border-radius: 10px !important;
}

/* Code 容器外圆角统一 */
.ts-tight .cm-editor,
.ts-tight .gr-code,
.ts-tight .code_wrap { border-radius: 12px !important; }

/* =====================================================
   列布局锁底线（避免运行前后高度跳变）
   ===================================================== */
.ts-tab-row { align-items: flex-start !important; }
.ts-col-input,
.ts-col-output {
  min-height: var(--ts-col-min);
}
/* 左列把"按钮"按 ts-action-btn 单独占行，不挤其他控件 */
.ts-col-input > .form,
.ts-col-output > .form { width: 100%; }

/* Accordion 自身视觉收敛 */
.gradio-container .label-wrap { font-weight: 600; }
.gradio-container .accordion {
  border: 1px solid var(--ts-line) !important;
  border-radius: 14px !important;
  background: var(--ts-card) !important;
}
.gradio-container .accordion > .label-wrap {
  padding: 12px 16px !important;
  letter-spacing: .01em;
}

/* =====================================================
   Markdown 区域内 padding（避免长文紧贴卡片圆角）
   ===================================================== */
.gradio-container .prose,
.gradio-container .markdown {
  padding: 4px 6px;
  line-height: 1.7;
}

/* =====================================================
   分析报告：居中、浅色背景卡、一键复制
   ===================================================== */
.ts-report-wrap {
  max-width: 860px;
  margin: 0 auto;
}
.ts-report-card {
  position: relative;
  background: #F8F5EF;                 /* 浅色背景，区别于纯白卡 */
  border: 1px solid var(--ts-line);
  border-radius: 16px;
  padding: 26px 30px 28px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.6),
              0 10px 28px -22px rgba(var(--ts-shadow),.5);
}
.ts-report-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px; margin-bottom: 14px;
  padding-bottom: 12px; border-bottom: 1px solid var(--ts-line-2);
}
.ts-report-head .ts-report-title {
  font-family: var(--font-display); font-weight: 600;
  font-size: 13px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--ts-faint);
  display: flex; align-items: center; gap: 9px;
}
.ts-report-head .ts-report-title .ts-section-dot {
  display: inline-block; width: 5px; height: 5px;
  border-radius: 50%; background: var(--ts-accent);
}
.ts-copy-btn {
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
  color: var(--ts-ink-soft); background: var(--ts-card);
  border: 1px solid var(--ts-line); border-radius: 8px;
  padding: 6px 13px; cursor: pointer;
  display: inline-flex; align-items: center; gap: 7px;
  transition: all .2s var(--ts-ease);
  white-space: nowrap; flex: 0 0 auto;
}
.ts-copy-btn:hover {
  border-color: var(--ts-accent); color: var(--ts-accent-d);
  background: #FBF0EC;
}
.ts-copy-btn:active { transform: scale(.96); }
.ts-copy-btn .ts-copy-ico {
  width: 6px; height: 6px; border-radius: 1px;
  border: 1.5px solid currentColor; display: inline-block;
}
.ts-copy-btn.ts-copied {
  border-color: #CBD6C8; color: var(--ts-sage); background: #F3F6F1;
}
/* 报告正文：把 markdown 排版收进卡片 */
.ts-report-body { line-height: 1.75; color: var(--ts-ink); }
.ts-report-body h1, .ts-report-body h2, .ts-report-body h3 {
  font-family: var(--font-display); letter-spacing: -.01em;
}
.ts-report-body h2 { font-size: 18px; margin: 18px 0 8px; }
.ts-report-body h3 { font-size: 15px; margin: 14px 0 6px; color: var(--ts-ink-soft); }
.ts-report-body pre, .ts-report-body code {
  font-family: 'JetBrains Mono', monospace;
}
.ts-report-body pre {
  background: var(--ts-card); border: 1px solid var(--ts-line);
  border-radius: 10px; padding: 12px 14px; overflow-x: auto;
}
/* 隐藏的纯文本副本，供复制按钮读取 */
.ts-report-raw { display: none; }

/* =====================================================
   对照参考样本（检索）源码展示
   ===================================================== */
.ts-refs-wrap { max-width: 860px; margin: 0 auto; }
.ts-ref-empty {
  text-align: center; color: var(--ts-faint); font-size: 13px;
  padding: 18px; border: 1px dashed var(--ts-line); border-radius: 12px;
  background: #FAF8F3;
}
.ts-ref-card {
  background: var(--ts-card); border: 1px solid var(--ts-line);
  border-radius: 14px; margin-bottom: 14px; overflow: hidden;
}
.ts-ref-head {
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  padding: 13px 18px; background: #F8F5EF;
  border-bottom: 1px solid var(--ts-line-2);
}
.ts-ref-name {
  font-family: var(--font-display); font-weight: 600;
  font-size: 14px; color: var(--ts-ink);
}
.ts-ref-id {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: var(--ts-faint);
}
.ts-ref-dist {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: var(--ts-accent-d); background: #FBF0EC;
  border: 1px solid #E6C9BF; border-radius: 6px; padding: 2px 8px;
  margin-left: auto;
}
.ts-ref-tags { display: flex; gap: 6px; flex-wrap: wrap; }
.ts-ref-tag {
  font-size: 10.5px; color: var(--ts-ink-soft);
  background: var(--ts-card); border: 1px solid var(--ts-line);
  border-radius: 6px; padding: 2px 8px;
  font-family: 'JetBrains Mono', monospace;
}
.ts-ref-code {
  margin: 0; padding: 16px 18px;
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  line-height: 1.6; color: var(--ts-ink);
  background: var(--ts-card); overflow-x: auto;
  white-space: pre-wrap; word-break: break-all;
  max-height: 360px; overflow-y: auto;
}
.ts-ref-head {
  word-break: break-all;
}
.ts-ref-excerpt-note {
  font-size: 11px; color: var(--ts-faint);
  padding: 0 18px 12px; font-style: italic;
}


.ts-footer {
  text-align: center; color: var(--ts-faint); font-size: 12px;
  padding: 24px 16px 10px; line-height: 1.7;
  border-top: 1px solid var(--ts-line); margin-top: 16px;
}
.ts-footer code {
  background: #F4F1EA; border: 1px solid var(--ts-line-2);
  padding: 1px 6px; border-radius: 6px; font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
}
.ts-footer .ts-foot-sep {
  display: inline-block; margin: 0 10px; color: var(--ts-line);
}

"""


# =====================================================================
# HTML 工具（签名与旧版兼容）
# =====================================================================

def hero_html() -> str:
    """左文右标二栏 hero。

    左：eyebrow + h1（不换行的一句主标题）。
    右：以"01 / 02 / 03"序号引导的能力清单。
    文案按用户指定改写为直陈式。
    """
    return (
        '<div class="ts-hero">'
        '<div class="ts-hero-left">'
        '<p class="ts-eyebrow">Shadertoy · GLSL · 多智能体</p>'
        '<h1><span class="ts-mark">自动化</span>分析，创作，改写 <span class="ts-mark">Shader</span></h1>'
        '</div>'
        '<div class="ts-hero-right">'
        '<div class="ts-cap">'
        '<span class="ts-cap-num">01</span>'
        '<span class="ts-cap-body"><b>Analyzer</b>'
        '<span class="ts-mute">粘贴 GLSL 代码，快速理解它在做什么；</span></span>'
        '</div>'
        '<div class="ts-cap">'
        '<span class="ts-cap-num">02</span>'
        '<span class="ts-cap-body"><b>Generator</b>'
        '<span class="ts-mute">用一句话描述效果，生成可运行的 shader；</span></span>'
        '</div>'
        '<div class="ts-cap">'
        '<span class="ts-cap-num">03</span>'
        '<span class="ts-cap-body"><b>Remixer</b>'
        '<span class="ts-mute">在现有代码上做局部调整，保留结构，只改你想改的部分。</span></span>'
        '</div>'
        '</div>'
        '</div>'
    )


def section_title(text: str) -> str:
    """细线 — 居中标签 — 细线 的区块标题。"""
    return (
        '<div class="ts-section">'
        f'<span class="ts-section-label">'
        f'<span class="ts-section-dot"></span>{text}'
        '</span>'
        '</div>'
    )


def _chip(text: str, kind: str = "") -> str:
    cls = "ts-chip" + (f" ts-chip-{kind}" if kind else "")
    return f'<span class="{cls}"><span class="ts-led"></span>{text}</span>'


# 兼容旧调用名
def badge(text: str, kind: str = "info") -> str:
    mapping = {"ok": "ok", "fail": "fail", "warn": "accent", "info": ""}
    return _chip(text, mapping.get(kind, ""))


def status_html(
    *,
    backend_label: str,
    vstore_label: str,
    elapsed_ms: float,
    iterations: int = 0,
    compile_ok: bool | None = None,
) -> str:
    parts: list[str] = [
        _chip(f"渲染 {backend_label}"),
        _chip(f"检索 {vstore_label}"),
    ]
    if compile_ok is True:
        parts.append(_chip("编译通过", "ok"))
    elif compile_ok is False:
        parts.append(_chip("编译失败", "fail"))
    if iterations:
        parts.append(_chip(f"迭代 {iterations}", "accent"))
    parts.append(_chip(f"{elapsed_ms/1000:.2f}s"))
    return '<div class="ts-status">' + "".join(parts) + "</div>"


def running_html(msg: str = "正在处理…") -> str:
    return (
        f'<div class="ts-status">'
        f'<span class="ts-chip ts-chip-run"><span class="ts-led"></span>{msg}</span>'
        f'</div>'
    )


def idle_status_html(msg: str = "待命 · 点击下方按钮开始") -> str:
    """初始 / 空闲状态条。让运行前后右列结构一致，不再"凭空出现"。"""
    return (
        f'<div class="ts-status">'
        f'<span class="ts-chip ts-chip-idle"><span class="ts-led"></span>{msg}</span>'
        f'</div>'
    )


def error_block(msg: str) -> str:
    if not msg:
        return ""
    return f'<div class="ts-error">{msg}</div>'


def diagnostics_html(items: list[str]) -> str:
    if not items:
        return ""
    body = "<br>".join(f"· {s}" for s in items if s)
    return f'<div class="ts-diag">{body}</div>'


def preview_placeholder(msg: str = "运行后在此显示实时预览") -> str:
    """预览占位：与运行后等高，给出操作提示，不再是一行小字。"""
    return (
        '<div class="ts-preview"><div class="ts-preview-empty">'
        '<span class="ts-ph-mark">◇</span>'
        f'<span class="ts-ph-msg">{msg}</span>'
        '<span class="ts-ph-hint">PREVIEW · IDLE</span>'
        '</div></div>'
    )

# =====================================================================
# 分析报告卡片 + 对照参考样本源码（新增）
# =====================================================================

import html as _html


def _md_to_html(md_text: str) -> str:
    """把报告 markdown 渲染成 HTML。

    优先用 markdown 库（环境里通常有）；缺库时优雅降级为转义后的 <pre>，
    保证零硬依赖——不会因为目标环境没装 markdown 而报错。
    """
    if not md_text:
        return ""
    try:
        import markdown as _markdown  # 可选依赖
        return _markdown.markdown(
            md_text, extensions=["fenced_code", "tables"]
        )
    except Exception:
        return f'<pre style="white-space:pre-wrap;">{_html.escape(md_text)}</pre>'


def report_html(report_md: str) -> str:
    """居中浅色背景的分析报告卡，自带一键复制按钮。

    - 复制按钮通过 data-copy-target 指向同卡内隐藏的原文节点，
      由全局 JS（GLOBAL_JS）接管点击 → navigator.clipboard 复制 markdown 原文。
    - 报告正文渲染为 HTML；缺 markdown 库时降级为可读 <pre>。
    """
    if not report_md or not report_md.strip():
        return (
            '<div class="ts-report-wrap"><div class="ts-report-card">'
            '<div class="ts-report-body" style="color:var(--ts-faint);'
            'text-align:center;padding:24px 0;">运行分析后，报告将显示在这里。</div>'
            '</div></div>'
        )
    body = _md_to_html(report_md)
    raw = _html.escape(report_md)
    return (
        '<div class="ts-report-wrap"><div class="ts-report-card">'
        '<div class="ts-report-head">'
        '<span class="ts-report-title"><span class="ts-section-dot"></span>分析报告</span>'
        '<button class="ts-copy-btn" data-copy-target="ts-report-raw" type="button">'
        '<span class="ts-copy-ico"></span><span class="ts-copy-label">复制报告</span>'
        '</button>'
        '</div>'
        f'<textarea class="ts-report-raw" id="ts-report-raw">{raw}</textarea>'
        f'<div class="ts-report-body">{body}</div>'
        '</div></div>'
    )


def references_html(refs: list[dict]) -> str:
    """对照参考样本（检索结果）的源码展示卡列表。

    refs 来自 runners.run_analyze 的 references 字段：
      [{ shader_id, name, distance, tags, code, is_excerpt }, ...]
    """
    if not refs:
        return (
            '<div class="ts-refs-wrap"><div class="ts-ref-empty">'
            '本次分析未检索到对照样本（向量库为空或已关闭）。'
            '</div></div>'
        )
    cards: list[str] = []
    for r in refs:
        name = _html.escape(str(r.get("name", "") or "(未命名)"))
        sid = _html.escape(str(r.get("shader_id", "") or ""))
        dist = r.get("distance", None)
        tags = r.get("tags", []) or []
        code = r.get("code", "") or ""
        is_excerpt = bool(r.get("is_excerpt"))

        dist_html = (f'<span class="ts-ref-dist">distance {dist:.4f}</span>'
                     if isinstance(dist, (int, float)) else "")
        tags_html = ""
        if tags:
            chips = "".join(
                f'<span class="ts-ref-tag">{_html.escape(str(t))}</span>' for t in tags
            )
            tags_html = f'<span class="ts-ref-tags">{chips}</span>'
        code_html = _html.escape(code) if code else "（无可展示源码）"
        note = ('<div class="ts-ref-excerpt-note">'
                '※ 仅展示代码片段（完整源码需从语料库按 ID 获取）</div>'
                if is_excerpt else "")
        cards.append(
            '<div class="ts-ref-card">'
            '<div class="ts-ref-head">'
            f'<span class="ts-ref-name">{name}</span>'
            f'<span class="ts-ref-id">{sid}</span>'
            f'{tags_html}{dist_html}'
            '</div>'
            f'<pre class="ts-ref-code">{code_html}</pre>'
            f'{note}'
            '</div>'
        )
    return '<div class="ts-refs-wrap">' + "".join(cards) + "</div>"


# =====================================================================
# 全局 JS：复制按钮 + 下拉框层叠兜底
# 通过 app.py 里的 gr.HTML(GLOBAL_JS) 注入一次。
# =====================================================================

GLOBAL_JS = """
<script>
(function(){
  // ---------- 一键复制报告 ----------
  document.addEventListener('click', function(ev){
    var btn = ev.target.closest && ev.target.closest('.ts-copy-btn');
    if(!btn) return;
    ev.preventDefault();
    var id = btn.getAttribute('data-copy-target');
    var src = id ? document.getElementById(id) : null;
    var text = src ? (src.value != null ? src.value : src.textContent) : '';
    var done = function(){
      btn.classList.add('ts-copied');
      var lab = btn.querySelector('.ts-copy-label');
      var old = lab ? lab.textContent : '';
      if(lab) lab.textContent = '已复制';
      setTimeout(function(){
        btn.classList.remove('ts-copied');
        if(lab) lab.textContent = old || '复制报告';
      }, 1600);
    };
    if(navigator.clipboard && navigator.clipboard.writeText){
      navigator.clipboard.writeText(text).then(done, function(){
        fallbackCopy(text); done();
      });
    } else { fallbackCopy(text); done(); }
  });
  function fallbackCopy(text){
    try{
      var ta = document.createElement('textarea');
      ta.value = text; ta.style.position='fixed'; ta.style.opacity='0';
      document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); document.body.removeChild(ta);
    }catch(e){}
  }

})();
</script>
"""
