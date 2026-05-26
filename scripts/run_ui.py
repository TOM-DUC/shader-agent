"""阶段七：Gradio UI 启动入口。

用法
=====
    python -m scripts.run_ui                       # 默认 127.0.0.1:7860 并自动打开浏览器
    python -m scripts.run_ui --port 8800
    python -m scripts.run_ui --host 0.0.0.0        # 暴露到局域网
    python -m scripts.run_ui --share               # Gradio 公网穿透（公开访问，慎用）
    python -m scripts.run_ui --no-browser          # 不自动开浏览器
    python -m scripts.run_ui --auth user:pass      # 基本鉴权
"""
from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--host", default="127.0.0.1",
                   help="监听地址（默认 127.0.0.1，仅本机；0.0.0.0 暴露到 LAN）")
    p.add_argument("--port", type=int, default=7860, help="监听端口")
    p.add_argument("--share", action="store_true",
                   help="启用 Gradio share 公网链接（72h 临时）")
    p.add_argument("--no-browser", action="store_true",
                   help="启动后不自动打开浏览器")
    p.add_argument("--auth", type=str, default="",
                   help="HTTP basic auth：'user:pass' 形式；多账号用逗号隔开 'a:1,b:2'")
    return p.parse_args()


def _parse_auth(spec: str):
    """把 'user:pass[,user2:pass2]' 解析成 Gradio 期望的 (u,p) 或 list。"""
    if not spec:
        return None
    pairs = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise SystemExit(f"--auth 形式不对（缺冒号）：{token!r}")
        u, p = token.split(":", 1)
        pairs.append((u, p))
    if not pairs:
        return None
    return pairs[0] if len(pairs) == 1 else pairs


def main() -> int:
    args = parse_args()
    try:
        from shader_agent.ui import launch
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        auth=_parse_auth(args.auth),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
