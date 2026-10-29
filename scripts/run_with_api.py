#!/usr/bin/env python3
"""Start PaperScope with an API key kept only in process memory."""

from __future__ import annotations

import argparse
from getpass import getpass
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaperScope with a temporary DeepSeek API key.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="deepseek-flash")
    args = parser.parse_args()

    api_key = os.environ.get("PAPERSCOPE_LLM_API_KEY", "").strip()
    if not api_key:
        api_key = getpass("DeepSeek API Key（隐藏输入，不会写入文件）：").strip()
    if not api_key:
        raise SystemExit("未输入 API Key。")

    os.environ["PAPERSCOPE_LLM_PROVIDER"] = "deepseek"
    os.environ["PAPERSCOPE_LLM_API_KEY"] = api_key
    os.environ.setdefault("PAPERSCOPE_LLM_BASE_URL", "https://api.deepseek.com")
    os.environ["PAPERSCOPE_LLM_MODEL"] = args.model

    from backend.server import PaperScopeHandler
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer((args.host, args.port), PaperScopeHandler)
    print(f"PaperScope 模型测试模式：http://{args.host}:{args.port}")
    print(f"模型：{args.model}；API Key 仅保存在当前进程内存中。")
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
