"""Start the separate local API/frontend together; stop owned processes on Ctrl+C."""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

from clsweb.jobs import terminate


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="启动本机分类实验室（前后端独立进程）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not npm or not (root / "web/node_modules/vite").is_dir():
        parser.error("请先安装 Node.js，然后执行：cd web → npm ci → cd ..")
    for port in (8000, 5173):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                parser.error(f"端口 {port} 已占用，请先关闭已有服务，或直接打开 http://127.0.0.1:5173")
    directory = root / "logs/web"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    processes, logs = [], []
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    try:
        for name, command, cwd in (
            ("backend", [sys.executable, "-m", "clsweb.app"], root),
            ("frontend", [npm, "run", "dev"], root / "web"),
        ):
            log_path = directory / f"{name}_{stamp}.log"
            log = log_path.open("wb")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            )
            print(f"{name} 日志：{log_path}", flush=True)
        deadline = time.monotonic() + 60
        ready = False
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("服务启动失败，请检查上面的日志")
            try:
                for url in ("http://127.0.0.1:8000/api/health", "http://127.0.0.1:5173"):
                    with urllib.request.urlopen(url, timeout=2) as response:
                        if response.status != 200:
                            raise OSError("服务尚未就绪")
                ready = True
                break
            except (OSError, urllib.error.URLError):
                time.sleep(0.5)
        if not ready:
            raise RuntimeError("服务启动超时，请检查日志")
        print("打开 http://127.0.0.1:5173；按 Ctrl+C 停止前后端及本次 Web 训练。", flush=True)
        if not args.no_browser:
            webbrowser.open("http://127.0.0.1:5173")
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("正在停止 Web 服务…", flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                terminate(process)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
