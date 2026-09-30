"""同时提供已构建前端与 API；默认本机访问，--host 可指定局域网 IPv4 地址。"""

import argparse
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles

from clsweb.app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="监听本机或局域网 IPv4 地址")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--frontend-dir", type=Path, help="生产前端目录，默认 workspace/web/dist")
    parser.add_argument("--container", action="store_true", help="容器内监听所有接口；Host/Origin 仍由 --host 限制")
    args = parser.parse_args()
    root = args.workspace.resolve()
    frontend = (args.frontend_dir or root / "web/dist").resolve()
    if not (frontend / "index.html").is_file():
        parser.error("缺少 web/dist/index.html，请先在 web 目录执行 npm run build")
    if not (root / "configs/flower").is_dir():
        parser.error("workspace 缺少 configs/flower")
    app = create_app(root, host=args.host, port=args.port)
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    uvicorn.run(
        app, host="0.0.0.0" if args.container else args.host,
        port=args.port, workers=1, timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
