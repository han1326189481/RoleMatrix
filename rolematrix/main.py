"""RoleMatrix 入口：启动 Bridge Server。"""
from __future__ import annotations

import uvicorn

from .bridge.server import create_app
from .config import get_settings
from .logger import get_logger, setup_logging


def main() -> None:
    setup_logging()
    settings = get_settings()
    log = get_logger("main")
    log.info(
        "RoleMatrix 启动 http://%s:%d",
        settings.server.host,
        settings.server.port,
    )
    app = create_app()
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
