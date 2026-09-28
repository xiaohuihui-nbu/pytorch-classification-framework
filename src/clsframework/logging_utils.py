"""Scoped UTF-8 operation logs without replacing the application's logging setup."""

import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

_operation = ContextVar("cls_log_operation", default=None)


@contextmanager
def operation_log(mode, settings, directory):
    if int(os.environ.get("RANK", "0")) or int(os.environ.get("LOCAL_RANK", "0")):
        yield None
        return
    identity = uuid.uuid4().hex[:8]
    token = _operation.set(identity)
    logger = logging.getLogger("clsframework")
    previous_level = logger.level
    handlers, attached = [], []
    path = None
    started = time.perf_counter()
    try:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        if settings.file:
            directory = Path(settings.directory or directory)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{datetime.now():%Y%m%d-%H%M%S}-{mode}-{identity}.log"
            handlers.append(logging.FileHandler(path, encoding="utf-8", mode="x"))
        if settings.console:
            handlers.append(logging.StreamHandler())
        logger.setLevel(settings.level)
        for handler in handlers:
            handler.setLevel(settings.level)
            handler.setFormatter(formatter)
            handler.addFilter(lambda record: _operation.get() == identity)
            logger.addHandler(handler)
            attached.append((logger, handler))
            # Lightning owns its console handlers; only capture its records to file.
            if isinstance(handler, logging.FileHandler):
                for name in ("lightning.pytorch", "lightning.fabric"):
                    upstream = logging.getLogger(name)
                    upstream.addHandler(handler)
                    attached.append((upstream, handler))
        logger.info("开始 %s；日志文件=%s", mode, path or "disabled")
        try:
            yield path
        except BaseException:
            logger.exception("%s 失败或中断，耗时 %.3fs", mode, time.perf_counter() - started)
            raise
        else:
            logger.info("完成 %s，耗时 %.3fs", mode, time.perf_counter() - started)
    finally:
        for target, handler in attached:
            target.removeHandler(handler)
        for handler in handlers:
            handler.close()
        logger.setLevel(previous_level)
        _operation.reset(token)
