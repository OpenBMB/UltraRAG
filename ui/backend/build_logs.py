"""Collect bounded, thread-scoped build logs without redirecting shared stdout."""

import logging
import threading


class BuildLogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.owner = threading.get_ident()
        self.lines: list[str] = []
        self.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.owner:
            self.lines.append(self.format(record)[:4000])
            self.lines[:] = self.lines[-200:]
