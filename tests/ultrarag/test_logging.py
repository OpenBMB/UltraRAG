"""Regression coverage for application logging and configuration boundaries."""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ultrarag import client


@pytest.mark.parametrize("level", ["debug", "info", "warn", "error"])
def test_local_cli_keeps_requested_level(tmp_path: Path, level: str) -> None:
    server = tmp_path / "probe"
    (server / "src").mkdir(parents=True)
    (server / "parameter.yaml").write_text("{}", encoding="utf-8")
    (server / "src" / "probe.py").write_text(
        "from ultrarag.server import UltraRAG_MCP_Server\n"
        "app = UltraRAG_MCP_Server('probe')\n"
        "@app.tool(output='->answer')\n"
        "def emit() -> dict:\n"
        "    app.logger.debug('SERVER_DEBUG_MARKER')\n"
        "    app.logger.info('SERVER_INFO_MARKER')\n"
        "    app.logger.warning('SERVER_WARNING_MARKER')\n"
        "    return {'answer': 'ok'}\n",
        encoding="utf-8",
    )
    pipeline = tmp_path / "probe.yaml"
    pipeline.write_text(
        yaml.safe_dump({"servers": {"probe": str(server)}, "pipeline": ["probe.emit"]}),
        encoding="utf-8",
    )
    env = {**os.environ, "COLUMNS": "240", "PYTHONPATH": str(Path(client.__file__).parents[1])}
    for command in ("build", "run"):
        result = subprocess.run(
            [sys.executable, "-m", "ultrarag.client", command, str(pipeline),
             "--no-mcp", "--log_level", level],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30, check=False,
        )
        assert result.returncode == 0, result.stderr
        info_marker = "All parameters have been saved" if command == "build" else "Pipeline execution completed"
        assert (info_marker in result.stderr) == (level in {"debug", "info"})
        if command == "run":
            assert ("SERVER_DEBUG_MARKER" in result.stderr) == (level == "debug")
            assert ("SERVER_INFO_MARKER" in result.stderr) == (level in {"debug", "info"})
            assert ("SERVER_WARNING_MARKER" in result.stderr) == (level != "error")
    log_text = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "logs").glob("*.log"))
    assert ("Pipeline execution completed" in log_text) == (level in {"debug", "info"})


def test_server_specific_level_does_not_reconfigure_application() -> None:
    from ultrarag.local_server import LocalServer
    from ultrarag.mcp_logging import get_logger

    base = logging.getLogger("UltraRAG")
    original = base.level
    try:
        get_logger("Client", "debug")
        server = LocalServer("quiet-probe", log_level="warn")
        LocalServer("inherited-probe")
        assert server.logger.level == logging.WARNING
        assert base.level == logging.DEBUG
        assert all(h.level == logging.DEBUG for h in base.handlers)
    finally:
        get_logger("Client", logging.getLevelName(original).lower())
        logging.getLogger("UltraRAG.Server.quiet-probe").setLevel(logging.NOTSET)
