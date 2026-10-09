"""Tests for how ``ultrarag.client`` launches local MCP servers.

Each local server runs as its own subprocess, and that subprocess must use the
very interpreter that has ``ultrarag`` importable. The launch config used to
hardcode the bare name ``"python"``, which is resolved through PATH and can
point at a different interpreter than the one running the client (a system
Python while the client sits in a virtual environment, an IDE-provided one,
...). The child then died on ``import ultrarag`` / ``import yaml`` while the
only thing the caller saw was the MCP client's opaque ``Connection closed``.
"""

import inspect
import os
import subprocess
import sys
from pathlib import Path

import ultrarag.client as client
from ultrarag.client import local_server_launch_config

SERVER = "servers/sayhello/src/sayhello.py"


def test_server_runs_under_the_client_interpreter():
    cfg = local_server_launch_config(SERVER)
    assert cfg["command"] == sys.executable


def test_entry_script_is_forwarded_as_the_only_argument():
    cfg = local_server_launch_config(SERVER)
    assert cfg["args"] == [SERVER]


def test_parent_environment_is_inherited():
    cfg = local_server_launch_config(SERVER)
    assert cfg["env"] == os.environ.copy()


def test_spawned_interpreter_can_import_ultrarag_and_its_dependencies(tmp_path):
    """The spawned command must be able to import what a server needs.

    With the old PATH-resolved ``"python"`` this failed on any machine whose
    PATH Python differs from the environment UltraRAG was installed into.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import ultrarag.server  # noqa: F401\nprint('PROBE_OK')\n", encoding="utf-8"
    )
    cfg = local_server_launch_config(str(probe))

    ret = subprocess.run(
        [cfg["command"], *cfg["args"]],
        env=cfg["env"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert ret.returncode == 0, ret.stderr
    assert "PROBE_OK" in ret.stdout


def test_module_entrypoint_does_not_wrap_the_sync_main():
    """``python -m ultrarag.client`` must call ``main()`` directly.

    ``main`` performs its own ``asyncio.run`` per subcommand. Wrapping it again
    made ``python -m ultrarag.client build ...`` finish its work and then die on
    ``ValueError: a coroutine was expected, got None`` with exit code 1. The
    console script calls ``main()`` directly, which is why CI never noticed.
    """
    assert not inspect.iscoroutinefunction(client.main)

    source = Path(client.__file__).read_text(encoding="utf-8")
    entry = source.split('if __name__ == "__main__":', 1)[1]
    assert "asyncio.run(main())" not in entry
    assert "main()" in entry
