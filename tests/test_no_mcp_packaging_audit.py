"""Optional MCP transports must not leak into local-only integrations."""

import subprocess
import sys
from pathlib import Path


def test_serply_loads_without_mcp_and_reports_its_own_missing_key() -> None:
    source = Path(__file__).resolve().parents[1] / "servers/retriever/src"
    code = r'''
import importlib.abc
import logging
import os
import sys

class NoMCP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mcp', 'fastmcp'}:
            raise AssertionError('Local backend imported MCP: ' + fullname)

sys.meta_path.insert(0, NoMCP())
sys.path.insert(0, sys.argv[1])
os.environ.pop('SERPLY_API_KEY', None)
from ultrarag.errors import ToolError
from websearch_backends import create_websearch_backend
try:
    create_websearch_backend('serply', logging.getLogger('test'))
except ToolError as exc:
    assert 'SERPLY_API_KEY' in str(exc)
else:
    raise AssertionError('Expected a missing Serply API key error')
assert not any(k.split('.')[0] in {'mcp', 'fastmcp'} for k in sys.modules)
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(source)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
