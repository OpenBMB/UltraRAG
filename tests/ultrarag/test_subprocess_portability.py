"""Parent-following subprocesses must work on macOS as well as Linux."""

import os
import sys
from pathlib import Path

from ultrarag.utils import popen_follow_parent, set_pdeathsig


def test_parent_following_subprocess_preserves_arguments_environment_and_exit(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "child.txt"
    command = [
        sys.executable,
        "-c",
        "import os,sys; from pathlib import Path; Path(sys.argv[1]).write_text(os.environ['ULTRARAG_CHILD_TEST']); sys.exit(7)",
        str(marker),
    ]
    process = popen_follow_parent(
        command, env={**os.environ, "ULTRARAG_CHILD_TEST": "argument with spaces"}
    )
    assert process.wait(timeout=10) == 7
    assert marker.read_text() == "argument with spaces"


def test_parent_death_signal_is_a_noop_on_non_linux_posix() -> None:
    if os.name == "posix" and not sys.platform.startswith("linux"):
        set_pdeathsig()
