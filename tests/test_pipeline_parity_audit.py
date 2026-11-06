"""Compare complete pipeline state through real stdio MCP and local execution."""

import asyncio
from pathlib import Path

import pytest
import yaml

from ultrarag.client import build, run


def make_pipeline(tmp_path: Path, steps: list, values: list[int]) -> Path:
    folder = tmp_path / "worker"
    (folder / "src").mkdir(parents=True)
    cleanup = tmp_path / "cleanup.txt"
    source = f'''
from pathlib import Path
from ultrarag.server import UltraRAG_MCP_Server
app = UltraRAG_MCP_Server("worker")
@app.tool(output="initial_values->values")
def seed(initial_values: list[int]) -> dict:
    return {{"values": initial_values}}
@app.tool(output="values,delta->values")
def advance(values: list[int], delta: int) -> dict:
    for i in range(len(values)):
        values[i] += delta
    return {{"values": values}}
@app.tool(output="values->ans_ls")
async def finish(values: list[int]) -> dict:
    return {{"ans_ls": values}}
@app.tool(output="->None")
def fail() -> None:
    raise ValueError("deliberate failure")
@app.tool(output="->None")
def vllm_shutdown() -> None:
    with Path({str(cleanup)!r}).open("a", encoding="utf-8") as stream:
        stream.write("closed\\n")
if __name__ == "__main__":
    app.run(transport="stdio")
'''
    (folder / "src/worker.py").write_text(source, encoding="utf-8")
    (folder / "parameter.yaml").write_text(
        yaml.safe_dump({"initial_values": values, "delta": 2}), encoding="utf-8"
    )
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump({"servers": {"alias_worker": str(folder)}, "pipeline": steps}),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("values", [[], [0], [-3, 0, 2, 100]])
@pytest.mark.parametrize("loops", [1, 3])
def test_remapping_loops_and_every_snapshot_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, values: list[int], loops: int
) -> None:
    pytest.importorskip("fastmcp")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    steps = [
        {"alias_worker.seed": {"output": {"values": "working"}}},
        {"loop": {"times": loops, "steps": [
            {"alias_worker.advance": {"input": {"values": "working"},
                                      "output": {"values": "working"}}}
        ]}},
        {"alias_worker.finish": {"input": {"values": "working"}}},
    ]
    path = make_pipeline(tmp_path, steps, values)

    async def exercise() -> None:
        results, configs, params = [], [], []
        for local in (False, True):
            await build(str(path), no_mcp=local)
            configs.append(yaml.safe_load((tmp_path / "server/pipeline_server.yaml").read_text("utf-8")))
            params.append(yaml.safe_load((tmp_path / "parameter/pipeline_parameter.yaml").read_text("utf-8")))
            results.append(await run(str(path), no_mcp=local, return_all=True))
        assert configs[0] == configs[1]
        assert params[0] == params[1]
        assert results[0] == results[1]
        assert results[0]["final_result"] == {"ans_ls": [v + 2 * loops for v in values]}
        assert (tmp_path / "cleanup.txt").read_text("utf-8").splitlines() == ["closed", "closed"]

    asyncio.run(exercise())


@pytest.mark.parametrize("local", [False, True])
def test_cleanup_also_runs_after_pipeline_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local: bool
) -> None:
    if not local:
        pytest.importorskip("fastmcp")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    path = make_pipeline(tmp_path, ["alias_worker.fail"], [])

    async def exercise() -> None:
        await build(str(path), no_mcp=local)
        with pytest.raises(Exception, match="deliberate failure"):
            await run(str(path), no_mcp=local)
        assert (tmp_path / "cleanup.txt").read_text("utf-8").splitlines() == ["closed"]

    asyncio.run(exercise())
