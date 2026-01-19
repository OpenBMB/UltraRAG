import os
import sys
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def _pick_gguf_filename(repo_files: list[str]) -> str:
    ggufs = [f for f in repo_files if f.lower().endswith(".gguf")]
    if not ggufs:
        raise RuntimeError("No .gguf files found in the repo.")

    # Prefer common quant names if available.
    prefer = ["Q4_K_M", "q4_k_m", "Q4", "q4"]
    for p in prefer:
        for f in ggufs:
            if p in f:
                return f
    return ggufs[0]


def main() -> int:
    repo_id = _env("GGUF_REPO_ID", "openbmb/AgentCPM-Report-GGUF")
    revision = _env("GGUF_REVISION", "main")
    filename = _env("GGUF_FILENAME")  # optional

    models_dir = Path(_env("GGUF_MODELS_DIR", "/models"))
    link_name = _env("GGUF_LINK_NAME", "model.gguf")

    token = _env("HF_TOKEN") or _env("HUGGINGFACE_TOKEN")
    endpoint = _env("HF_ENDPOINT")

    models_dir.mkdir(parents=True, exist_ok=True)
    link_path = models_dir / link_name

    # If the alias file already exists and is non-empty, skip.
    if link_path.exists() and link_path.is_file() and link_path.stat().st_size > 0:
        print(f"[gguf-downloader] already present: {link_path}")
        return 0

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        print(
            "[gguf-downloader] ERROR: huggingface_hub not installed (did pip install run?)",
            file=sys.stderr,
        )
        print(f"[gguf-downloader] import error: {e}", file=sys.stderr)
        return 2

    api = HfApi(endpoint=endpoint) if endpoint else HfApi()
    print(f"[gguf-downloader] repo_id={repo_id} revision={revision} endpoint={endpoint or 'default'}")

    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision, token=token)
    except Exception as e:
        print(f"[gguf-downloader] ERROR: failed to list repo files: {e}", file=sys.stderr)
        return 3

    if not filename:
        try:
            filename = _pick_gguf_filename(repo_files)
        except Exception as e:
            print(f"[gguf-downloader] ERROR: {e}", file=sys.stderr)
            return 4
        print(f"[gguf-downloader] auto-selected GGUF: {filename}")
    else:
        print(f"[gguf-downloader] using GGUF: {filename}")

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"[gguf-downloader] ERROR: download failed: {e}", file=sys.stderr)
        return 5

    src = Path(local_path)
    if not src.exists() or src.stat().st_size == 0:
        print(f"[gguf-downloader] ERROR: downloaded file missing/empty: {src}", file=sys.stderr)
        return 6

    # Create/refresh an alias at /models/model.gguf for llama.cpp.
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(src)
        print(f"[gguf-downloader] linked: {link_path} -> {src}")
    except Exception:
        # Fallback: copy if symlinks are not allowed.
        import shutil

        shutil.copyfile(src, link_path)
        print(f"[gguf-downloader] copied: {src} -> {link_path}")

    print(f"[gguf-downloader] done: {link_path} ({link_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


