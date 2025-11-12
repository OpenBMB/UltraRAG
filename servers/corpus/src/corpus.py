import asyncio
import ast
import json
import os
import shutil
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path

from fastmcp.exceptions import ToolError
from PIL import Image
from tqdm import tqdm

from ultrarag.server import UltraRAG_MCP_Server

app = UltraRAG_MCP_Server("corpus")


def _save_jsonl(rows: Iterable[Dict[str, Any]], file_path: str) -> None:
    out_dir = Path(file_path).parent
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


@app.tool(output="parse_file_path,text_corpus_save_path->None")
async def build_text_corpus(
    parse_file_path: str,
    text_corpus_save_path: str,
) -> None:
    TEXT_EXTS = [".txt", ".md"]
    PMLIKE_EXT = [".pdf", ".xps", ".oxps", ".epub", ".mobi", ".fb2"]

    in_path = os.path.abspath(parse_file_path)
    if not os.path.exists(in_path):
        err_msg = f"Input path not found: {in_path}"
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    rows: List[Dict[str, Any]] = []

    def process_one_file(fp: str) -> None:
        ext = os.path.splitext(fp)[1].lower()
        stem = os.path.splitext(os.path.basename(fp))[0]

        if ext in TEXT_EXTS:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(fp, "r", encoding="latin-1", errors="ignore") as f:
                    text = f.read()
            text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
            rows.append({"id": stem, "title": stem, "contents": text})

        elif ext in PMLIKE_EXT:
            try:
                import pymupdf
            except ImportError:
                err_msg = "pymupdf not installed. Please `pip install pymupdf`."
                app.logger.error(err_msg)
                raise ToolError(err_msg)

            try:
                doc = pymupdf.open(fp)
            except Exception as e:
                err_msg = f"Skip (open failed): {fp} | reason: {e}"
                app.logger.warning(err_msg)
                return

            if getattr(doc, "is_encrypted", False):
                try:
                    doc.authenticate("")
                except Exception:
                    warn_msg = f"Skip (encrypted): {fp}"
                    app.logger.warning(warn_msg)
                    return

            texts = []
            for pg in doc:
                try:
                    t = pg.get_text("text")
                except Exception as e:
                    err_msg = f"Skip (get text failed): {fp} | reason: {e}"
                    app.logger.warning(err_msg)
                    t = ""
                texts.append(t.replace("\r\n", "\n").replace("\r", "\n").strip())
            merged = "\n\n".join(texts).strip()
            rows.append({"id": stem, "title": stem, "contents": merged})
        else:
            warn_msg = f"Unsupported file type, skip: {fp}"
            app.logger.warning(warn_msg)
            return

    if os.path.isfile(in_path):
        process_one_file(in_path)
    else:
        all_files = []
        for dp, _, fns in os.walk(in_path):
            for fn in sorted(fns):
                all_files.append(os.path.join(dp, fn))

        for fp in tqdm(all_files, desc="Building text corpus", unit="file"):
            process_one_file(fp)

    out_path = os.path.abspath(text_corpus_save_path)
    _save_jsonl(rows, out_path)

    info_msg = (
        f"Built text corpus: {out_path} "
        f"(rows={len(rows)}, from={'dir' if os.path.isdir(in_path) else 'file'}: {in_path})"
    )
    app.logger.info(info_msg)


@app.tool(output="parse_file_path,image_corpus_save_path->None")
async def build_image_corpus(
    parse_file_path: str,
    image_corpus_save_path: str,
) -> None:
    try:
        import pymupdf
    except ImportError:
        err_msg = "pymupdf not installed. Please `pip install pymupdf`."
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    in_path = os.path.abspath(parse_file_path)
    if not os.path.exists(in_path):
        err_msg = f"Input path not found: {in_path}"
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    corpus_jsonl = os.path.abspath(image_corpus_save_path)
    out_root = os.path.dirname(corpus_jsonl) or os.getcwd()
    base_img_dir = os.path.join(out_root, "image")
    os.makedirs(base_img_dir, exist_ok=True)

    pdf_list: List[str] = []
    if os.path.isfile(in_path):
        if not in_path.lower().endswith(".pdf"):
            err_msg = f"Only PDF is supported here. Got: {os.path.splitext(in_path)[1]}"
            app.logger.error(err_msg)
            raise ToolError(err_msg)
        pdf_list = [in_path]
    else:
        for dp, _, fns in os.walk(in_path):
            for fn in sorted(fns):
                if fn.lower().endswith(".pdf"):
                    pdf_list.append(os.path.join(dp, fn))
        pdf_list.sort()

    if not pdf_list:
        err_msg = f"No PDF files found under: {in_path}"
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    valid_rows: List[Dict[str, Any]] = []
    gid = 0

    for pdf_path in tqdm(pdf_list, desc="Building image corpus", unit="pdf"):
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        out_img_dir = os.path.join(base_img_dir, stem)
        os.makedirs(out_img_dir, exist_ok=True)

        try:
            doc = pymupdf.open(pdf_path)
        except Exception as e:
            warn_msg = f"Skip PDF (open failed): {pdf_path} | reason: {e}"
            app.logger.warning(warn_msg)
            continue

        if getattr(doc, "is_encrypted", False):
            try:
                doc.authenticate("")
            except Exception:
                warn_msg = f"Skip PDF (encrypted): {pdf_path}"
                app.logger.warning(warn_msg)
                continue

        zoom = 144 / 72.0
        mat = pymupdf.Matrix(zoom, zoom)

        for i, page in enumerate(doc):
            try:
                pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=pymupdf.csRGB)
            except Exception as e:
                warn_msg = f"Skip page {i} in {pdf_path}: render error: {e}"
                app.logger.warning(warn_msg)
                continue

            filename = f"page_{i}.jpg"
            save_path = os.path.join(out_img_dir, filename)
            rel_path = Path(os.path.join("image", stem, filename)).as_posix()

            try:
                pix.save(save_path, jpg_quality=90)
            except Exception as e:
                warn_msg = f"Skip page {i} in {pdf_path}: save error: {e}"
                app.logger.warning(warn_msg)
                continue
            finally:
                pix = None

            try:
                with Image.open(save_path) as im:
                    im.verify()
            except Exception as e:
                warn_msg = f"Skip page {i} in {pdf_path}: invalid image after save: {e}"
                app.logger.warning(warn_msg)
                try:
                    os.remove(save_path)
                except OSError as e:
                    warn_msg = f"Skip page {i} in {pdf_path}: remove error: {e}"
                    app.logger.warning(warn_msg)
                continue

            valid_rows.append(
                {
                    "id": gid,
                    "image_id": Path(os.path.join(stem, filename)).as_posix(),
                    "image_path": rel_path,
                }
            )
            gid += 1

    _save_jsonl(valid_rows, corpus_jsonl)
    info_msg = (
        f"Built image corpus: {corpus_jsonl} (valid images={len(valid_rows)}), "
        f"images root: {base_img_dir}, "
        f"pdf_count={len(pdf_list)}"
    )
    app.logger.info(info_msg)


@app.tool(output="parse_file_path,mineru_dir,mineru_extra_params->None")
async def mineru_parse(
    parse_file_path: str,
    mineru_dir: str,
    mineru_extra_params: Optional[Dict[str, Any]] = None,
) -> None:

    if shutil.which("mineru") is None:
        err_msg = "`mineru` executable not found. Please install it or add it to PATH."
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    if not parse_file_path:
        err_msg = "`parse_file_path` cannot be empty."
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    in_path = os.path.abspath(parse_file_path)
    if not os.path.exists(in_path):
        err_msg = f"Input path not found: {in_path}"
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    if os.path.isfile(in_path) and not in_path.lower().endswith(".pdf"):
        err_msg = f"Only .pdf files or directories are supported: {in_path}"
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    out_root = os.path.abspath(mineru_dir)
    os.makedirs(out_root, exist_ok=True)

    extra_args: List[str] = []
    if mineru_extra_params:
        for k in sorted(mineru_extra_params.keys()):
            v = mineru_extra_params[k]
            extra_args.append(f"--{k}")
            if v is not None and v != "":
                extra_args.append(str(v))

    cmd = ["mineru", "-p", in_path, "-o", out_root] + extra_args
    info_msg = f"Starting mineru command: {' '.join(cmd)}"
    app.logger.info(info_msg)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for line in proc.stdout:
            app.logger.info(line.decode("utf-8", errors="replace").rstrip())

        returncode = await proc.wait()
        if returncode != 0:
            err_msg = f"mineru exited with non-zero code: {returncode}"
            app.logger.error(err_msg)
            raise ToolError(err_msg)
    except Exception as e:
        err_msg = f"Unexpected error while running mineru: {e}"
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    info_msg = f"mineru finished processing {in_path} into {out_root}"
    app.logger.info(info_msg)


def _list_images(images_dir: str) -> List[str]:
    if not os.path.isdir(images_dir):
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    rels = []
    for dp, _, fns in os.walk(images_dir):
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() in exts:
                rel = os.path.relpath(os.path.join(dp, fn), start=images_dir)
                rels.append(Path(rel).as_posix())
    rels.sort()
    return rels


@app.tool(
    output="mineru_dir,parse_file_path,text_corpus_save_path,image_corpus_save_path->None"
)
async def build_mineru_corpus(
    mineru_dir: str,
    parse_file_path: str,
    text_corpus_save_path: str,
    image_corpus_save_path: str,
) -> None:
    import os, shutil
    from typing import List, Dict, Any, Set
    from fastmcp.exceptions import ToolError
    from PIL import Image

    root = os.path.abspath(mineru_dir)
    if not os.path.isdir(root):
        err_msg = f"MinerU root not found: {root}"
        app.logger.error(err_msg)
        raise ToolError(err_msg)
    if not parse_file_path:
        err_msg = "`parse_file_path` cannot be empty."
        app.logger.error(err_msg)
        raise ToolError(err_msg)
    in_path = os.path.abspath(parse_file_path)
    if not os.path.exists(in_path):
        err_msg = f"Input path not found: {in_path}"
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    stems: List[str] = []
    if os.path.isfile(in_path):
        if not in_path.lower().endswith(".pdf"):
            err_msg = f"Only .pdf supported for file input: {in_path}"
            app.logger.error(err_msg)
            raise ToolError(err_msg)
        stems = [os.path.splitext(os.path.basename(in_path))[0]]
    else:
        seen: Set[str] = set()
        for dp, _, fns in os.walk(in_path):
            for fn in sorted(fns):
                if fn.lower().endswith(".pdf"):
                    stem = os.path.splitext(fn)[0]
                    if stem not in seen:
                        stems.append(stem)
                        seen.add(stem)
        stems.sort()
        if not stems:
            err_msg = f"No PDF files found under: {in_path}"
            app.logger.error(err_msg)
            raise ToolError(err_msg)

    text_rows: List[Dict[str, Any]] = []
    image_rows: List[Dict[str, Any]] = []
    image_out = os.path.abspath(image_corpus_save_path)
    out_root_dir = os.path.dirname(image_out)
    base_out_img_dir = os.path.join(out_root_dir, "images")
    os.makedirs(base_out_img_dir, exist_ok=True)

    for stem in stems:
        auto_dir = os.path.join(root, stem, "auto")
        if not os.path.isdir(auto_dir):
            warn_msg = f"Auto dir not found for '{stem}': {auto_dir} (skip)"
            app.logger.warning(warn_msg)
            continue

        md_path = os.path.join(auto_dir, f"{stem}.md")
        if not os.path.isfile(md_path):
            warn_msg = f"Markdown not found for '{stem}': {md_path} (skip text)"
            app.logger.warning(warn_msg)
        else:
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read().strip()
            text_rows.append({"id": stem, "title": stem, "contents": md_text})

        images_dir = os.path.join(auto_dir, "images")
        if not os.path.isdir(images_dir):
            warn_msg = f"No images dir for '{stem}': {images_dir} (skip images)"
            app.logger.warning(warn_msg)
            continue

        rel_list = _list_images(images_dir)
        for idx, rel in enumerate(rel_list):
            src = os.path.join(images_dir, rel)
            dst = os.path.join(base_out_img_dir, stem, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            try:
                with Image.open(src) as im:
                    im.convert("RGB").copy()
            except Exception as e:
                warn_msg = f"Skip invalid image for '{stem}': {src}, reason: {e}"
                app.logger.warning(warn_msg)
                continue

            shutil.copy2(src, dst)
            image_rows.append(
                {
                    "id": len(image_rows),
                    "image_id": Path(os.path.join(stem, rel)).as_posix(),
                    "image_path": Path(os.path.join("images", stem, rel)).as_posix(),
                }
            )

    text_out = os.path.abspath(text_corpus_save_path)
    _save_jsonl(text_rows, text_out)
    _save_jsonl(image_rows, image_out)

    info_msg = (
        f"Built MinerU corpus from {in_path} | docs={len(stems)} | "
        f"text_rows={len(text_rows)} | image_rows={len(image_rows)}\n"
        f"Text corpus -> {text_out}\n"
        f"Image corpus -> {image_out} (images root: {base_out_img_dir})"
    )
    app.logger.info(info_msg)


@app.tool(
    output="raw_chunk_path,chunk_backend_configs,chunk_backend,chunk_path,use_title->None"
)
async def chunk_documents(
    raw_chunk_path: str,
    chunk_backend_configs: Dict[str, Any],
    chunk_backend: str = "token",
    chunk_path: Optional[str] = None,
    use_title: bool = True,
) -> None:

    try:
        import chonkie
        # default uv install chonkie version is 1.3.1, need to check for 1.4.0+
        chonkie_ver = getattr(chonkie, "__version__", "")
        is_chonkie_140 = chonkie_ver.startswith("1.4.0")
    except ImportError:
        err_msg = "chonkie not installed. Please `pip install chonkie`."
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    if chunk_path is None:
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        output_dir = os.path.join(project_root, "output", "corpus")
        chunk_path = os.path.join(output_dir, "chunks.jsonl")
    else:
        chunk_path = str(chunk_path)
        output_dir = os.path.dirname(chunk_path)
    os.makedirs(output_dir, exist_ok=True)

    documents = _load_jsonl(raw_chunk_path)

    cfg = (chunk_backend_configs.get(chunk_backend) or {}).copy()
    if chunk_backend == "token":
        from chonkie import TokenChunker
        import tiktoken

        tokenizer_name = cfg.get("tokenizer_or_token_counter")
        if not tokenizer_name:
            err_msg = "`tokenizer_or_token_counter` is required for token chunking."
            app.logger.error(err_msg)
            raise ToolError(err_msg)
        if tokenizer_name not in ["word", "character"]:
            tokenizer = tiktoken.get_encoding(tokenizer_name)
        else:
            tokenizer = tokenizer_name
        chunk_size = cfg.get("chunk_size")
        if not chunk_size:
            err_msg = "`chunk_size` is required for token chunking."
            app.logger.error(err_msg)
            raise ToolError(err_msg)
        chunk_overlap = cfg.get("chunk_overlap")
        if not chunk_overlap:
            err_msg = "`chunk_overlap` is required for token chunking."
            app.logger.error(err_msg)
            raise ToolError(err_msg)

        chunker = TokenChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif chunk_backend == "sentence":
        from chonkie import SentenceChunker
        import tiktoken

        tokenizer_name = cfg.get("tokenizer_or_token_counter")
        if not tokenizer_name:
            err_msg = "`tokenizer_or_token_counter` is required for sentence chunking."
            app.logger.error(err_msg)
            raise ToolError(err_msg)
        if tokenizer_name not in ["word", "character"]:
            tokenizer = tiktoken.get_encoding(tokenizer_name)
        else:
            tokenizer = tokenizer_name
        chunk_size = cfg.get("chunk_size")
        if not chunk_size:
            err_msg = "`chunk_size` is required for sentence chunking."
            app.logger.error(err_msg)
            raise ToolError(err_msg)
        chunk_overlap = cfg.get("chunk_overlap")
        if not chunk_overlap:
            err_msg = "`chunk_overlap` is required for sentence chunking."
            app.logger.error(err_msg)
            raise ToolError(err_msg)
        min_sentences_per_chunk = cfg.get("min_sentences_per_chunk")
        if not min_sentences_per_chunk:
            err_msg = "`min_sentences_per_chunk` is required for sentence chunking."
            app.logger.error(err_msg)
            raise ToolError(err_msg)

        delim = cfg.get("delim")
        DELIM_DEFAULT = [".", "!", "?", "；", "。", "！", "？"]
        if isinstance(delim, str):
            try:
                delim = ast.literal_eval(delim)
            except Exception:
                delim = DELIM_DEFAULT
        elif delim is None:
            delim = DELIM_DEFAULT

        if is_chonkie_140:
            chunker = SentenceChunker(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_sentences_per_chunk=min_sentences_per_chunk,
                delim=delim,
            )
        else:
            chunker = SentenceChunker(
                tokenizer_or_token_counter=tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_sentences_per_chunk=min_sentences_per_chunk,
                delim=delim,
            )
            
    elif chunk_backend == "recursive":
        from chonkie import RecursiveChunker, RecursiveRules
        import tiktoken

        tokenizer_name = cfg.get("tokenizer_or_token_counter")
        if not tokenizer_name:
            err_msg = "`tokenizer_or_token_counter` is required for recursive chunking."
            app.logger.error(err_msg)
            raise ToolError(err_msg)
        if tokenizer_name not in ["word", "character"]:
            tokenizer = tiktoken.get_encoding(tokenizer_name)
        else:
            tokenizer = tokenizer_name
        chunk_size = cfg.get("chunk_size")
        if not chunk_size:
            err_msg = "`chunk_size` is required for recursive chunking."
            app.logger.error(err_msg)
            raise ToolError(err_msg)
        min_characters_per_chunk = cfg.get("min_characters_per_chunk")
        if not min_characters_per_chunk:
            err_msg = "`min_characters_per_chunk` is required for recursive chunking."
            app.logger.error(err_msg)
            raise ToolError(err_msg)
        if is_chonkie_140:
            chunker = RecursiveChunker(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                rules=RecursiveRules(),
                min_characters_per_chunk=min_characters_per_chunk,
            )
        else:
            chunker = RecursiveChunker(
                tokenizer_or_token_counter=tokenizer,
                chunk_size=chunk_size,
                rules=RecursiveRules(),
                min_characters_per_chunk=min_characters_per_chunk,
            )
            
    else:
        err_msg = (
            f"Invalid chunking method: {chunk_backend}. "
            "Supported: token, sentence, recursive."
        )
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    chunked_documents = []
    current_chunk_id = 0
    for doc in tqdm(documents, desc=f"Chunking ({chunk_backend})", unit="doc"):
        doc_id = doc.get("id") or ""
        title = (doc.get("title") or "").strip()
        text = (doc.get("contents") or "").strip()

        if not text:
            warn_msg = f"doc_id={doc_id} has no contents, skipped."
            app.logger.warning(warn_msg)
            continue
        try:
            chunks = chunker.chunk(text)
        except Exception as e:
            err_msg = f"fail chunked(doc_id={doc_id}): {e}"
            app.logger.error(err_msg)
            raise ToolError(err_msg)

        for chunk in chunks:
            if use_title:
                contents = title + "\n" + chunk.text
            else:
                contents = chunk.text
            meta_chunk = {
                "id": current_chunk_id,
                "doc_id": doc_id,
                "title": title,
                "contents": contents.strip(),
            }
            chunked_documents.append(meta_chunk)
            current_chunk_id += 1

    _save_jsonl(chunked_documents, chunk_path)


if __name__ == "__main__":
    app.run(transport="stdio")
