import asyncio
import ast
import json
import os
import shutil
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path
import copy
import re
import random

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

@app.tool(output="input_file_path,input_one_two_label_path->q_ls,label_len,label_str")
async def load_onelabel_data(
    input_file_path: str,
    input_one_two_label_path: str
) -> Dict[str, Any]:

    documents = _load_jsonl(input_file_path)
    labels = _load_jsonl(input_one_two_label_path)
    
    doc_content_list = []
    for doc_obj in documents:
        doc_content_list.append(doc_obj['content'])
    
    one_label_list = []
    for label_obj in labels:
        if label_obj['one_label'] not in one_label_list:
            one_label_list.append(label_obj['one_label'])
    label_len = str(len(one_label_list))
    label_str = ';'.join(one_label_list)

    data = {'q_ls': doc_content_list, 'label_len': label_len, 'label_str':label_str}
    return data

@app.tool(output="input_file_path,input_one_two_label_path,output_onelabel_file_path,ans_ls->q_ls,label_len_list,label_str_list")
async def load_twolabel_data(
    input_file_path: str,
    input_one_two_label_path: str,
    output_onelabel_file_path: str,
    ans_ls: List[str]
) -> Dict[str, List[str]]:

    onelabel_pre_list = []
    for onelabel_pre in ans_ls:
        onelabel_pre_list.append(onelabel_pre.strip())
        
    documents = _load_jsonl(input_file_path)
    labels = _load_jsonl(input_one_two_label_path)
    
    onelabel_has_two_label_dict = {}
    onelabel_list = []
    onelabel_desc_dict = {}
    twolabel_desc_dict = {}
    for label_obj in labels:
        onelabel_desc_dict[label_obj['one_label']] = label_obj['one_label_desc']
        twolabel_desc_dict[label_obj['two_label']] = label_obj['two_label_desc']
        if label_obj['one_label'] not in onelabel_list:
            onelabel_list.append(label_obj['one_label'])
        if label_obj['one_label'] in onelabel_has_two_label_dict:
            if label_obj['two_label'] not in onelabel_has_two_label_dict[label_obj['one_label']]:
                onelabel_has_two_label_dict[label_obj['one_label']].append(label_obj['two_label'])
        else:
            onelabel_has_two_label_dict[label_obj['one_label']] = [label_obj['two_label']]
    
    doc_content_list = []
    two_label_len_list = []
    two_label_str_list = []
    raws = []
    for doci, onelabel in enumerate(onelabel_pre_list):
        not_continue_flag = True
        if onelabel in onelabel_list:
            not_continue_flag = False
        else:
            for origin_onelabel in onelabel_list:
                if origin_onelabel.find(onelabel) != -1 or onelabel.find(origin_onelabel) != -1:
                    onelabel = origin_onelabel
                    not_continue_flag = False
        if not_continue_flag:
            continue
        two_label_list = onelabel_has_two_label_dict[onelabel]
        two_label_len_list.append(str(len(two_label_list)))
        two_label_str_list.append(';'.join(two_label_list))
        content = documents[doci]['content']
        del documents[doci]['content']
        doc_content_list.append(content)
        documents[doci]["id"] = str(doci)
        documents[doci]["contents"] = content
        documents[doci]['one_label'] = onelabel
        documents[doci]['one_label_desc'] = onelabel_desc_dict[onelabel]
        raws.append(documents[doci])
    out_path = os.path.abspath(output_onelabel_file_path)
    _save_jsonl(raws, out_path)

    data = {'q_ls': doc_content_list, 'label_len_list': two_label_len_list, 'label_str_list':two_label_str_list}
    return data
    
@app.tool(output="input_one_two_label_path,output_onelabel_file_path,ans_ls->q_ls,onelabel_prompt_list,onelabel_desc_list")
async def twolabel_result_pro(
    input_one_two_label_path: str,
    output_onelabel_file_path: str,
    ans_ls: List[str]
) -> Dict[str, List[str]]:
    
    # info_msg = (
    #     f"twolabel_result_pro ans_ls: {ans_ls} "
    # )
    # app.logger.info(info_msg)

    twolabel_pre_list = []
    for twolabel_pre in ans_ls:
        twolabel_pre_list.append(twolabel_pre.strip())

    labels = _load_jsonl(input_one_two_label_path)
    documents = _load_jsonl(output_onelabel_file_path)
    
    onelabel_has_two_label_dict = {}
    onelabel_list = []
    onelabel_desc_dict = {}
    twolabel_desc_dict = {}
    for label_obj in labels:
        onelabel_desc_dict[label_obj['one_label']] = label_obj['one_label_desc']
        twolabel_desc_dict[label_obj['two_label']] = label_obj['two_label_desc']
        if label_obj['one_label'] not in onelabel_list:
            onelabel_list.append(label_obj['one_label'])
        if label_obj['one_label'] in onelabel_has_two_label_dict:
            if label_obj['two_label'] not in onelabel_has_two_label_dict[label_obj['one_label']]:
                onelabel_has_two_label_dict[label_obj['one_label']].append(label_obj['two_label'])
        else:
            onelabel_has_two_label_dict[label_obj['one_label']] = [label_obj['two_label']]
    
    doc_content_list = []
    onelabel_prompt_list = []
    onelabel_desc_list = []
    raws = []
    for doc_i, doc_obj in enumerate(documents):
        one_label = doc_obj['one_label']
        two_label_list = onelabel_has_two_label_dict[one_label]

        two_label_flag = True
        twolabel_pre = twolabel_pre_list[doc_i]
        if twolabel_pre in two_label_list:
            doc_obj['two_label'] = twolabel_pre
            doc_obj['two_label_desc'] = twolabel_desc_dict[twolabel_pre]
            doc_obj['two_label_flag'] = 'right'
            two_label_flag = False
        else:
            for origin_twolabel in two_label_list:
                if origin_twolabel.find(twolabel_pre) != -1 or twolabel_pre.find(origin_twolabel) != -1:
                    twolabel_pre = origin_twolabel
                    doc_obj['two_label'] = twolabel_pre
                    doc_obj['two_label_desc'] = twolabel_desc_dict[twolabel_pre]
                    doc_obj['two_label_flag'] = 'right'
                    two_label_flag = False
        if two_label_flag:
            doc_obj['two_label'] = 'No label'
            doc_obj['two_label_desc'] = 'No label'
            doc_obj['two_label_flag'] = 'error'
            doc_content_list.append(doc_obj['contents'])
            onelabel_prompt_list.append(one_label)
            onelabel_desc_list.append(onelabel_desc_dict[one_label])
        raws.append(doc_obj)
    
    out_path = os.path.abspath(output_onelabel_file_path)
    _save_jsonl(raws, out_path)
    
    data = {'q_ls': doc_content_list, 'onelabel_prompt_list': onelabel_prompt_list, 'onelabel_desc_list':onelabel_desc_list}
    return data


@app.tool(output="input_one_two_label_path,output_onelabel_file_path,ans_ls->tag_len_list,tag_str_list")
async def twolabel_new_result_pro(
    input_one_two_label_path: str,
    output_onelabel_file_path: str,
    ans_ls: List[str]
) -> Dict[str, List[str]]:
    twolabel_new_list = []
    for twolabel_new in ans_ls:
        twolabel_new_list.append(twolabel_new.strip())
    
    labels = _load_jsonl(input_one_two_label_path)
    documents = _load_jsonl(output_onelabel_file_path)
    
    onelabel_has_two_label_dict = {}
    onelabel_list = []
    onelabel_desc_dict = {}
    twolabel_desc_dict = {}
    for label_obj in labels:
        onelabel_desc_dict[label_obj['one_label']] = label_obj['one_label_desc']
        twolabel_desc_dict[label_obj['two_label']] = label_obj['two_label_desc']
        if label_obj['one_label'] not in onelabel_list:
            onelabel_list.append(label_obj['one_label'])
        if label_obj['one_label'] in onelabel_has_two_label_dict:
            if label_obj['two_label'] not in onelabel_has_two_label_dict[label_obj['one_label']]:
                onelabel_has_two_label_dict[label_obj['one_label']].append(label_obj['two_label'])
        else:
            onelabel_has_two_label_dict[label_obj['one_label']] = [label_obj['two_label']]
    
    raws = []
    error_num = 0
    onelabel_has_two_label_dict_new = copy.deepcopy(onelabel_has_two_label_dict)
    for doc_i, doc_obj in enumerate(documents):
        one_label = doc_obj['one_label']
        two_label_list = onelabel_has_two_label_dict[one_label]
        if doc_obj['two_label_flag'] == 'error':
            error_num += 1
            new_twolabel = twolabel_new_list[error_num-1]
            if new_twolabel in two_label_list:
                doc_obj['two_label'] = new_twolabel
                doc_obj['two_label_desc'] = twolabel_desc_dict[new_twolabel]
                doc_obj['two_label_flag'] = 'right'
            else:
                if new_twolabel not in onelabel_has_two_label_dict_new[one_label]:
                    onelabel_has_two_label_dict_new[one_label].append(new_twolabel)
        raws.append(doc_obj)
    
    tag_len_list = []
    tag_str_list = []
    for onelabel_kv in onelabel_has_two_label_dict:
        if len(onelabel_has_two_label_dict[onelabel_kv]) != len(onelabel_has_two_label_dict_new[onelabel_kv]):
            tag_list = onelabel_has_two_label_dict_new[onelabel_kv]
            tag_len = len(tag_list)
            tag_str = ';'.join(tag_list)
            tag_len_list.append(str(tag_len))
            tag_str_list.append(tag_str)
    
    out_path = os.path.abspath(output_onelabel_file_path)
    _save_jsonl(raws, out_path)

    data = {'tag_len_list': tag_len_list, 'tag_str_list':tag_str_list}
    return data

@app.tool(output="input_one_two_label_path,output_onelabel_file_path,output_generate_twolabel_file_path,ans_ls->prompt_twolabel_list,prompt_onelabel_list,prompt_onelabel_desc_list")
async def twolabel_merge_result_pro(
    input_one_two_label_path: str,
    output_onelabel_file_path: str,
    output_generate_twolabel_file_path: str,
    ans_ls: List[str]
) -> Dict[str, List[str]]:
    merge_label_json_list = []
    for merge_label_json in ans_ls:
        result = merge_label_json.strip()
        json_match = re.search(r'```(?:json)?\s*(\[{.*?}\]|{.*?})\s*```', result, re.DOTALL)
        if json_match:
            result = json_match.group(1)
        result = result.replace("'", '"')
        result = result.replace('\"s ', '\'s ')
        merge_label_json_list.append(result)

    labels = _load_jsonl(input_one_two_label_path)
    documents = _load_jsonl(output_onelabel_file_path)
    
    onelabel_has_two_label_dict = {}
    onelabel_list = []
    onelabel_desc_dict = {}
    twolabel_desc_dict = {}
    for label_obj in labels:
        onelabel_desc_dict[label_obj['one_label']] = label_obj['one_label_desc']
        twolabel_desc_dict[label_obj['two_label']] = label_obj['two_label_desc']
        if label_obj['one_label'] not in onelabel_list:
            onelabel_list.append(label_obj['one_label'])
        if label_obj['one_label'] in onelabel_has_two_label_dict:
            if label_obj['two_label'] not in onelabel_has_two_label_dict[label_obj['one_label']]:
                onelabel_has_two_label_dict[label_obj['one_label']].append(label_obj['two_label'])
        else:
            onelabel_has_two_label_dict[label_obj['one_label']] = [label_obj['two_label']]

    error_onelabel_list = []
    for doc_i, doc_obj in enumerate(documents):
        one_label = doc_obj['one_label']
        if doc_obj['two_label_flag'] == 'error':
            if one_label not in error_onelabel_list:
                error_onelabel_list.append(one_label)
    
    onelabel_has_two_label_dict_new = copy.deepcopy(onelabel_has_two_label_dict)

    raws = []
    onelabel_result_json_dict = {}
    for doc_i, doc_obj in enumerate(documents):
        one_label = doc_obj['one_label']
        two_label = doc_obj['two_label']
        if one_label not in error_onelabel_list:
            raws.append(doc_obj)
            continue
        one_label_index = error_onelabel_list.index(one_label)
        if one_label_index >= 0:
            result = merge_label_json_list[one_label_index]
            try:
                result_json = json.loads(result)
            except Exception as e:
                result_json = 'Exception_None'
            if result_json != 'Exception_None':
                del onelabel_has_two_label_dict_new[one_label]
                all_new_twolabel_list = []
                exit_new_twolabel_list = []
                no_exit_new_twolabel_list = []
                onelabel_result_json_dict[one_label] = result_json
                for origin_twolabel_kv in result_json:
                    if result_json[origin_twolabel_kv] not in all_new_twolabel_list:
                        all_new_twolabel_list.append(result_json[origin_twolabel_kv])
                        if result_json[origin_twolabel_kv] in twolabel_desc_dict:
                            exit_new_twolabel_list.append(result_json[origin_twolabel_kv])
                        else:
                            no_exit_new_twolabel_list.append(result_json[origin_twolabel_kv])
                onelabel_has_two_label_dict_new[one_label] = all_new_twolabel_list
                if two_label in result_json:
                    new_twolabel = result_json[two_label]
                    if new_twolabel in twolabel_desc_dict:
                        doc_obj['two_label'] = new_twolabel
                        doc_obj['two_label_desc'] = twolabel_desc_dict[new_twolabel]
                        doc_obj['two_label_flag'] = 'right'
                    else:
                        doc_obj['two_label'] = new_twolabel
                        doc_obj['two_label_desc'] = 'None_Desc'
                        doc_obj['two_label_flag'] = 'update'
                else:
                    label_res = random.choice(all_new_twolabel_list)
                    doc_obj['two_label'] = label_res
                    if label_res in twolabel_desc_dict:
                        doc_obj['two_label_desc'] = twolabel_desc_dict[label_res]
                    else:
                        doc_obj['two_label_desc'] = 'None_Desc'
                    doc_obj['two_label_flag'] = 'random_not_in'
            else:
                two_label_list = onelabel_has_two_label_dict[one_label]
                label_res = random.choice(two_label_list)
                doc_obj['two_label'] = label_res
                doc_obj['two_label_desc'] = twolabel_desc_dict[label_res]
                doc_obj['two_label_flag'] = 'random_no_result'
            raws.append(doc_obj)
    
    prompt_twolabel_list = []
    prompt_onelabel_list = []
    prompt_onelabel_desc_list = []

    result_list = []
    for onelabel in onelabel_list:
        for temp_two_label in onelabel_has_two_label_dict_new[onelabel]:
            new_obj = {}
            new_obj['one_label'] = onelabel
            new_obj['one_label_desc'] = onelabel_desc_dict[onelabel]
            new_obj['two_label'] = temp_two_label
            if temp_two_label not in twolabel_desc_dict:
                new_obj['two_label_desc'] = 'None_Desc'
                prompt_twolabel_list.append(temp_two_label)
                prompt_onelabel_list.append(onelabel)
                prompt_onelabel_desc_list.append(onelabel_desc_dict[onelabel])
            else:
                new_obj['two_label_desc'] = twolabel_desc_dict[temp_two_label]
            result_list.append(new_obj)
    
    if len(onelabel_result_json_dict) > 0:
        new_two_label_list = []
        for onelabel in onelabel_result_json_dict:
            result_json = json.loads(result)
            for origin_kv in result_json:
                temp_obj = {}
                temp_obj['one_label'] = onelabel
                temp_obj['origin_two_label'] = origin_kv
                temp_obj['new_two_label'] = result_json[origin_kv]
                new_two_label_list.append(temp_obj)
        out_new_twolabel_path = os.path.abspath(output_generate_twolabel_file_path)
        _save_jsonl(new_two_label_list, out_new_twolabel_path)

    out_path = os.path.abspath(output_onelabel_file_path)
    _save_jsonl(raws, out_path)

    input_path = os.path.abspath(input_one_two_label_path)
    _save_jsonl(result_list, input_path)

    # info_msg = (
    #         f"twolabel_merge_result_pro prompt_twolabel_list prompt_onelabel_list : {prompt_twolabel_list} {prompt_onelabel_list}"
    #     )
    # app.logger.info(info_msg)

    data = {'prompt_twolabel_list': prompt_twolabel_list, 'prompt_onelabel_list': prompt_onelabel_list, 'prompt_onelabel_desc_list': prompt_onelabel_desc_list}

    return data

@app.tool(output="input_one_two_label_path,output_onelabel_file_path,ans_ls->None")
async def twolabel_desc_result_pro(
    input_one_two_label_path: str,
    output_onelabel_file_path: str,
    ans_ls: List[str]
) -> None:

    desc_data_list = []
    for desc_data in ans_ls:
        desc_data_list.append(desc_data.strip())
    
    labels = _load_jsonl(input_one_two_label_path)
    documents = _load_jsonl(output_onelabel_file_path)
    
    onelabel_has_two_label_dict = {}
    onelabel_list = []
    onelabel_desc_dict = {}
    twolabel_desc_dict = {}
    none_desc_num = 0
    twolabel_in_onelabel_dict = {}
    for label_obj in labels:
        onelabel_desc_dict[label_obj['one_label']] = label_obj['one_label_desc']
        if label_obj['two_label_desc'] == 'None_Desc':
            none_desc_num += 1
            label_obj['two_label_desc'] = desc_data_list[none_desc_num-1]
            
        twolabel_desc_dict[label_obj['two_label']] = label_obj['two_label_desc']
        if label_obj['one_label'] not in onelabel_list:
            onelabel_list.append(label_obj['one_label'])
        if label_obj['one_label'] in onelabel_has_two_label_dict:
            if label_obj['two_label'] not in onelabel_has_two_label_dict[label_obj['one_label']]:
                onelabel_has_two_label_dict[label_obj['one_label']].append(label_obj['two_label'])
        else:
            onelabel_has_two_label_dict[label_obj['one_label']] = [label_obj['two_label']]
        if label_obj['two_label'] in twolabel_in_onelabel_dict:
            if label_obj['one_label'] not in twolabel_in_onelabel_dict[label_obj['two_label']]:
                twolabel_in_onelabel_dict[label_obj['two_label']].append(label_obj['one_label'])
        else:
            twolabel_in_onelabel_dict[label_obj['two_label']] = [label_obj['one_label']]
    
    change_twolabel_dict = {}
    for twolabel_temp in twolabel_in_onelabel_dict:
        change_twolabel_dict[twolabel_temp] = twolabel_in_onelabel_dict[twolabel_temp][0]
    
    raws = []
    for doc_i, doc_obj in enumerate(documents):
        two_label = doc_obj['two_label']
        two_label_desc = doc_obj['two_label_desc']
        del doc_obj['two_label_flag']
        if two_label in change_twolabel_dict:
            if doc_obj['one_label'] != change_twolabel_dict[two_label]:
                doc_obj['one_label'] = change_twolabel_dict[two_label]
                doc_obj['one_label_desc'] = onelabel_desc_dict[doc_obj['one_label']]
        if two_label_desc == 'None_Desc':
            doc_obj['two_label_desc'] = twolabel_desc_dict[two_label]
        raws.append(doc_obj)
    
    result_list = []
    for onelabel in onelabel_list:
        for temp_two_label in onelabel_has_two_label_dict[onelabel]:
            new_obj = {}
            no_onelabel_change_flag = True
            if temp_two_label in change_twolabel_dict:
                if onelabel != change_twolabel_dict[temp_two_label]:
                    no_onelabel_change_flag = False
                    new_obj['one_label'] = change_twolabel_dict[temp_two_label]
                    new_obj['one_label_desc'] = onelabel_desc_dict[change_twolabel_dict[temp_two_label]]
                    new_obj['two_label'] = temp_two_label
                    new_obj['two_label_desc'] = twolabel_desc_dict[temp_two_label]
            if no_onelabel_change_flag:
                new_obj['one_label'] = onelabel
                new_obj['one_label_desc'] = onelabel_desc_dict[onelabel]
                new_obj['two_label'] = temp_two_label
                new_obj['two_label_desc'] = twolabel_desc_dict[temp_two_label]
            result_list.append(new_obj)

    out_path = os.path.abspath(output_onelabel_file_path)
    _save_jsonl(raws, out_path)

    input_path = os.path.abspath(input_one_two_label_path)
    _save_jsonl(result_list, input_path)

@app.tool(output="input_one_two_label_path,input_old_data_file_path,output_onelabel_file_path,output_generate_twolabel_file_path->None")
async def merge_new_old_data_pro(
    input_one_two_label_path: str,
    input_old_data_file_path: str,
    output_onelabel_file_path: str,
    output_generate_twolabel_file_path: str
) -> None:

    labels = _load_jsonl(input_one_two_label_path)
    update_labels = _load_jsonl(output_generate_twolabel_file_path)
    documents = _load_jsonl(output_onelabel_file_path)
    old_documents = _load_jsonl(input_old_data_file_path)

    onelabel_has_two_label_dict = {}
    onelabel_list = []
    onelabel_desc_dict = {}
    twolabel_desc_dict = {}
    twolabel_in_onelabel_dict = {}
    for label_obj in labels:
        onelabel_desc_dict[label_obj['one_label']] = label_obj['one_label_desc']
        twolabel_desc_dict[label_obj['two_label']] = label_obj['two_label_desc']
        if label_obj['one_label'] not in onelabel_list:
            onelabel_list.append(label_obj['one_label'])
        if label_obj['one_label'] in onelabel_has_two_label_dict:
            if label_obj['two_label'] not in onelabel_has_two_label_dict[label_obj['one_label']]:
                onelabel_has_two_label_dict[label_obj['one_label']].append(label_obj['two_label'])
        else:
            onelabel_has_two_label_dict[label_obj['one_label']] = [label_obj['two_label']]
        if label_obj['two_label'] in twolabel_in_onelabel_dict:
            if label_obj['one_label'] not in twolabel_in_onelabel_dict[label_obj['two_label']]:
                twolabel_in_onelabel_dict[label_obj['two_label']].append(label_obj['one_label'])
        else:
            twolabel_in_onelabel_dict[label_obj['two_label']] = [label_obj['one_label']]
    
    change_twolabel_dict = {}
    for twolabel_temp in twolabel_in_onelabel_dict:
        change_twolabel_dict[twolabel_temp] = twolabel_in_onelabel_dict[twolabel_temp][0]

    onelabel_twolabel_json_dict = {}
    onelabel_new_twolabel_dict = {}
    for update_label_obj in update_labels:
        if update_label_obj['one_label'] not in onelabel_twolabel_json_dict:
            onelabel_twolabel_json_dict[update_label_obj['one_label']] = {}
            onelabel_twolabel_json_dict[update_label_obj['one_label']][update_label_obj['origin_two_label']] = update_label_obj['new_two_label']
        else:
            onelabel_twolabel_json_dict[update_label_obj['one_label']][update_label_obj['origin_two_label']] = update_label_obj['new_two_label']
        if update_label_obj['one_label'] not in onelabel_new_twolabel_dict:
            onelabel_new_twolabel_dict[update_label_obj['one_label']] = []
            if update_label_obj['new_two_label'] not in onelabel_new_twolabel_dict[update_label_obj['one_label']]:
                onelabel_new_twolabel_dict[update_label_obj['one_label']].append(update_label_obj['new_two_label'])
    
    raws = []
    for old_data in old_documents:
        if old_data['one_label'] in onelabel_new_twolabel_dict:
            replace_two_label = onelabel_twolabel_json_dict[old_data['one_label']]
            if old_data['two_label'] in replace_two_label:
                old_data['two_label'] = replace_two_label[old_data['two_label']]
                old_data['two_label_desc'] = twolabel_desc_dict[old_data['two_label']]
            else:
                label_res = random.choice(onelabel_new_twolabel_dict[old_data['one_label']])
                old_data['two_label'] = label_res
                old_data['two_label_desc'] = twolabel_desc_dict[old_data['two_label']]
        
        if old_data['two_label'] in change_twolabel_dict:
            if old_data['one_label'] != change_twolabel_dict[old_data['two_label']]:
                old_data['one_label'] = change_twolabel_dict[old_data['two_label']]
                old_data['one_label_desc'] = onelabel_desc_dict[old_data['one_label']]

        raws.append(old_data)
    raws.extend(documents)

    out_path = os.path.abspath(input_old_data_file_path)
    _save_jsonl(raws, out_path)


if __name__ == "__main__":
    app.run(transport="stdio")
