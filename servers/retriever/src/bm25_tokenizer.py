from __future__ import annotations

from typing import Any, Callable, Optional

CHINESE_LANG_ALIASES = frozenset(
    {
        "zh",
        "zh-cn",
        "zh-hans",
        "zh-hant",
        "zh-tw",
        "cn",
        "chinese",
    }
)

SUPPORTED_BM25_TOKENIZERS = frozenset(
    {"auto", "default", "regex", "jieba", "character", "char"}
)


def _normalize_lang(lang: Any) -> Any:
    if not isinstance(lang, str):
        return lang
    return lang.strip().lower()


def is_chinese_bm25_lang(lang: Any) -> bool:
    return _normalize_lang(lang) in CHINESE_LANG_ALIASES


def character_splitter(text: str) -> list[str]:
    return [char for char in text if char.strip()]


def build_bm25_splitter(
    lang: Any,
    tokenizer_mode: str = "auto",
    *,
    logger: Any = None,
) -> Optional[Callable[[str], list[str]]]:
    normalized_mode = str(tokenizer_mode or "auto").strip().lower()
    if normalized_mode not in SUPPORTED_BM25_TOKENIZERS:
        supported = ", ".join(sorted(SUPPORTED_BM25_TOKENIZERS))
        raise ValueError(
            f"Unsupported BM25 tokenizer mode '{tokenizer_mode}'. Supported modes: {supported}."
        )

    if normalized_mode in {"default", "regex"}:
        return None

    if normalized_mode in {"character", "char"}:
        return character_splitter

    if not is_chinese_bm25_lang(lang):
        return None

    try:
        import jieba
    except ImportError as exc:
        if normalized_mode == "jieba":
            raise ImportError(
                "BM25 Chinese tokenization requires `jieba`. "
                "Install it with `pip install jieba` or `uv sync --extra retriever`."
            ) from exc

        if logger is not None:
            logger.warning(
                "[bm25] `jieba` is not installed. Falling back to character-based tokenization for Chinese."
            )
        return character_splitter

    if logger is not None:
        logger.info("[bm25] Using jieba search tokenization for Chinese.")

    def jieba_splitter(text: str) -> list[str]:
        return [token.strip() for token in jieba.cut_for_search(text) if token.strip()]

    return jieba_splitter
