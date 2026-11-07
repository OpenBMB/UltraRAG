"""Caching and negotiated compression for the bundled frontend only."""

import gzip
import re
from functools import lru_cache

from flask import Response, request

_HASHED_ASSET = re.compile(r"^/assets/[^/]+-[\w-]{8,}\.[\w]+$")
_TEXT_TYPES = {
    "text/html",
    "text/css",
    "text/javascript",
    "application/javascript",
    "image/svg+xml",
}


@lru_cache(maxsize=16)
def _gzip_content(data: bytes) -> bytes:
    """Bound the cache and avoid recompressing identical assets per request."""
    return gzip.compress(data, compresslevel=6, mtime=0)


def prepare_static_response(response: Response) -> Response:
    """Keep APIs/SSE untouched and negotiate gzip without requiring a proxy."""
    if request.endpoint not in {
        "static",
        "index",
        "chat_page",
        "settings_page",
        "favicon",
    }:
        return response
    if response.status_code not in {200, 206, 304}:
        return response
    if request.path.startswith("/assets/") and _HASHED_ASSET.fullmatch(request.path):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    else:
        # Entry pages and unversioned theme files must revalidate after upgrades.
        response.headers["Cache-Control"] = "no-cache"
    if response.mimetype not in _TEXT_TYPES:
        return response
    response.vary.add("Accept-Encoding")
    if (
        request.method != "GET"
        or response.status_code != 200
        or response.headers.get("Content-Encoding")
        or "no-transform" in response.headers.get("Cache-Control", "")
        or request.accept_encodings["gzip"] <= 0
        or request.accept_encodings["identity"] > request.accept_encodings["gzip"]
    ):
        return response
    response.direct_passthrough = False
    data = response.get_data()
    if len(data) < 1024:
        return response
    compressed = _gzip_content(data)
    if len(compressed) >= len(data):
        return response
    response.set_data(compressed)
    response.headers["Content-Encoding"] = "gzip"
    etag, _weak = response.get_etag()
    if etag:
        # Both representations have equivalent content, but different bytes.
        response.set_etag(etag, weak=True)
    return response
