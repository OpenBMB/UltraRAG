"""Static transport changes must not alter API or streamed responses."""

import gzip
from pathlib import Path

import pytest
from flask import Flask, Response

from ui.backend.static_assets import prepare_static_response


@pytest.fixture
def app(tmp_path: Path) -> Flask:
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app-abcdefgh.js").write_text(
        "const value = 123;\n" * 500, encoding="utf-8"
    )
    app = Flask(__name__, static_folder=str(tmp_path), static_url_path="")
    app.after_request(prepare_static_response)

    @app.get("/chat", endpoint="chat_page")
    def chat() -> Response:
        return Response("<html>" + "test " * 500 + "</html>", mimetype="text/html")

    @app.get("/api/stream")
    def stream() -> Response:
        return Response(iter(["data: test\n\n"] * 200), mimetype="text/event-stream")

    return app


def test_static_content_negotiates_gzip_and_immutable_caching(app: Flask) -> None:
    with app.test_client() as client:
        plain = client.get("/assets/app-abcdefgh.js")
        encoded = client.get(
            "/assets/app-abcdefgh.js", headers={"Accept-Encoding": "gzip"}
        )
        assert encoded.headers["Content-Encoding"] == "gzip"
        assert gzip.decompress(encoded.data) == plain.data
        assert len(encoded.data) < len(plain.data) / 5
        assert "Accept-Encoding" in encoded.vary
        assert "immutable" in encoded.headers["Cache-Control"]
        assert encoded.get_etag()[1] is True


@pytest.mark.parametrize("accept", ["gzip;q=0", "gzip;q=0.5, identity;q=1", "br"])
def test_static_content_respects_negotiation(app: Flask, accept: str) -> None:
    response = app.test_client().get(
        "/assets/app-abcdefgh.js", headers={"Accept-Encoding": accept}
    )
    assert "Content-Encoding" not in response.headers
    assert "Accept-Encoding" in response.vary


def test_conditional_and_range_requests_still_work(app: Flask) -> None:
    with app.test_client() as client:
        full = client.get(
            "/assets/app-abcdefgh.js", headers={"Accept-Encoding": "gzip"}
        )
        cached = client.get(
            "/assets/app-abcdefgh.js",
            headers={"Accept-Encoding": "gzip", "If-None-Match": full.headers["ETag"]},
        )
        assert cached.status_code == 304
        partial = client.get(
            "/assets/app-abcdefgh.js",
            headers={"Accept-Encoding": "gzip", "Range": "bytes=0-9"},
        )
        assert partial.status_code == 206
        assert "Content-Encoding" not in partial.headers
        assert len(partial.data) == 10


def test_entry_page_revalidates_and_api_stream_is_not_buffered(app: Flask) -> None:
    with app.test_client() as client:
        entry = client.get("/chat", headers={"Accept-Encoding": "gzip"})
        assert entry.headers["Cache-Control"] == "no-cache"
        assert gzip.decompress(entry.data).startswith(b"<html>")
        stream = client.get(
            "/api/stream", headers={"Accept-Encoding": "gzip"}, buffered=False
        )
        assert "Content-Encoding" not in stream.headers
        assert next(iter(stream.response)) == b"data: test\n\n"
