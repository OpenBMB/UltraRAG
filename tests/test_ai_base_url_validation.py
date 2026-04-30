"""Tests for the SSRF guard around the user-supplied AI provider baseUrl.

Regression tests for issue #393. Without the guard, any caller can point
``/api/ai/test`` or ``/api/ai/chat`` at arbitrary internal hosts (cloud
metadata services, RFC1918 networks, link-local addresses, ...) and have
the UltraRAG backend fetch them.
"""

from __future__ import annotations

import socket

import pytest

from ui.backend._ai_base_url import validate_ai_base_url


# --- helpers ---------------------------------------------------------------


def _patch_resolver(monkeypatch, mapping):
    """Force ``socket.getaddrinfo`` to return the given hostname → IPs map.

    Each value is a list of address strings (v4 and/or v6). Anything not in
    ``mapping`` raises ``socket.gaierror`` so we never hit real DNS.
    """

    def fake_getaddrinfo(host, *_args, **_kwargs):
        if host not in mapping:
            raise socket.gaierror(8, f"nodename nor servname provided ({host})")
        out = []
        for ip in mapping[host]:
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sockaddr = (ip, 0, 0, 0) if family == socket.AF_INET6 else (ip, 0)
            out.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr))
        return out

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test runs with a clean policy by default; tests opt into env knobs."""
    monkeypatch.delenv("ULTRARAG_AI_BASE_URL_BLOCK_PRIVATE", raising=False)
    monkeypatch.delenv("ULTRARAG_AI_BASE_URL_ALLOWLIST", raising=False)


# --- empty / shape errors --------------------------------------------------


@pytest.mark.parametrize("value", ["", None, "   "])
def test_empty_base_url_rejected(value):
    err = validate_ai_base_url(value)
    assert err is not None
    assert any(token in err.lower() for token in ("required", "scheme", "hostname"))


@pytest.mark.parametrize(
    "value",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "data:text/plain,hi",
        "gopher://example.com/",
        "javascript:alert(1)",
        "//api.openai.com/v1",  # missing scheme
    ],
)
def test_non_http_scheme_rejected(value):
    err = validate_ai_base_url(value)
    assert err is not None
    assert "scheme" in err.lower() or "hostname" in err.lower()


def test_missing_hostname_rejected():
    # urlparse can produce empty hostname for things like "http:///foo"
    err = validate_ai_base_url("http:///models")
    assert err is not None


# --- IP literals (the core SSRF scenarios) ---------------------------------


@pytest.mark.parametrize(
    "url",
    [
        # Issue #393's exact attack: AWS / GCP / Azure IMDS.
        "http://169.254.169.254/latest",
        "https://169.254.169.254/computeMetadata/v1/",
        # Other link-local IPv4.
        "http://169.254.5.5/",
    ],
)
def test_imds_and_link_local_v4_rejected(url):
    err = validate_ai_base_url(url)
    assert err is not None
    assert "disallowed" in err.lower()


def test_link_local_v6_literal_rejected():
    err = validate_ai_base_url("http://[fe80::1]/")
    assert err is not None
    assert "disallowed" in err.lower()


@pytest.mark.parametrize(
    "url",
    [
        "http://224.0.0.1/",  # multicast
        "http://0.0.0.0/",  # unspecified
        "http://[::]/",  # unspecified v6
        "http://[ff02::1]/",  # v6 multicast
    ],
)
def test_multicast_and_unspecified_rejected(url):
    err = validate_ai_base_url(url)
    assert err is not None


def test_loopback_allowed_by_default(monkeypatch):
    """Self-hosted RAG users run Ollama / vLLM / LM Studio on localhost."""
    _patch_resolver(monkeypatch, {})
    assert validate_ai_base_url("http://127.0.0.1:11434/v1") is None
    assert validate_ai_base_url("http://[::1]:11434/v1") is None


def test_private_v4_allowed_by_default(monkeypatch):
    _patch_resolver(monkeypatch, {})
    assert validate_ai_base_url("http://10.0.0.5/v1") is None
    assert validate_ai_base_url("http://192.168.1.10/v1") is None
    assert validate_ai_base_url("http://172.16.0.1/v1") is None


# --- DNS resolution (anti-rebinding) ---------------------------------------


def test_public_hostname_allowed(monkeypatch):
    _patch_resolver(monkeypatch, {"api.openai.com": ["104.18.6.192"]})
    assert validate_ai_base_url("https://api.openai.com/v1") is None


def test_hostname_resolving_to_imds_rejected(monkeypatch):
    """DNS rebinding: attacker controls a public hostname, points it at IMDS."""
    _patch_resolver(monkeypatch, {"evil.example.com": ["169.254.169.254"]})
    err = validate_ai_base_url("https://evil.example.com/v1")
    assert err is not None
    assert "disallowed" in err.lower()


def test_hostname_with_one_safe_one_unsafe_record_rejected(monkeypatch):
    """If ANY resolved address is unsafe, reject — defends against
    DNS rebinding where the response rotates between a public IP and
    an internal one."""
    _patch_resolver(
        monkeypatch,
        {"mixed.example.com": ["1.1.1.1", "169.254.169.254"]},
    )
    err = validate_ai_base_url("https://mixed.example.com/")
    assert err is not None


def test_hostname_with_aaaa_link_local_rejected(monkeypatch):
    _patch_resolver(monkeypatch, {"v6evil.example.com": ["fe80::1"]})
    err = validate_ai_base_url("https://v6evil.example.com/")
    assert err is not None


def test_unresolvable_hostname_rejected(monkeypatch):
    _patch_resolver(monkeypatch, {})
    err = validate_ai_base_url("https://nope.nonexistent.invalid/v1")
    assert err is not None
    assert "could not be resolved" in err.lower()


# --- env-var policy knobs --------------------------------------------------


def test_block_private_mode_rejects_loopback(monkeypatch):
    monkeypatch.setenv("ULTRARAG_AI_BASE_URL_BLOCK_PRIVATE", "1")
    _patch_resolver(monkeypatch, {})
    err = validate_ai_base_url("http://127.0.0.1:11434/v1")
    assert err is not None


def test_block_private_mode_rejects_rfc1918_via_dns(monkeypatch):
    monkeypatch.setenv("ULTRARAG_AI_BASE_URL_BLOCK_PRIVATE", "1")
    _patch_resolver(monkeypatch, {"intranet.local": ["10.0.0.5"]})
    err = validate_ai_base_url("http://intranet.local/v1")
    assert err is not None


def test_block_private_mode_still_allows_public(monkeypatch):
    monkeypatch.setenv("ULTRARAG_AI_BASE_URL_BLOCK_PRIVATE", "1")
    _patch_resolver(monkeypatch, {"api.openai.com": ["104.18.6.192"]})
    assert validate_ai_base_url("https://api.openai.com/v1") is None


def test_allowlist_mode_accepts_listed_hostname(monkeypatch):
    monkeypatch.setenv(
        "ULTRARAG_AI_BASE_URL_ALLOWLIST",
        "api.openai.com, api.anthropic.com",
    )
    _patch_resolver(monkeypatch, {"api.openai.com": ["104.18.6.192"]})
    assert validate_ai_base_url("https://api.openai.com/v1") is None


def test_allowlist_mode_rejects_unlisted_hostname(monkeypatch):
    monkeypatch.setenv(
        "ULTRARAG_AI_BASE_URL_ALLOWLIST",
        "api.openai.com",
    )
    err = validate_ai_base_url("https://api.anthropic.com/v1")
    assert err is not None
    assert "ULTRARAG_AI_BASE_URL_ALLOWLIST" in err


def test_allowlist_mode_runs_before_dns(monkeypatch):
    """Allowlist short-circuits — no DNS lookup for unlisted hosts.

    Important so that a hostile caller can't trigger DNS exfiltration
    or pin the worker thread on a slow resolver in strict-mode deployments.
    """
    called = {"n": 0}

    def boom(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("DNS resolver must not be called")

    monkeypatch.setenv("ULTRARAG_AI_BASE_URL_ALLOWLIST", "api.openai.com")
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    err = validate_ai_base_url("https://attacker.example.com/")
    assert err is not None
    assert called["n"] == 0


def test_allowlist_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("ULTRARAG_AI_BASE_URL_ALLOWLIST", "api.openai.com")
    _patch_resolver(monkeypatch, {"api.openai.com": ["104.18.6.192"]})
    assert validate_ai_base_url("https://API.OpenAI.com/v1") is None
