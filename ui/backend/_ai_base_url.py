"""SSRF guards for the user-supplied AI provider ``baseUrl``.

The ``/api/ai/test`` and ``/api/ai/chat`` endpoints accept a ``baseUrl`` from
the request body and use it directly to construct outbound HTTP requests.
Without validation, a caller can point UltraRAG at:

* loopback / private addresses on the host running UltraRAG,
* cloud instance metadata services (AWS / GCP / Azure ``169.254.169.254``,
  link-local ``fe80::/10``),
* arbitrary internal hosts unreachable from the public internet.

This module exposes :func:`validate_ai_base_url`, used by both endpoints.

Default policy (matches self-hosted RAG expectations):

* Only ``http://`` and ``https://`` schemes are accepted.
* Link-local, multicast, reserved and unspecified addresses are always
  rejected (closing the IMDS attack from issue #393).
* Loopback and RFC1918 private addresses are allowed by default so that
  Ollama / vLLM / LM Studio at ``localhost`` keep working.
* All A/AAAA records resolved for the hostname are checked, defending
  against DNS rebinding where one record points at a public IP and a
  second points at IMDS.

Two opt-in env vars tighten the policy for production deployments:

* ``ULTRARAG_AI_BASE_URL_BLOCK_PRIVATE=1`` — also reject loopback /
  private / shared / site-local addresses.
* ``ULTRARAG_AI_BASE_URL_ALLOWLIST=api.openai.com,api.anthropic.com,...``
  — only hostnames in this CSV are accepted; everything else fails fast
  before DNS resolution.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Optional, Sequence
from urllib.parse import urlparse

ALLOWED_SCHEMES = ("http", "https")


def _is_unsafe_address(ip: ipaddress._BaseAddress, *, block_private: bool) -> bool:
    """Decide whether ``ip`` should be blocked from outbound AI requests.

    Order matters:

    1. Always reject link-local, multicast and the unspecified address —
       these are never legitimate AI provider destinations and link-local
       in particular is the IMDS attack from issue #393.
    2. In ``block_private`` (strict) mode, also reject loopback and
       RFC1918 private — operators set this when the host has no
       legitimate sibling AI service.
    3. Otherwise, *allow* loopback and private explicitly so that
       Ollama / vLLM / LM Studio at ``localhost`` keep working. We do
       this before the ``is_reserved`` check because in Python 3.12
       ``IPv6Address('::1').is_reserved`` is True (``::1`` sits inside
       the reserved ``0::/8`` block) — without this short-circuit,
       legitimate IPv6 loopback would be rejected.
    4. Reject any other reserved address (e.g. IPv4 240.0.0.0/4).
    """
    if ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        return True
    if block_private and (ip.is_loopback or ip.is_private):
        return True
    if ip.is_loopback or ip.is_private:
        return False
    if ip.is_reserved:
        return True
    return False


def _resolve_host(host: str) -> Sequence[ipaddress._BaseAddress]:
    """Return all A/AAAA addresses for ``host``.

    Raises ``socket.gaierror`` on resolution failure so the caller can map
    it to a user-facing error.
    """
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    addrs: list[ipaddress._BaseAddress] = []
    for info in infos:
        sockaddr = info[4]
        # IPv4: (ip, port) ; IPv6: (ip, port, flowinfo, scopeid)
        ip_str = sockaddr[0]
        try:
            addrs.append(ipaddress.ip_address(ip_str))
        except ValueError:
            continue
    return addrs


def _read_allowlist() -> Optional[set[str]]:
    raw = os.environ.get("ULTRARAG_AI_BASE_URL_ALLOWLIST", "")
    items = {h.strip().lower() for h in raw.split(",") if h.strip()}
    return items or None


def _read_block_private() -> bool:
    return os.environ.get("ULTRARAG_AI_BASE_URL_BLOCK_PRIVATE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def validate_ai_base_url(base_url: str) -> Optional[str]:
    """Return ``None`` when ``base_url`` is safe to fetch, else a reason string.

    The returned string is suitable for surfacing in an API ``error`` field.
    """
    if not base_url or not isinstance(base_url, str):
        return "baseUrl is required"

    parsed = urlparse(base_url.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return (
            f"baseUrl scheme must be http or https (got {parsed.scheme or 'empty'!r})"
        )

    host = parsed.hostname
    if not host:
        return "baseUrl is missing a hostname"

    allowlist = _read_allowlist()
    if allowlist is not None and host.lower() not in allowlist:
        return f"baseUrl host is not in ULTRARAG_AI_BASE_URL_ALLOWLIST: {host}"

    block_private = _read_block_private()

    # If the host is already an IP literal, validate it directly without
    # touching DNS — bracketed IPv6 hosts come back from urlparse without
    # the brackets, so ip_address() is happy.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        if _is_unsafe_address(literal, block_private=block_private):
            return f"baseUrl host {host} resolves to a disallowed address ({literal})"
        return None

    try:
        addrs = _resolve_host(host)
    except socket.gaierror as exc:
        return f"baseUrl host {host} could not be resolved: {exc}"

    if not addrs:
        return f"baseUrl host {host} did not resolve to any address"

    for ip in addrs:
        if _is_unsafe_address(ip, block_private=block_private):
            return f"baseUrl host {host} resolves to a disallowed address ({ip})"

    return None
