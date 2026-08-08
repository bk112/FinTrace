"""No-key web search and page-reading primitives for local reproduction."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class EvidenceDocument:
    title: str
    url: str
    text: str
    snippet: str = ""


class WebSearchProvider(Protocol):
    def search(self, query: str, limit: int) -> list[SearchHit]:
        """Return ranked web-search candidates."""


class PageReader(Protocol):
    def read(self, hit: SearchHit) -> EvidenceDocument:
        """Read a candidate page into bounded plain text."""


class WebRetrievalError(RuntimeError):
    """Raised for expected search, networking, or page-parsing failures."""


class _DuckDuckGoResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hits: list[SearchHit] = []
        self._in_result_link = False
        self._href = ""
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        classes = attributes.get("class", "") or ""
        if "result__a" in classes:
            self._in_result_link = True
            self._href = attributes.get("href", "") or ""
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._in_result_link:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._in_result_link:
            return
        title = " ".join("".join(self._parts).split())
        url = _unwrap_duckduckgo_url(self._href)
        if title and url:
            self.hits.append(SearchHit(title=title, url=url))
        self._in_result_link = False


def _unwrap_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        destination = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(destination)
    return url


class DuckDuckGoHtmlSearchProvider:
    """A no-key provider intended for local smoke runs, not production traffic."""

    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(self, *, timeout_seconds: float = 15.0, user_agent: str = "FinTrace/0.1") -> None:
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent

    def search(self, query: str, limit: int) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        request = Request(
            f"{self.endpoint}?q={quote_plus(query)}",
            headers={"User-Agent": self._user_agent},
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                html = response.read(2_000_000).decode("utf-8", errors="replace")
        except OSError as exc:
            raise WebRetrievalError(f"web search failed: {exc}") from exc

        parser = _DuckDuckGoResultParser()
        parser.feed(html)
        return parser.hits[:limit]


class _VisibleTextParser(HTMLParser):
    _ignored_tags = {"script", "style", "noscript", "svg", "template"}

    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._ignored_tags:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self._parts).split())


def _is_public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        return False
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, None)}
    except OSError:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return True


class HtmlPageReader:
    """Fetch a public HTTP(S) page and extract bounded visible text."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        max_bytes: int = 2_000_000,
        max_characters: int = 6_000,
        user_agent: str = "FinTrace/0.1",
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._max_bytes = max_bytes
        self._max_characters = max_characters
        self._user_agent = user_agent

    def read(self, hit: SearchHit) -> EvidenceDocument:
        if not _is_public_http_url(hit.url):
            raise WebRetrievalError(f"refusing unsafe or unsupported URL: {hit.url}")
        request = Request(hit.url, headers={"User-Agent": self._user_agent})
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                content_type = response.headers.get_content_type()
                if content_type not in {"text/html", "text/plain"}:
                    raise WebRetrievalError(f"unsupported content type: {content_type}")
                body = response.read(self._max_bytes).decode("utf-8", errors="replace")
        except OSError as exc:
            raise WebRetrievalError(f"page read failed: {exc}") from exc

        if content_type == "text/plain":
            text = " ".join(body.split())
        else:
            parser = _VisibleTextParser()
            parser.feed(body)
            text = parser.text()
        if not text:
            raise WebRetrievalError("page has no extractable text")
        return EvidenceDocument(
            title=hit.title,
            url=hit.url,
            text=text[: self._max_characters],
            snippet=hit.snippet,
        )
