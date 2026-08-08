"""Local Tool-aware retrieval router built from web search and page-reading tools."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from .base import RetrievalResult
from .local_web import (
    DuckDuckGoHtmlSearchProvider,
    EvidenceDocument,
    HtmlPageReader,
    PageReader,
    SearchHit,
    WebRetrievalError,
    WebSearchProvider,
)


@dataclass(frozen=True)
class LocalToolAwareConfig:
    cache_dir: Path
    search_limit: int = 6
    document_limit: int = 3
    preferred_domains: tuple[str, ...] = (
        "cninfo.com.cn",
        "sse.com.cn",
        "szse.cn",
        "hkexnews.hk",
        "sec.gov",
    )


class LocalToolAwareRetriever:
    """Select and execute local web tools, then produce citable text evidence."""

    def __init__(
        self,
        config: LocalToolAwareConfig,
        *,
        search_provider: WebSearchProvider | None = None,
        page_reader: PageReader | None = None,
    ) -> None:
        self._config = config
        self._search_provider = search_provider or DuckDuckGoHtmlSearchProvider()
        self._page_reader = page_reader or HtmlPageReader()
        self._config.cache_dir.mkdir(parents=True, exist_ok=True)

    def search(self, query: str) -> RetrievalResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must be non-empty")
        cache_path = self._cache_path(normalized_query)
        if cache_path.is_file():
            # 强化训练阶段优先重放同一份证据，避免在线网页变化让同一轨迹奖励漂移。
            return self._read_cache(cache_path, normalized_query)

        documents, strategy = self._retrieve(normalized_query)
        if not documents:
            raise WebRetrievalError("no readable evidence documents were found")
        result = RetrievalResult(
            query=normalized_query,
            text=self._format_evidence(documents),
            metadata={
                "strategy": strategy,
                "sources": [{"title": doc.title, "url": doc.url} for doc in documents],
                "cached": False,
            },
        )
        self._write_cache(cache_path, result)
        return result

    def _retrieve(self, query: str) -> tuple[list[EvidenceDocument], str]:
        parsed = urlparse(query)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            # 模型已经给出明确页面时跳过搜索，直接调用 page_read 工具。
            document = self._page_reader.read(SearchHit(title=query, url=query))
            return [document], "page_read"

        # 普通自然语言查询采用 web_search -> page_read 的两步工具链。
        hits = self._search_provider.search(query, self._config.search_limit)
        ranked_hits = sorted(hits, key=self._domain_priority, reverse=True)
        documents: list[EvidenceDocument] = []
        for hit in ranked_hits:
            try:
                documents.append(self._page_reader.read(hit))
            except WebRetrievalError:
                continue
            if len(documents) == self._config.document_limit:
                break
        return documents, "web_search_then_page_read"

    def _domain_priority(self, hit: SearchHit) -> tuple[int, int]:
        hostname = (urlparse(hit.url).hostname or "").lower()
        is_preferred = any(hostname == domain or hostname.endswith(f".{domain}") for domain in self._config.preferred_domains)
        # 财报与公告优先使用交易所/监管披露页，网页长度仅作为稳定的次级排序。
        return (1 if is_preferred else 0, -len(hit.url))

    @staticmethod
    def _format_evidence(documents: list[EvidenceDocument]) -> str:
        blocks = []
        for index, document in enumerate(documents, start=1):
            blocks.append(
                f"[source {index}] title: {document.title}\n"
                f"url: {document.url}\n"
                f"content: {document.text}"
            )
        return "\n\n".join(blocks)

    def _cache_path(self, query: str) -> Path:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return self._config.cache_dir / f"{digest}.json"

    def _read_cache(self, path: Path, query: str) -> RetrievalResult:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return RetrievalResult(
            query=query,
            text=payload["text"],
            metadata={**payload["metadata"], "cached": True},
        )

    def _write_cache(self, path: Path, result: RetrievalResult) -> None:
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "text": result.text,
            "metadata": result.metadata,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
