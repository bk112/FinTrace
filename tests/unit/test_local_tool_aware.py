"""Tests for local Tool-aware routing, evidence formatting, and cache reuse."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fintrace.tools.local_tool_aware import LocalToolAwareConfig, LocalToolAwareRetriever
from fintrace.tools.local_web import EvidenceDocument, SearchHit


class FakeSearchProvider:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls = 0

    def search(self, query: str, limit: int) -> list[SearchHit]:
        self.calls += 1
        return self.hits[:limit]


class FakePageReader:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def read(self, hit: SearchHit) -> EvidenceDocument:
        self.calls.append(hit.url)
        return EvidenceDocument(title=hit.title, url=hit.url, text=f"evidence for {hit.title}")


class LocalToolAwareTest(unittest.TestCase):
    def test_prefers_financial_domain_and_reuses_cached_evidence(self) -> None:
        search = FakeSearchProvider(
            [
                SearchHit(title="secondary", url="https://example.com/report"),
                SearchHit(title="exchange", url="https://www.sse.com.cn/disclosure"),
            ]
        )
        reader = FakePageReader()
        with tempfile.TemporaryDirectory() as directory:
            retriever = LocalToolAwareRetriever(
                LocalToolAwareConfig(cache_dir=Path(directory), document_limit=2),
                search_provider=search,
                page_reader=reader,
            )
            first = retriever.search("company annual report")
            second = retriever.search("company annual report")

        self.assertEqual(search.calls, 1)
        self.assertEqual(reader.calls[0], "https://www.sse.com.cn/disclosure")
        self.assertIn("[source 1] title: exchange", first.text)
        self.assertFalse(first.metadata["cached"])
        self.assertTrue(second.metadata["cached"])

    def test_direct_url_skips_web_search(self) -> None:
        search = FakeSearchProvider([])
        reader = FakePageReader()
        with tempfile.TemporaryDirectory() as directory:
            retriever = LocalToolAwareRetriever(
                LocalToolAwareConfig(cache_dir=Path(directory)),
                search_provider=search,
                page_reader=reader,
            )
            result = retriever.search("https://www.sse.com.cn/disclosure")

        self.assertEqual(search.calls, 0)
        self.assertEqual(reader.calls, ["https://www.sse.com.cn/disclosure"])
        self.assertEqual(result.metadata["strategy"], "page_read")
