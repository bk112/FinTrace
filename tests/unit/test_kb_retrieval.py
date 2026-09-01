"""Test suite for financial knowledge-base retrieval service.

Covers:
  - search_records: basic query, empty query, top_k bounds
  - retrieve: text formatting, opt-in metadata embedding
  - sample_seed_facts: random sampling, entity filter, edge cases
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture(scope="module")
def kb():
    """Import the KB module (lazy-loads on first call)."""
    from fintrace.kb import retrieve, sample_seed_facts, search_records

    return search_records, retrieve, sample_seed_facts


class TestSearchRecords:
    def test_basic_query(self, kb):
        search, _, _ = kb
        results = search("银行营收增速", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        for r in results:
            assert "fact" in r
            assert "entity" in r
            assert "value_text" in r
            assert "_score" in r

    def test_empty_query(self, kb):
        search, _, _ = kb
        results = search("", top_k=3)
        assert results == []

    def test_whitespace_query(self, kb):
        search, _, _ = kb
        results = search("   ", top_k=3)
        assert results == []

    def test_top_k_exceeds_total(self, kb):
        search, _, _ = kb
        # Request more than KB size — should return all without error
        results = search("营收", top_k=100000)
        assert len(results) > 0
        assert len(results) < 100000  # should be capped at KB size

    def test_top_k_zero(self, kb):
        search, _, _ = kb
        results = search("营收", top_k=0)
        assert results == []

    def test_result_scores_descending(self, kb):
        search, _, _ = kb
        results = search("贵州茅台净利润", top_k=5)
        scores = [r["_score"] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"scores not descending: {scores}"

    def test_entity_relevant_to_query(self, kb):
        search, _, _ = kb
        results = search("比亚迪汽车", top_k=5)
        entities = [r["entity"] for r in results]
        assert "比亚迪" in entities, f"Expected 比亚迪 in top results, got: {entities}"


class TestRetrieve:
    def test_returns_formatted_text(self, kb):
        _, retrieve_fn, _ = kb
        text = retrieve_fn("银行净利润", top_k=3)
        assert isinstance(text, str)
        assert "银行" in text or "搜索" in text
        # 默认观测只给 entity/metric/value_text/source_doc，不暴露可直接复制的结构化记录。
        assert "<!-- metadata:" not in text

    def test_retrieve_empty_query(self, kb):
        _, retrieve_fn, _ = kb
        text = retrieve_fn("", top_k=3)
        assert "未找到" in text or "query" in text.lower()

    def test_metadata_is_valid_json(self, kb):
        _, retrieve_fn, _ = kb
        text = retrieve_fn("保险营收", top_k=2, include_metadata=True)
        # Extract JSON between comments
        start = text.find("<!-- metadata: ")
        end = text.find(" -->", start)
        assert start >= 0 and end > start, f"No metadata found in: {text[:200]}"
        json_str = text[start + len("<!-- metadata: "):end]
        meta = json.loads(json_str)
        assert "records" in meta
        assert isinstance(meta["records"], list)
        for rec in meta["records"]:
            assert "fact_id" in rec
            assert "fact" in rec
            assert "value_text" in rec
            # Should NOT leak internal _score
            assert "_score" not in rec

    def test_metadata_records_have_v1_1_schema_fields(self, kb):
        _, retrieve_fn, _ = kb
        text = retrieve_fn("宁德时代净利润", top_k=1, include_metadata=True)
        start = text.find("<!-- metadata: ")
        end = text.find(" -->", start)
        json_str = text[start + len("<!-- metadata: "):end]
        meta = json.loads(json_str)
        for rec in meta["records"]:
            assert "value_number" in rec
            assert "value_type" in rec
            assert "unit" in rec
            assert "source_url" in rec
            assert "document_id" in rec
            assert "retrieved_at" in rec


class TestSampleSeedFacts:
    def test_default_sampling(self, kb):
        _, _, sampler = kb
        seeds = sampler(n=1)
        assert len(seeds) == 1
        assert "fact" in seeds[0]

    def test_multiple_samples(self, kb):
        _, _, sampler = kb
        seeds = sampler(n=5)
        assert len(seeds) == 5
        # All should be unique
        ids = [s["fact_id"] for s in seeds]
        assert len(set(ids)) == 5

    def test_entity_filter(self, kb):
        _, _, sampler = kb
        seeds = sampler(n=3, entity_filter="贵州茅台")
        assert len(seeds) > 0
        for s in seeds:
            assert s["entity"] == "贵州茅台"

    def test_nonexistent_entity(self, kb):
        _, _, sampler = kb
        seeds = sampler(n=3, entity_filter="不存在的公司XYZ")
        assert seeds == []

    def test_n_exceeds_pool(self, kb):
        _, _, sampler = kb
        seeds = sampler(n=100000)
        # Should be capped
        assert len(seeds) < 100000

    def test_returned_records_are_copies(self, kb):
        _, _, sampler = kb
        s1 = sampler(n=1, entity_filter="招商银行")
        if s1:
            s1[0]["fact"] = "MODIFIED"
            s2 = sampler(n=1, entity_filter="招商银行")
            if s2:
                assert s2[0]["fact"] != "MODIFIED", "Returned records should be copies"


class TestCrossInterface:
    """Verify search_records and retrieve return consistent results."""

    def test_same_query_same_records(self, kb):
        search, retrieve_fn, _ = kb
        results = search("格力电器毛利率", top_k=2)
        text = retrieve_fn("格力电器毛利率", top_k=2)
        for r in results:
            # 观测正文只暴露 entity/metric/value_text/source_doc；fact_id 走 metadata 通道。
            assert r["entity"] in text, f"entity {r['entity']} missing from retrieve output"
            assert str(r["value_text"]) in text, (
                f"value_text {r['value_text']} missing from retrieve output"
            )


def test_kb_excludes_financecomplexqa_gold_answers():
    """Gold labels must never be indexable evidence for the same benchmark question."""
    from fintrace.kb import search_records

    records = search_records("FinanceComplexQA", top_k=100000)
    assert all("标准答案为" not in record["fact"] for record in records)


class TestKbRetrievalClient:
    """KbRetrievalClient implements the RetrievalClient protocol used by rollout."""

    def test_search_returns_protocol_result(self, kb):
        from fintrace.kb import KbRetrievalClient

        client = KbRetrievalClient(top_k=2)
        result = client.search("贵州茅台净利润")
        # RetrievalClient protocol contract: query / text / metadata
        assert result.query == "贵州茅台净利润"
        assert isinstance(result.text, str) and result.text
        assert isinstance(result.metadata, dict)
        records = result.metadata.get("records")
        assert isinstance(records, list) and len(records) <= 2

    def test_text_matches_retrieve_output(self, kb):
        from fintrace.kb import KbRetrievalClient, retrieve

        client = KbRetrievalClient(top_k=3)
        result = client.search("招商银行营收")
        assert result.text == retrieve("招商银行营收", top_k=3)

    def test_client_omits_inline_metadata_by_default(self, kb):
        """策略可见文本不得包含可复制的结构化记录（fact/value_number/fact_id）。"""
        from fintrace.kb import KbRetrievalClient, retrieve

        client = KbRetrievalClient(top_k=3)
        result = client.search("招商银行营收")
        assert "<!-- metadata:" not in result.text
        # 奖励侧仍然拿到完整 v1.1 记录，检索正确性判定不受影响。
        assert result.metadata["records"]
        assert result.text == retrieve("招商银行营收", top_k=3)

    def test_client_can_inline_metadata_for_debug(self, kb):
        from fintrace.kb import KbRetrievalClient, retrieve

        client = KbRetrievalClient(top_k=3, include_metadata=True)
        result = client.search("招商银行营收")
        assert "<!-- metadata:" in result.text
        assert result.text == retrieve("招商银行营收", top_k=3, include_metadata=True)

    def test_records_match_search_records_output(self, kb):
        from fintrace.kb import KbRetrievalClient, search_records

        client = KbRetrievalClient(top_k=2)
        result = client.search("宁德时代资产负债率")
        direct = search_records("宁德时代资产负债率", top_k=2)
        assert result.metadata["records"] == [
            {k: v for k, v in r.items() if k != "_score"} for r in direct
        ], "client records should equal search_records output without _score"

    def test_records_have_reward_fields(self, kb):
        """Reward function matches value_text/value_number — records must carry them."""
        from fintrace.kb import KbRetrievalClient

        client = KbRetrievalClient(top_k=3)
        result = client.search("贵州茅台销售毛利率")
        for rec in result.metadata["records"]:
            assert "value_text" in rec
            assert "value_number" in rec
            assert "industry" in rec
            assert "_score" not in rec, "client records should not leak _score"

    def test_empty_query(self, kb):
        from fintrace.kb import KbRetrievalClient

        result = KbRetrievalClient(top_k=3).search("   ")
        assert result.metadata["records"] == []
        assert "未找到" in result.text

    def test_frozen_dataclass(self, kb):
        from dataclasses import FrozenInstanceError

        from fintrace.kb import KbRetrievalClient

        client = KbRetrievalClient(top_k=1)
        with pytest.raises(FrozenInstanceError):
            client.top_k = 5
