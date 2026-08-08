"""Tests for the canonical, auditable financial-fact schema."""

from __future__ import annotations

import unittest

from fintrace.knowledge_base import KnowledgeRecordError, parse_knowledge_record


def record(**overrides):
    data = {
        "fact_id": "fact-001",
        "fact": "某公司 2025 年 Q4 营收同比增长 15.3%",
        "source_doc": "某公司 2025 年 Q4 财报",
        "source_url": "https://example.com/report",
        "source_type": "annual_report",
        "document_id": "doc-001",
        "published_at": "2026-03-01",
        "retrieved_at": "2026-08-08T00:00:00Z",
        "raw_content_hash": "sha256:abc",
        "entity": "某公司",
        "entity_type": "company",
        "metric": "营收增速",
        "value_text": "15.3%",
        "value_type": "percentage",
        "value_number": 15.3,
        "unit": "%",
        "period": "2025-Q4",
        "related_entities": ["新能源板块"],
    }
    data.update(overrides)
    return data


class KnowledgeRecordTest(unittest.TestCase):
    def test_preserves_numeric_value_with_its_unit(self) -> None:
        parsed = parse_knowledge_record(record())

        self.assertEqual(parsed.value_number, 15.3)
        self.assertEqual(parsed.unit, "%")
        self.assertEqual(parsed.related_entities, ("新能源板块",))

    def test_accepts_non_numeric_entity_fact(self) -> None:
        parsed = parse_knowledge_record(
            record(
                metric="营收增速最高业务板块",
                value_text="新能源板块",
                value_type="entity",
                value_number=None,
                unit=None,
            )
        )

        self.assertEqual(parsed.value_text, "新能源板块")

    def test_rejects_percentage_without_normalized_number(self) -> None:
        with self.assertRaises(KnowledgeRecordError):
            parse_knowledge_record(record(value_number=None))

    def test_accepts_current_kb_v1_1_fields(self) -> None:
        current_record = record(
            source_url="",
            entity_type=None,
            period=None,
            date="2025-Q4",
            value_type="ratio",
        )
        current_record.pop("entity_type")
        current_record.pop("period")

        parsed = parse_knowledge_record(current_record)

        self.assertEqual(parsed.entity_type, "company")
        self.assertEqual(parsed.period, "2025-Q4")
        self.assertEqual(parsed.value_type, "ratio")
