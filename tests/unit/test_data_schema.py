"""Tests for the training-data contract."""

from __future__ import annotations

import unittest

from fintrace.data.schema import DatasetValidationError, parse_financial_qa_sample


class FinancialQASchemaTest(unittest.TestCase):
    def test_accepts_documented_record_shape(self) -> None:
        sample = parse_financial_qa_sample(
            {
                "qid": "finance-001",
                "prompt": "Which business segment had the highest revenue growth?",
                "ground_truth": {"target": ["New energy", "新能源"]},
                "source": {"url": "https://example.invalid/report"},
            }
        )

        self.assertEqual(sample.primary_target, "New energy")
        self.assertEqual(sample.targets, ("New energy", "新能源"))

    def test_rejects_scalar_target(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "non-empty list"):
            parse_financial_qa_sample(
                {
                    "qid": "finance-001",
                    "prompt": "Question",
                    "ground_truth": {"target": "not a list"},
                    "source": "report",
                }
            )

    def test_rejects_missing_provenance(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "source is required"):
            parse_financial_qa_sample(
                {
                    "qid": "finance-001",
                    "prompt": "Question",
                    "ground_truth": {"target": ["Answer"]},
                }
            )
