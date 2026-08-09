"""Tests for conservative RL synthesis target consistency checks."""

from __future__ import annotations

import unittest

from fintrace.data.target_validation import find_target_alignment_issue


def sample_record(prompt: str, *, verified: bool = True) -> dict:
    return {
        "qid": "ft_00031",
        "prompt": prompt,
        "ground_truth": {"target": ["8.16%"]},
        "meta": {
            **({"target_fact_id": "fact-2"} if verified else {}),
            "involved_records": [
                {"value_text": "-16.81%"},
                {"value_text": "8.16%"},
            ]
        },
    }


class TargetValidationTest(unittest.TestCase):
    def test_rejects_another_bank_claim_when_all_records_are_same_entity(self) -> None:
        record = sample_record(
            "哪家银行的指标恰好与另一家银行的数值相同？"
        )
        finding = find_target_alignment_issue(record)

        self.assertEqual(
            finding["reason"], "multi_entity_claim_unsupported_by_involved_records"
        )

    def test_rejects_equality_claim_when_values_differ(self) -> None:
        record = sample_record("该指标数值恰好与另一报告期相同？")
        finding = find_target_alignment_issue(record)

        self.assertEqual(
            finding["reason"], "equality_claim_conflicts_with_involved_values"
        )

    def test_keeps_negative_equality_claim_when_values_differ(self) -> None:
        finding = find_target_alignment_issue(
            sample_record("两期披露的该指标数值不相同，目标期间的数值是多少？")
        )

        self.assertIsNone(finding)

    def test_rejects_percentage_point_calculation_with_scalar_target(self) -> None:
        finding = find_target_alignment_issue(
            sample_record("2025年第一季度净利润同比增速较2024年第三季度下降了多少个百分点？")
        )

        self.assertIsNotNone(finding)
        self.assertEqual(finding["reason"], "derived_question_with_scalar_final_record_target")

    def test_keeps_direct_lookup_question(self) -> None:
        finding = find_target_alignment_issue(
            sample_record("某医疗器械企业在特定报告期的净利润同比增速具体是多少？")
        )

        self.assertIsNone(finding)

    def test_rejects_legacy_multihop_row_without_alignment_provenance(self) -> None:
        finding = find_target_alignment_issue(
            sample_record("某医疗器械企业在特定报告期的净利润同比增速具体是多少？", verified=False)
        )

        self.assertEqual(finding["reason"], "legacy_target_alignment_unverifiable")
