"""Tests for deterministic parts of the documented Agentic RL synthesis flow."""

from __future__ import annotations

import pytest

from fintrace.data.agentic_synthesis import (
    answer_targets,
    build_anchored_lookup_question,
    build_entity_groups,
    output_record,
    parse_candidate_choice,
    unique_verdict_is_yes,
)
from fintrace.data.schema import parse_financial_qa_sample


def record(fact_id: str, *, value_text: str = "15.3%") -> dict:
    return {
        "fact_id": fact_id,
        "fact": "测试公司2025-Q4营收同比增速为15.3%。",
        "entity": "测试公司",
        "metric": "营收同比增速",
        "date": "2025-Q4",
        "value_text": value_text,
        "value_number": 15.3,
        "unit": "%",
        "structured": True,
    }


def test_entity_grouping_excludes_records_without_ground_truth_value() -> None:
    invalid = record("bad", value_text="")
    groups = build_entity_groups([record("a"), invalid, record("b")])

    assert list(groups) == ["测试公司"]
    assert [item["fact_id"] for item in groups["测试公司"]] == ["a", "b"]


def test_choice_parser_only_accepts_one_based_bounded_indices() -> None:
    assert parse_candidate_choice("2", 3) == 1
    assert parse_candidate_choice(" 3。", 3) == 2
    assert parse_candidate_choice("候选2", 3) is None
    assert parse_candidate_choice("4", 3) is None


def test_unique_verdict_accepts_api_affirmative_variants_only() -> None:
    assert unique_verdict_is_yes("是")
    assert unique_verdict_is_yes("是，问题仍可唯一确定。")
    assert not unique_verdict_is_yes("否")
    assert not unique_verdict_is_yes("不是")


def test_output_is_grounded_in_final_record_and_has_audit_meta() -> None:
    root = record("root", value_text="10%")
    final = record("final")
    output = output_record(qid="ft_00001", prompt="模糊化问题", root=root, involved=[root, final], actions=["SELECT", "FUZZ"])

    assert output["source"] == "fintrace_kb"
    assert output["ground_truth"]["target"] == answer_targets(final)
    assert output["meta"]["hop_count"] == 2
    assert output["meta"]["target_fact_id"] == "final"
    assert output["meta"]["involved_records"][1]["fact_id"] == "final"
    assert parse_financial_qa_sample(output).qid == "ft_00001"


def test_anchored_template_hides_entity_and_never_invents_a_relation() -> None:
    root = record("root", value_text="10%")
    root["date"] = "2021-Q1"
    target = record("target", value_text="15%")
    target["date"] = "2022-Q1"

    question = build_anchored_lookup_question(root, target)

    assert "测试公司" not in question
    assert "2021-Q1" in question
    assert "10%" in question
    assert "2022-Q1" in question
    assert "15%" not in question


def test_anchored_template_rejects_target_answer_leak() -> None:
    root = record("root", value_text="10%")
    target = record("target", value_text="10%")
    target["date"] = "2024-Q1"

    with pytest.raises(ValueError, match="leak"):
        build_anchored_lookup_question(root, target)
