"""Unit tests for executable multi-hop QA construction."""

from __future__ import annotations

from fintrace.data.synthesis import SynthesisConfig, build_temporal_comparison_examples, split_pilot_examples


def fact(fact_id: str, period: str, value: float, *, segment: str = "train") -> dict:
    return {
        "fact_id": fact_id,
        "entity": "测试公司",
        "metric": "营收增速",
        "date": period,
        "unit": "%",
        "value_type": "percentage",
        "value_number": value,
        "segment": segment,
    }


def test_builds_two_fact_percentage_comparison() -> None:
    examples = build_temporal_comparison_examples([fact("old", "2024-Q1", 10.0), fact("new", "2024-Q2", 12.5)])

    assert len(examples) == 1
    example = examples[0]
    assert example["source"]["required_fact_ids"] == ["old", "new"]
    assert example["source"]["retrieval_queries"] == ["测试公司2024-Q1营收增速", "测试公司2024-Q2营收增速"]
    assert example["ground_truth"]["target"] == ["增加2.5个百分点", "增加2.5%"]
    assert "2024-Q1" in example["prompt"] and "2024-Q2" in example["prompt"]


def test_skips_conflicting_same_period_values() -> None:
    examples = build_temporal_comparison_examples(
        [fact("a", "2024-Q1", 10.0), fact("b", "2024-Q1", 11.0), fact("c", "2024-Q2", 12.0)]
    )

    assert examples == []


def test_split_respects_kb_segments_and_is_reproducible() -> None:
    examples = []
    for segment, offset in (("train", 0), ("val", 100), ("test", 200)):
        examples.extend(
            build_temporal_comparison_examples(
                [fact(f"{segment}-{i}", f"2024-Q{i + 1}", float(offset + i), segment=segment) for i in range(12)]
            )
        )

    config = SynthesisConfig(seed=7, total_examples=10)
    first = split_pilot_examples(examples, config)
    second = split_pilot_examples(examples, config)

    assert {name: len(items) for name, items in first.items()} == {"train": 8, "val": 1, "test": 1}
    assert first == second
    assert all(example["kb_segment"] == name for name, items in first.items() for example in items)
