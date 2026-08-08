"""Deterministic, auditable construction of financial multi-hop QA examples.

The first-pass generator intentionally uses executable templates instead of an
LLM judge: two source facts are selected from the KB and the target is computed
in code. This gives GRPO a trustworthy pilot set before introducing LLM-based
question paraphrasing or difficulty filtering.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SynthesisConfig:
    """Controls a reproducible pilot split without changing source facts."""

    seed: int = 42
    total_examples: int = 100


def _format_number(value: float) -> str:
    """Render a stable decimal answer and avoid binary floating-point noise."""
    rendered = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered


def _period_sort_key(period: str) -> tuple[int, int, str]:
    """Sort common ``YYYY-QN`` labels before falling back to lexical order."""
    try:
        year_text, quarter_text = period.split("-Q", maxsplit=1)
        return int(year_text), int(quarter_text), period
    except (ValueError, TypeError):
        return 0, 0, period


def _answer_targets(delta: float, unit: str) -> list[str]:
    """Return answer variants accepted by the text-and-numeric reward."""
    direction = "增加" if delta > 0 else "减少"
    magnitude = _format_number(abs(delta))
    if unit == "%":
        return [f"{direction}{magnitude}个百分点", f"{direction}{magnitude}%"]
    return [f"{direction}{magnitude}{unit}"]


def _build_example(older: dict[str, Any], newer: dict[str, Any]) -> dict[str, Any]:
    """Create one two-fact temporal comparison item with programmatic targets."""
    older_value = float(older["value_number"])
    newer_value = float(newer["value_number"])
    delta = newer_value - older_value
    unit = str(newer["unit"])
    entity = str(newer["entity"])
    metric = str(newer["metric"])
    older_period = str(older["date"])
    newer_period = str(newer["date"])
    targets = _answer_targets(delta, unit)
    fact_ids = [str(older["fact_id"]), str(newer["fact_id"])]
    retrieval_queries = [f"{entity}{older_period}{metric}", f"{entity}{newer_period}{metric}"]
    qid_seed = f"temporal-difference|{'|'.join(fact_ids)}"

    prompt = (
        f"比较{entity}{older_period}和{newer_period}的{metric}。"
        f"请先检索两个时期的数值，再计算从{older_period}到{newer_period}的变化量。"
        "最终只回答增加或减少了多少，并保留单位。"
    )
    return {
        "qid": f"kbcmp_{hashlib.sha256(qid_seed.encode()).hexdigest()[:16]}",
        "prompt": prompt,
        "ground_truth": {"target": targets, "valid_inst": True},
        "source": {
            "generator": "temporal_comparison_v1",
            "operation": "newer_value - older_value",
            "required_fact_ids": fact_ids,
            "entity": entity,
            "metric": metric,
            "unit": unit,
            "periods": [older_period, newer_period],
            "values": [older_value, newer_value],
            "retrieval_queries": retrieval_queries,
        },
        "kb_segment": str(newer.get("segment", "train")),
    }


def build_temporal_comparison_examples(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build candidates from pairs of unambiguous numeric facts.

    A relation is eligible only when both facts have the same entity, metric,
    unit and KB segment. Conflicting values for the same period are excluded,
    rather than selecting an arbitrary report revision.
    """
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value_type = record.get("value_type")
        unit = str(record.get("unit", ""))
        if value_type not in {"amount", "percentage"}:
            continue
        if value_type == "percentage" and unit != "%":
            continue
        if value_type == "amount" and unit not in {"万", "亿", "万元", "亿元", "万亿元"}:
            continue
        value = record.get("value_number")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        if not record.get("date"):
            continue
        key = (
            str(record.get("entity", "")),
            str(record.get("metric", "")),
            unit,
            str(record.get("segment", "")),
        )
        if all(key):
            groups[key].append(record)

    examples: list[dict[str, Any]] = []
    for group_records in groups.values():
        by_period: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in group_records:
            by_period[str(record["date"])].append(record)

        # 同一时期若存在不同值，不能构成唯一、可验证的标准答案。
        canonical: list[dict[str, Any]] = []
        for period, period_records in by_period.items():
            values = {float(record["value_number"]) for record in period_records}
            if len(values) == 1:
                canonical.append(min(period_records, key=lambda record: str(record["fact_id"])))
        canonical.sort(key=lambda record: _period_sort_key(str(record["date"])))

        for older, newer in zip(canonical, canonical[1:], strict=False):
            if float(older["value_number"]) != float(newer["value_number"]):
                examples.append(_build_example(older, newer))
    return examples


def split_pilot_examples(
    examples: Iterable[dict[str, Any]], config: SynthesisConfig
) -> dict[str, list[dict[str, Any]]]:
    """Sample a stable 80/10/10 pilot while preserving KB relation boundaries."""
    if config.total_examples < 3:
        raise ValueError("total_examples must be at least 3")
    quotas = {
        "train": config.total_examples * 8 // 10,
        "val": config.total_examples // 10,
        "test": config.total_examples - (config.total_examples * 8 // 10) - (config.total_examples // 10),
    }
    candidates: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for example in examples:
        segment = example["kb_segment"]
        if segment in candidates:
            candidates[segment].append(example)

    rng = random.Random(config.seed)
    selected: dict[str, list[dict[str, Any]]] = {}
    for segment, quota in quotas.items():
        pool = candidates[segment]
        if len(pool) < quota:
            raise ValueError(f"{segment} candidates insufficient: need {quota}, got {len(pool)}")
        selected[segment] = rng.sample(pool, quota)
    return selected
