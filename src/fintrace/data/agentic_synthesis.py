"""Pure helpers for the documented FinTrace RL dataset construction workflow."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable


def eligible_record(record: dict[str, Any]) -> bool:
    """Only structured, time-bound numeric facts may enter an RL reasoning chain."""
    value_number = record.get("value_number")
    return bool(
        record.get("fact_id")
        and record.get("entity")
        and record.get("metric")
        and record.get("value_text")
        and record.get("structured") is True
        and str(record.get("date", "")).strip()
        and isinstance(value_number, (int, float))
        and math.isfinite(value_number)
    )


def build_entity_groups(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group eligible records so root sampling is balanced by entity, not row count."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if eligible_record(record):
            groups[str(record["entity"])].append(record)
    return dict(groups)


def parse_candidate_choice(response: str, candidate_count: int) -> int | None:
    """Parse the required one-based candidate index and reject free-form output."""
    match = re.fullmatch(r"\s*(\d+)\s*[。.]?\s*", response)
    if not match:
        return None
    index = int(match.group(1)) - 1
    return index if 0 <= index < candidate_count else None


def unique_verdict_is_yes(response: str) -> bool:
    """Accept the API's common affirmative variants while rejecting negative answers."""
    compact = re.sub(r"\s+", "", response).strip("\"'“”")
    return compact.startswith("是")


def answer_targets(record: dict[str, Any]) -> list[str]:
    """Anchor answers in the final record; an LLM never invents these values."""
    value_text = str(record["value_text"]).strip()
    targets = [value_text]
    value_number = record.get("value_number")
    unit = str(record.get("unit", "")).strip()
    if isinstance(value_number, (int, float)) and math.isfinite(value_number):
        numeric = f"{float(value_number):.6f}".rstrip("0").rstrip(".")
        normalized = f"{numeric}{unit}" if unit else numeric
        if normalized not in targets:
            targets.append(normalized)
    return targets


def render_candidates(candidates: list[dict[str, Any]]) -> str:
    """Give the remote LLM only numbered, provenance-preserving choices."""
    return "\n".join(
        f"{index}. [{record['fact_id']}] {record['fact']}"
        for index, record in enumerate(candidates, start=1)
    )


def output_record(
    *,
    qid: str,
    prompt: str,
    root: dict[str, Any],
    involved: list[dict[str, Any]],
    actions: list[str],
) -> dict[str, Any]:
    """Build the documented public training schema and retain full audit metadata."""
    final_record = involved[-1]
    return {
        "prompt": prompt.strip(),
        "ground_truth": {"target": answer_targets(final_record), "valid_inst": True},
        "source": "fintrace_kb",
        "qid": qid,
        "meta": {
            "root_entity": root["entity"],
            "hop_count": len(involved),
            "actions": actions,
            "involved_records": [
                {
                    "fact_id": record["fact_id"],
                    "entity": record["entity"],
                    "metric": record["metric"],
                    "date": record.get("date", ""),
                    "value_text": record["value_text"],
                }
                for record in involved
            ],
        },
    }
