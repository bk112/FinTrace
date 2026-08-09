"""Conservative consistency checks for record-anchored RL synthesis targets."""

from __future__ import annotations

import re
from typing import Any


DERIVED_OPERATOR_PATTERNS = (
    r"多少个?百分点",
    r"相差多少",
    r"两者.*(?:差值|之差)",
    r"(?:高出|低了|增加了|减少了|提升了|下降了|降低了)多少",
    r"(?:增长|提升|下降|变化).*(?:幅度|倍数)",
    r"变动幅度",
    r"累计变化",
    r"多少倍",
    r"(?:比例|占比).*(?:多少|约为)",
)

MULTI_ENTITY_CLAIM_PATTERNS = (
    r"另一(?:家)?(?:公司|银行|企业|券商|机构)",
    r"其他(?:公司|银行|企业|券商|机构)",
)

EQUALITY_CLAIM_PATTERNS = (
    r"(?:数值|指标|增速|比例).*?(?<!不)(?:相同|一致|相等)",
    r"恰好与.*?(?<!不)(?:相同|一致|相等)",
    r"(?<!不)(?:相同|一致|相等).*(?:数值|指标|增速|比例)",
)


def find_target_alignment_issue(record: dict[str, Any]) -> dict[str, Any] | None:
    """Flag calculation questions whose target is only the final raw record value.

    The synthesis workflow deliberately anchors targets to the final record. Such a
    target cannot answer a question that asks for a difference, ratio, or percentage
    change across multiple records. Legacy multi-hop rows without target-alignment
    provenance are also rejected because their question intent cannot be verified.
    """
    prompt = str(record.get("prompt", ""))
    meta = record.get("meta", {})
    if not isinstance(meta, dict):
        return {"qid": record.get("qid"), "reason": "missing_target_alignment_metadata"}
    involved = meta.get("involved_records", [])
    if not isinstance(involved, list) or len(involved) < 2:
        return None

    values = [str(item.get("value_text", "")) for item in involved if isinstance(item, dict)]
    entities = {
        str(item.get("entity", "")).strip()
        for item in involved
        if isinstance(item, dict) and str(item.get("entity", "")).strip()
    }
    target = record.get("ground_truth", {}).get("target", [])
    if not meta.get("target_fact_id"):
        return {
            "qid": record.get("qid"),
            "reason": "legacy_target_alignment_unverifiable",
            "prompt": prompt,
            "targets": target if isinstance(target, list) else [target],
            "involved_values": values,
        }

    matched_entity_patterns = [
        pattern for pattern in MULTI_ENTITY_CLAIM_PATTERNS if re.search(pattern, prompt)
    ]
    if matched_entity_patterns and len(entities) < 2:
        return {
            "qid": record.get("qid"),
            "reason": "multi_entity_claim_unsupported_by_involved_records",
            "matched_patterns": matched_entity_patterns,
            "prompt": prompt,
            "involved_entities": sorted(entities),
        }

    matched_equality_patterns = [
        pattern for pattern in EQUALITY_CLAIM_PATTERNS if re.search(pattern, prompt)
    ]
    if matched_equality_patterns and len(set(values)) > 1:
        return {
            "qid": record.get("qid"),
            "reason": "equality_claim_conflicts_with_involved_values",
            "matched_patterns": matched_equality_patterns,
            "prompt": prompt,
            "involved_values": values,
        }

    matched_patterns = [pattern for pattern in DERIVED_OPERATOR_PATTERNS if re.search(pattern, prompt)]
    if not matched_patterns:
        return None

    return {
        "qid": record.get("qid"),
        "reason": "derived_question_with_scalar_final_record_target",
        "matched_patterns": matched_patterns,
        "prompt": prompt,
        "targets": target if isinstance(target, list) else [target],
        "involved_values": values,
    }
