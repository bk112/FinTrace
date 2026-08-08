"""Validated on-disk schema for financial multi-hop QA records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class DatasetValidationError(ValueError):
    """Raised when a record cannot safely enter a training or evaluation set."""


@dataclass(frozen=True)
class FinancialQASample:
    """One question with one or more acceptable, verifiable answers."""

    qid: str
    prompt: str
    targets: tuple[str, ...]
    source: Any
    valid_inst: bool = True

    @property
    def primary_target(self) -> str:
        """Compatibility target for the supplied single-target reward module."""
        return self.targets[0]


def parse_financial_qa_sample(record: Mapping[str, Any]) -> FinancialQASample:
    """Validate the documented ``prompt, ground_truth.target, source, qid`` contract."""
    required_strings = ("qid", "prompt")
    for key in required_strings:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DatasetValidationError(f"{key} must be a non-empty string")

    if "source" not in record:
        raise DatasetValidationError("source is required for provenance")

    ground_truth = record.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise DatasetValidationError("ground_truth must be an object")

    targets = ground_truth.get("target")
    if not isinstance(targets, list) or not targets:
        raise DatasetValidationError("ground_truth.target must be a non-empty list")
    if any(not isinstance(target, str) or not target.strip() for target in targets):
        raise DatasetValidationError("every ground_truth.target item must be a non-empty string")

    valid_inst = ground_truth.get("valid_inst", True)
    if not isinstance(valid_inst, bool):
        raise DatasetValidationError("ground_truth.valid_inst must be a boolean when provided")

    return FinancialQASample(
        qid=record["qid"].strip(),
        prompt=record["prompt"].strip(),
        targets=tuple(target.strip() for target in targets),
        source=record["source"],
        valid_inst=valid_inst,
    )
