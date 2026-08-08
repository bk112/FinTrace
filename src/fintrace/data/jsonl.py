"""JSONL loading utilities with record-level validation errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .schema import DatasetValidationError, FinancialQASample, parse_financial_qa_sample


class JsonlDatasetError(DatasetValidationError):
    """Adds a file and line location to a schema-validation failure."""


def iter_financial_qa_samples(path: Path) -> Iterator[FinancialQASample]:
    """Yield validated samples from a UTF-8 JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlDatasetError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise JsonlDatasetError(f"{path}:{line_number}: record must be a JSON object")
            try:
                yield parse_financial_qa_sample(record)
            except DatasetValidationError as exc:
                raise JsonlDatasetError(f"{path}:{line_number}: {exc}") from exc
