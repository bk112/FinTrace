#!/usr/bin/env python3
"""Validate a financial multi-hop JSONL dataset before any rollout or training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fintrace.data.jsonl import JsonlDatasetError, iter_financial_qa_samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="UTF-8 JSONL dataset to validate")
    args = parser.parse_args()

    if not args.dataset.is_file():
        print(f"Dataset does not exist: {args.dataset}", file=sys.stderr)
        return 2

    try:
        samples = list(iter_financial_qa_samples(args.dataset))
    except JsonlDatasetError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    if not samples:
        print("Validation failed: dataset has no records", file=sys.stderr)
        return 1

    duplicate_count = len(samples) - len({sample.qid for sample in samples})
    if duplicate_count:
        print(f"Validation failed: {duplicate_count} duplicate qid value(s)", file=sys.stderr)
        return 1

    answerable_count = sum(sample.valid_inst for sample in samples)
    print(
        f"Validated {len(samples)} records; answerable={answerable_count}; "
        f"unanswerable={len(samples) - answerable_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
