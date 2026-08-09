#!/usr/bin/env python3
"""Audit record-anchored RL targets and create a safe filtered training copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fintrace.data.target_validation import find_target_alignment_issue


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--filtered-output", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    findings = [finding for row in rows if (finding := find_target_alignment_issue(row)) is not None]
    invalid_qids = {str(finding["qid"]) for finding in findings}
    filtered_rows = [row for row in rows if str(row.get("qid")) not in invalid_qids]

    # 原始合成集保持不变；报告和训练副本必须使用从未存在过的新路径。
    write_jsonl_new(args.report, findings)
    if args.filtered_output is not None and filtered_rows:
        write_jsonl_new(args.filtered_output, filtered_rows)
    print(
        json.dumps(
            {
                "input_records": len(rows),
                "rejected_records": len(findings),
                "kept_records": len(filtered_rows),
                "report": str(args.report),
                "filtered_output": str(args.filtered_output) if filtered_rows and args.filtered_output else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
