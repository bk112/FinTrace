#!/usr/bin/env python3
"""Audit whether every synthesized RL question is grounded in its source records."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from construct_rl_dataset import (
    PROJECT_ROOT,
    DeepSeekFlashClient,
    RELATION_GROUNDING_PROMPT,
    RemoteLLMError,
    _status,
)
from fintrace.data.target_validation import find_target_alignment_issue
from fintrace.data.agentic_synthesis import unique_verdict_is_yes


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_facts(record: dict[str, Any]) -> tuple[str, str]:
    """Render auditable metadata into the same fact-chain shape used by synthesis."""
    meta = record.get("meta", {})
    involved = meta.get("involved_records", []) if isinstance(meta, dict) else []
    if not isinstance(involved, list) or not involved:
        raise ValueError("record has no involved_records")
    facts: list[str] = []
    final_fact = involved[-1]
    for item in involved:
        if not isinstance(item, dict):
            raise ValueError("involved_records contains a non-object value")
        facts.append(
            "- [{fact_id}] {entity}{date}的{metric}为{value_text}".format(
                fact_id=item.get("fact_id", ""),
                entity=item.get("entity", ""),
                date=item.get("date", ""),
                metric=item.get("metric", ""),
                value_text=item.get("value_text", ""),
            )
        )
    return "\n".join(facts), facts[-1]


def audit_relation(record: dict[str, Any], client: DeepSeekFlashClient) -> dict[str, Any] | None:
    """Return a finding for any unsupported relation; API failures fail closed."""
    local_issue = find_target_alignment_issue(record)
    if local_issue is not None:
        return local_issue
    try:
        facts, final_fact = render_facts(record)
        verdict = client.complete(
            RELATION_GROUNDING_PROMPT.format(
                target_fact=final_fact,
                facts=facts,
                question=str(record.get("prompt", "")),
            )
        )
    except (RemoteLLMError, ValueError) as error:
        return {
            "qid": record.get("qid"),
            "reason": "relation_grounding_unverified",
            "detail": str(error),
            "prompt": record.get("prompt"),
        }
    if unique_verdict_is_yes(verdict):
        return None
    return {
        "qid": record.get("qid"),
        "reason": "relation_grounding_failed",
        "verdict": verdict[:240],
        "prompt": record.get("prompt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--filtered-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"input file does not exist: {args.input}", file=sys.stderr)
        return 2
    for path in (args.report, args.filtered_output):
        if path.exists():
            print(f"refusing to overwrite existing output: {path}", file=sys.stderr)
            return 2

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("CODEBUDDY_API_KEY", "").strip()
    if not api_key:
        print("CODEBUDDY_API_KEY is missing. Fill it in .env before running this script.", file=sys.stderr)
        return 2

    rows = read_jsonl(args.input)
    client = DeepSeekFlashClient(api_key, args.timeout_seconds)
    findings: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    _status(f"开始关系事实复审: {len(rows)} 条；API 单并发、失败闭合")
    for row in tqdm(rows, desc="关系复审", unit="题", file=sys.stdout, dynamic_ncols=True):
        finding = audit_relation(row, client)
        if finding is None:
            kept.append(row)
        else:
            findings.append(finding)

    write_jsonl_new(args.report, findings)
    if not kept:
        print("no relation-grounded records remain; filtered output was not created", file=sys.stderr)
        return 1
    write_jsonl_new(args.filtered_output, kept)
    print(
        json.dumps(
            {
                "input_records": len(rows),
                "rejected_records": len(findings),
                "kept_records": len(kept),
                "report": str(args.report),
                "filtered_output": str(args.filtered_output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
