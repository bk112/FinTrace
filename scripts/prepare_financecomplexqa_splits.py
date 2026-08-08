#!/usr/bin/env python3
"""Split isolated FinanceComplexQA labels into training-data contract files.

The input contains supervision only and is intentionally outside the KB build
chain. Splitting by qid keeps the bootstrap train/validation/test files
deterministic without reintroducing gold answers into retrieval evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = PROJECT_ROOT / "data" / "interim" / "financecomplexqa_gold_labels.jsonl"
OUTPUTS = {
    "train": PROJECT_ROOT / "data" / "processed" / "train.jsonl",
    "validation": PROJECT_ROOT / "data" / "processed" / "validation.jsonl",
    "test": PROJECT_ROOT / "data" / "evaluation" / "financial_multihop_test.jsonl",
}


def split_for_qid(qid: str) -> str:
    """Assign a stable split from qid; do not depend on input file order."""
    bucket = int(hashlib.sha256(f"fintrace-fcqa-v1:{qid}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def main() -> None:
    if not LABELS_PATH.exists():
        raise SystemExit(f"标签文件不存在: {LABELS_PATH}; 请先运行 build_kb_stage3_local.py")

    grouped = {name: [] for name in OUTPUTS}
    seen_qids: set[str] = set()
    labels_by_prompt: dict[str, dict] = {}
    conflicting_prompts: set[str] = set()
    duplicate_prompts = 0
    with LABELS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            qid = record["qid"]
            if qid in seen_qids:
                raise ValueError(f"标签文件存在重复 qid: {qid}")
            seen_qids.add(qid)

            prompt = record["prompt"]
            existing = labels_by_prompt.get(prompt)
            if existing is None:
                labels_by_prompt[prompt] = record
            elif existing["ground_truth"]["target"] == record["ground_truth"]["target"]:
                duplicate_prompts += 1
            else:
                # rollout 当前按 prompt 保存 metadata，冲突答案不能任选一条继续训练。
                conflicting_prompts.add(prompt)

    for prompt, record in labels_by_prompt.items():
        if prompt not in conflicting_prompts:
            grouped[split_for_qid(record["qid"])].append(record)

    for name, path in OUTPUTS.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for record in grouped[name]:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{name}: {len(grouped[name])} 条 -> {path}")
    print(f"合并同答案重复 prompt: {duplicate_prompts} 条; 剔除冲突 prompt: {len(conflicting_prompts)} 条")


if __name__ == "__main__":
    main()
