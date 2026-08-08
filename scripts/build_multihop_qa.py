#!/usr/bin/env python3
"""Build a verified temporal-comparison QA pilot from the canonical KB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fintrace.data import parse_financial_qa_sample
from fintrace.data.synthesis import SynthesisConfig, build_temporal_comparison_examples, split_pilot_examples

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORDS = PROJECT_ROOT / "data" / "kb" / "records.jsonl"
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "interim" / "multihop_qa_candidates.jsonl"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_VAL = PROJECT_ROOT / "data" / "processed" / "validation.jsonl"
DEFAULT_TEST = PROJECT_ROOT / "data" / "evaluation" / "financial_multihop_test.jsonl"


def _load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--total-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-retrieval", action="store_true", help="require both source facts to be in Top-k")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    records = _load_records(args.records)
    candidates = build_temporal_comparison_examples(records)
    selected = split_pilot_examples(candidates, SynthesisConfig(args.seed, args.total_examples))

    verification = {"checked": 0, "passed": 0}
    if args.verify_retrieval:
        # 只在显式验收时加载 embedding 模型，避免普通构造阶段被模型启动时间拖慢。
        from fintrace.kb import search_records

        for examples in selected.values():
            for example in examples:
                source = example["source"]
                expected_ids = source["required_fact_ids"]
                hit_ids = [
                    {record["fact_id"] for record in search_records(query, args.top_k)}
                    for query in source["retrieval_queries"]
                ]
                verification["checked"] += 1
                if all(fact_id in ids for fact_id, ids in zip(expected_ids, hit_ids, strict=True)):
                    verification["passed"] += 1
                    continue
                raise RuntimeError(
                    f"检索验收失败: {example['qid']} expected={expected_ids}, queries={source['retrieval_queries']}"
                )

    # 在写盘前复用训练 loader 的 schema，防止生成脚本与训练入口各自解释格式。
    for examples in selected.values():
        for example in examples:
            parse_financial_qa_sample(example)

    _write_jsonl(DEFAULT_CANDIDATES, candidates)
    _write_jsonl(DEFAULT_TRAIN, selected["train"])
    _write_jsonl(DEFAULT_VAL, selected["val"])
    _write_jsonl(DEFAULT_TEST, selected["test"])
    print(
        json.dumps(
            {
                "candidates": len(candidates),
                "train": len(selected["train"]),
                "validation": len(selected["val"]),
                "test": len(selected["test"]),
                "candidate_path": str(DEFAULT_CANDIDATES),
                "retrieval_verification": verification,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
