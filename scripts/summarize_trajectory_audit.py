#!/usr/bin/env python3
"""Summarize GRPO ReAct trajectory audits into rollout and group health metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read audit rows and fail early on malformed records."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}") from error
            if not isinstance(row, dict) or not isinstance(row.get("reward"), dict):
                raise ValueError(f"audit line {line_number} lacks a reward object")
            rows.append(row)
    if not rows:
        raise ValueError("audit file contains no records")
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report whether trajectories finish, use search, and retain GRPO reward variance."""
    rewards = [float(row["reward"]["total"]) for row in rows]
    answer_rows = [row for row in rows if row.get("final_answer")]
    searched_and_answered = [row for row in answer_rows if row.get("searches")]
    terminated_zero = [row for row in rows if row["reward"].get("terminated_with_zero")]
    # 同一 prompt 的 num_generations 条轨迹会连续写入；按连续块分组，避免训练
    # 多次抽到同一 qid 时被错误合并为一个 GRPO 组。
    reward_groups: list[list[float]] = []
    current_qid: str | None = None
    current_rewards: list[float] = []
    for row, reward in zip(rows, rewards, strict=True):
        qid = str(row.get("qid"))
        if current_qid is not None and qid != current_qid:
            reward_groups.append(current_rewards)
            current_rewards = []
        current_qid = qid
        current_rewards.append(reward)
    if current_rewards:
        reward_groups.append(current_rewards)
    group_stds = [population_std(group_rewards) for group_rewards in reward_groups]
    return {
        "trajectories": len(rows),
        "prompt_groups": len(reward_groups),
        "answer_rate": len(answer_rows) / len(rows),
        "search_to_answer_rate": len(searched_and_answered) / len(rows),
        "search_rate": sum(1 for row in rows if row.get("searches")) / len(rows),
        "repeated_query_rate": sum(1 for row in rows if row.get("is_repeated_query")) / len(rows),
        "terminated_zero_rate": len(terminated_zero) / len(rows),
        "reward_mean": mean(rewards),
        "reward_std": population_std(rewards),
        "zero_reward_std_group_rate": sum(std == 0.0 for std in group_stds) / len(group_stds),
        "avg_rounds": mean(float(row.get("num_rounds", 0)) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="trajectory audit JSONL")
    parser.add_argument("--output", type=Path, help="optional JSON summary output")
    args = parser.parse_args()

    summary = summarize_rows(read_jsonl(args.input))
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
