"""Bridge the supplied single-target reward module to multi-target QA data."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from fintrace.rewards import RewardBreakdown, Trajectory, compute_total_reward
from fintrace.rewards.constants import MAX_ROUNDS, MAX_TOOL_CALLS_PER_ROUND


def best_reward_breakdown(
    trajectory: Trajectory,
    accepted_targets: Sequence[str],
    *,
    max_rounds: int = MAX_ROUNDS,
    max_tool_calls_per_round: int = MAX_TOOL_CALLS_PER_ROUND,
) -> tuple[str, RewardBreakdown]:
    """Return the best scoring accepted target and its complete reward breakdown."""
    if not accepted_targets:
        raise ValueError("every trajectory needs at least one accepted target")
    return max(
        (
            (
                target,
                compute_total_reward(
                    replace(trajectory, ground_truth=target),
                    max_rounds=max_rounds,
                    max_tool_calls_per_round=max_tool_calls_per_round,
                ),
            )
            for target in accepted_targets
        ),
        key=lambda candidate: candidate[1].total,
    )


# 分维度奖励名称；聚合指标按此顺序生成 mean 值。
_REWARD_COMPONENTS = (
    "answer_correctness",
    "answer_cem",
    "format_compliance",
    "search_incentive",
    "retrieval_correctness",
    "total",
)


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-completion audit rows into one wandb-friendly metrics dict."""
    count = len(rows)
    if count == 0:
        return {}
    metrics = {"trajectory/count": float(count)}
    for name in _REWARD_COMPONENTS:
        metrics[f"reward/{name}_mean"] = sum(row["reward"][name] for row in rows) / count
    metrics["trajectory/answer_rate"] = sum(1 for row in rows if row["final_answer"]) / count
    metrics["trajectory/search_rate"] = sum(1 for row in rows if row["searches"]) / count
    metrics["trajectory/repeated_query_rate"] = sum(
        1 for row in rows if row["is_repeated_query"]
    ) / count
    metrics["trajectory/avg_num_rounds"] = sum(row["num_rounds"] for row in rows) / count
    metrics["trajectory/terminated_zero_rate"] = sum(
        1 for row in rows if row["reward"]["terminated_with_zero"]
    ) / count
    return metrics


def financial_trajectory_reward(
    *,
    trajectories: Sequence[Trajectory],
    targets: Sequence[Sequence[str]],
    max_rounds: int = MAX_ROUNDS,
    max_tool_calls_per_round: int = MAX_TOOL_CALLS_PER_ROUND,
    **_: Any,
) -> list[float]:
    """Score every acceptable target and retain the best valid formulation."""
    if len(trajectories) != len(targets):
        raise ValueError("trajectories and targets must have equal lengths")

    rewards: list[float] = []
    for trajectory, accepted_targets in zip(trajectories, targets, strict=True):
        # 数据集允许同一事实有多个规范表述；不能只拿第一个表述误伤正确回答。
        _, breakdown = best_reward_breakdown(
            trajectory,
            accepted_targets,
            max_rounds=max_rounds,
            max_tool_calls_per_round=max_tool_calls_per_round,
        )
        rewards.append(breakdown.total)
    return rewards


class TrajectoryAuditReward:
    """Score trajectories; optionally append audit records to JSONL.

    ``output_path`` may be ``None`` to keep scoring and wandb aggregation without
    persisting per-completion records.
    """

    def __init__(
        self,
        output_path: Path | None = None,
        *,
        max_rounds: int = MAX_ROUNDS,
        max_tool_calls_per_round: int = MAX_TOOL_CALLS_PER_ROUND,
    ) -> None:
        self._output_path = output_path
        # 终止阈值取自训练时实际生效的 rollout 配置，避免与 constants.py 的默认值脱节。
        self._max_rounds = max_rounds
        self._max_tool_calls_per_round = max_tool_calls_per_round

    def __call__(
        self,
        *,
        trajectories: Sequence[Trajectory],
        targets: Sequence[Sequence[str]],
        prompts: Sequence[str] | None = None,
        qid: Sequence[str] | None = None,
        **_: Any,
    ) -> list[float]:
        if len(trajectories) != len(targets):
            raise ValueError("trajectories and targets must have equal lengths")

        rows: list[dict[str, Any]] = []
        rewards: list[float] = []
        for index, (trajectory, accepted_targets) in enumerate(zip(trajectories, targets, strict=True)):
            selected_target, breakdown = best_reward_breakdown(
                trajectory,
                accepted_targets,
                max_rounds=self._max_rounds,
                max_tool_calls_per_round=self._max_tool_calls_per_round,
            )
            rewards.append(breakdown.total)
            rows.append(
                {
                    "qid": qid[index] if qid is not None else None,
                    "prompt": prompts[index] if prompts is not None else None,
                    "accepted_targets": list(accepted_targets),
                    "selected_target": selected_target,
                    "assistant_trace": trajectory.raw_text,
                    "final_answer": trajectory.final_answer,
                    "num_rounds": trajectory.num_rounds,
                    "is_repeated_query": trajectory.is_repeated_query,
                    "searches": [
                        {
                            "query": step.query,
                            "retrieved_characters": len(step.retrieved_text),
                            "retrieved_preview": step.retrieved_text[:500],
                            "retrieved_record_count": len(step.retrieved_records),
                        }
                        for step in trajectory.search_steps
                    ],
                    "reward": asdict(breakdown),
                }
            )

        # 采用追加写入，确保多 step 训练保留全部采样；入口会拒绝误覆写旧审计文件。
        if self._output_path is not None:
            with self._output_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        # wandb 聚合指标；未安装或未初始化时静默跳过，不影响奖励计算。
        try:
            import wandb
        except ImportError:
            wandb = None  # type: ignore[assignment]
        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.log(_aggregate_metrics(rows))
        return rewards
