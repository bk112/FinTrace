"""Bridge the supplied single-target reward module to multi-target QA data."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from fintrace.rewards import Trajectory, compute_total_reward


def financial_trajectory_reward(
    *,
    trajectories: Sequence[Trajectory],
    targets: Sequence[Sequence[str]],
    **_: Any,
) -> list[float]:
    """Score every acceptable target and retain the best valid formulation."""
    if len(trajectories) != len(targets):
        raise ValueError("trajectories and targets must have equal lengths")

    rewards: list[float] = []
    for trajectory, accepted_targets in zip(trajectories, targets, strict=True):
        if not accepted_targets:
            raise ValueError("every trajectory needs at least one accepted target")
        # 数据集允许同一事实有多个规范表述；不能只拿第一个表述误伤正确回答。
        rewards.append(
            max(
                compute_total_reward(replace(trajectory, ground_truth=target)).total
                for target in accepted_targets
            )
        )
    return rewards
