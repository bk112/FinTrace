"""Tests for multi-target reward selection and vLLM logprob extraction."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from fintrace.rewards import SearchStep, Trajectory, compute_total_reward
from fintrace.training.react_grpo_rollout import extract_sampled_logprobs
from fintrace.training.reward_adapter import financial_trajectory_reward


class TrainingRewardTest(unittest.TestCase):
    def test_accepts_any_ground_truth_variant(self) -> None:
        trajectory = Trajectory(
            raw_text="<think>done</think><answer>新能源</answer>",
            ground_truth="unused",
            final_answer="新能源",
        )

        reward = financial_trajectory_reward(
            trajectories=[trajectory], targets=[("New energy", "新能源")]
        )

        self.assertGreater(reward[0], 0.5)

    def test_extracts_sampled_token_logprobs(self) -> None:
        values = extract_sampled_logprobs(
            [101, 102],
            [{101: SimpleNamespace(logprob=-0.1)}, {102: SimpleNamespace(logprob=-0.2)}],
        )

        self.assertEqual(values, [-0.1, -0.2])

    def test_rejects_missing_sampled_logprob(self) -> None:
        with self.assertRaises(ValueError):
            extract_sampled_logprobs([101], [{102: SimpleNamespace(logprob=-0.1)}])

    def test_structured_retrieval_record_contributes_retrieval_reward(self) -> None:
        trajectory = Trajectory(
            raw_text="<think>done</think><search>report</search><answer>15.3%</answer>",
            ground_truth="15.3%",
            final_answer="15.3%",
            search_steps=[
                SearchStep(
                    query="report",
                    retrieved_text="",
                    retrieved_records=[{"value_text": "15.3%", "fact": "营收同比增长15.3%"}],
                )
            ],
        )

        self.assertEqual(compute_total_reward(trajectory).retrieval_correctness, 0.1)

    def test_structured_numeric_match_requires_compatible_percentage_unit(self) -> None:
        trajectory = Trajectory(
            raw_text="<think>done</think><search>report</search><answer>15.3%</answer>",
            ground_truth="15.3%",
            final_answer="15.3%",
            search_steps=[
                SearchStep(
                    query="report",
                    retrieved_text="",
                    retrieved_records=[{"value_text": "15.3亿元", "fact": "营收为15.3亿元", "unit": "亿元"}],
                )
            ],
        )

        self.assertEqual(compute_total_reward(trajectory).retrieval_correctness, 0.0)
