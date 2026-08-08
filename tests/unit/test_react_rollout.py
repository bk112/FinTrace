"""Tests for multi-turn rollout control flow independent of vLLM."""

from __future__ import annotations

import unittest
from collections.abc import Sequence

from fintrace.rollout import ReActRollout, RolloutTermination
from fintrace.tools import RetrievalResult


class FakeEngine:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = iter(outputs)
        self.stop_sequences: list[tuple[str, ...]] = []

    def generate(self, prompt: str, stop_sequences: Sequence[str]) -> str:
        self.stop_sequences.append(tuple(stop_sequences))
        return next(self._outputs)


class FakeRetrievalClient:
    def __init__(self, results: dict[str, str]) -> None:
        self._results = results

    def search(self, query: str) -> RetrievalResult:
        return RetrievalResult(query=query, text=self._results[query])


class ReActRolloutTest(unittest.TestCase):
    def test_two_searches_then_answer_preserves_environment_segments(self) -> None:
        rollout = ReActRollout(
            FakeEngine(
                [
                    "<think>Find the report</think><search>company annual report</search>",
                    "<think>Find the ranking</think><search>industry ranking</search>",
                    "<think>Combine the facts</think><answer>New energy ranked first</answer>",
                ]
            ),
            FakeRetrievalClient(
                {
                    "company annual report": "RETRIEVED_REPORT_FACT",
                    "industry ranking": "RETRIEVED_RANKING_FACT",
                }
            ),
        )

        result = rollout.run("Question", "New energy ranked first")

        self.assertTrue(result.completed)
        self.assertEqual(result.termination, RolloutTermination.ANSWER)
        self.assertEqual(result.trajectory.num_rounds, 3)
        self.assertEqual(len(result.trajectory.search_steps), 2)
        self.assertEqual(
            [segment.owner for segment in result.segments],
            ["assistant", "environment", "assistant", "environment", "assistant"],
        )
        self.assertNotIn("RETRIEVED_REPORT_FACT", result.trajectory.raw_text)

    def test_repeated_query_stops_before_a_second_tool_request(self) -> None:
        rollout = ReActRollout(
            FakeEngine(
                [
                    "<think>Search</think><search>annual report</search>",
                    "<think>Search again</think><search>ANNUAL REPORT</search>",
                ]
            ),
            FakeRetrievalClient({"annual report": "result"}),
        )

        result = rollout.run("Question", "Answer")

        self.assertEqual(result.termination, RolloutTermination.REPEATED_QUERY)
        self.assertTrue(result.trajectory.is_repeated_query)
        self.assertEqual(len(result.trajectory.search_steps), 1)

    def test_missing_closed_action_is_malformed(self) -> None:
        rollout = ReActRollout(
            FakeEngine(["<think>Search</think><search>annual report"]),
            FakeRetrievalClient({}),
        )

        result = rollout.run("Question", "Answer")

        self.assertEqual(result.termination, RolloutTermination.MALFORMED_ACTION)
