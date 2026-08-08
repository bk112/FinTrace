"""Tests for assistant/environment token ownership masks."""

from __future__ import annotations

import unittest

from fintrace.rollout import TraceSegment, tokenize_trace_with_env_mask


class CharacterOffsetTokenizer:
    """A deterministic tokenizer where each character is one token."""

    def __call__(self, text: str, **kwargs):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class LossMaskingTest(unittest.TestCase):
    def test_environment_tokens_are_masked_from_loss(self) -> None:
        trace = tokenize_trace_with_env_mask(
            CharacterOffsetTokenizer(),
            [
                TraceSegment(owner="assistant", text="AB"),
                TraceSegment(owner="environment", text="XYZ"),
                TraceSegment(owner="assistant", text="C"),
            ],
        )

        self.assertEqual(trace.text, "ABXYZC")
        self.assertEqual(trace.env_mask, [1, 1, 0, 0, 0, 1])
