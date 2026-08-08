"""Tests for vLLM adapter behavior without loading model weights."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from fintrace.rollout import VllmGenerationEngine


class RecordingLlm:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], object, bool]] = []

    def generate(self, prompts, sampling_params, *, use_tqdm):
        self.calls.append((prompts, sampling_params, use_tqdm))
        return [SimpleNamespace(outputs=[SimpleNamespace(text="<answer>answer</answer>")])]


class VllmGenerationEngineTest(unittest.TestCase):
    def test_preserves_stop_tags_for_react_parser(self) -> None:
        llm = RecordingLlm()
        engine = VllmGenerationEngine(
            "unused-model-path",
            max_tokens=128,
            llm=llm,
            sampling_params_factory=lambda **kwargs: kwargs,
        )

        output = engine.generate("prompt", ["</search>", "</answer>"])

        self.assertEqual(output, "<answer>answer</answer>")
        prompts, sampling_params, use_tqdm = llm.calls[0]
        self.assertEqual(prompts, ["prompt"])
        self.assertFalse(use_tqdm)
        self.assertTrue(sampling_params["include_stop_str_in_output"])
        self.assertEqual(sampling_params["stop"], ["</search>", "</answer>"])
