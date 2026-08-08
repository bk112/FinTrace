"""Contract test for the custom rollout output consumed by TRL GRPO."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from fintrace.tools import RetrievalResult
from fintrace.training.react_grpo_rollout import PromptMetadata, VllmReActGRPORollout


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return f"user:{messages[0]['content']}\nassistant:"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) for char in text]}


class FakeLlm:
    def __init__(self, texts: list[str]) -> None:
        self._texts = iter(texts)

    def generate(self, prompts, sampling_params, *, use_tqdm):
        text = next(self._texts)
        token_ids = [1000 + index for index in range(len(text))]
        logprobs = [{token_id: SimpleNamespace(logprob=-0.1)} for token_id in token_ids]
        output = SimpleNamespace(text=text, token_ids=token_ids, logprobs=logprobs)
        return [SimpleNamespace(outputs=[output])]


class FakeRetriever:
    def search(self, query: str) -> RetrievalResult:
        return RetrievalResult(query=query, text="retrieved evidence")


class FakeTrainer:
    num_generations = 1


class GRPORolloutContractTest(unittest.TestCase):
    def test_returns_aligned_ids_logprobs_and_environment_mask(self) -> None:
        rollout = VllmReActGRPORollout(
            model_path="unused",
            tokenizer=CharacterTokenizer(),
            retrieval_client=FakeRetriever(),
            metadata_by_prompt={"question": PromptMetadata(("answer",), True)},
            llm=FakeLlm(
                [
                    "<think>search</think><search>annual report</search>",
                    "<think>answer</think><answer>answer</answer>",
                ]
            ),
            sampling_params_factory=lambda **kwargs: kwargs,
        )

        output = rollout(["question"], FakeTrainer())

        completion_ids = output["completion_ids"][0]
        self.assertEqual(len(completion_ids), len(output["logprobs"][0]))
        self.assertEqual(len(completion_ids), len(output["env_mask"][0]))
        self.assertIn(0, output["env_mask"][0])
        self.assertTrue(any(value == 0.0 for value in output["logprobs"][0]))
        self.assertEqual(output["trajectories"][0].final_answer, "answer")
