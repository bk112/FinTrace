"""Contract test for the custom rollout output consumed by TRL GRPO."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from fintrace.tools import RetrievalResult
from fintrace.training.react_grpo_rollout import (
    PromptMetadata,
    TransformersReActGRPORollout,
    VllmReActGRPORollout,
)


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return f"user:{messages[0]['content']}\nassistant:"

    def __call__(self, text, **kwargs):
        ids = [ord(char) for char in text]
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, token_ids, *, skip_special_tokens):
        return "".join(chr(token_id) for token_id in token_ids if token_id)


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


class GroupedFakeTrainer:
    num_generations = 2


class FakeTransformersModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, **kwargs):
        del kwargs
        completion = torch.tensor(
            [[ord(char) for char in "<answer>answer</answer>"]], dtype=torch.long
        )
        return torch.cat([input_ids, completion.to(input_ids.device)], dim=1)

    def forward(self, input_ids):
        vocab_size = 2048
        logits = torch.zeros(
            (*input_ids.shape, vocab_size), dtype=torch.float32, device=input_ids.device
        )
        return SimpleNamespace(logits=logits)


class FakeTransformersTrainer:
    num_generations = 1

    def __init__(self) -> None:
        self.model = FakeTransformersModel()


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

    def test_transformers_backend_returns_aligned_sampling_contract(self) -> None:
        rollout = TransformersReActGRPORollout(
            tokenizer=CharacterTokenizer(),
            retrieval_client=FakeRetriever(),
            metadata_by_prompt={"question": PromptMetadata(("answer",), True)},
            max_tokens_per_turn=32,
        )

        output = rollout(["question"], FakeTransformersTrainer())

        self.assertEqual(len(output["completion_ids"][0]), len(output["logprobs"][0]))
        self.assertEqual(len(output["completion_ids"][0]), len(output["env_mask"][0]))
        self.assertEqual(output["trajectories"][0].final_answer, "answer")

    def test_sampler_repeated_prompts_are_not_expanded_twice(self) -> None:
        rollout = VllmReActGRPORollout(
            model_path="unused",
            tokenizer=CharacterTokenizer(),
            retrieval_client=FakeRetriever(),
            metadata_by_prompt={"question": PromptMetadata(("answer",), True)},
            llm=FakeLlm(["<answer>answer</answer>", "<answer>answer</answer>"]),
            sampling_params_factory=lambda **kwargs: kwargs,
        )

        output = rollout(["question", "question"], GroupedFakeTrainer())

        self.assertEqual(len(output["prompt_ids"]), 2)
        self.assertEqual(len(output["completion_ids"]), 2)
        self.assertEqual(len(output["targets"]), 2)
