"""Custom vLLM ReAct rollout that returns the contract required by TRL GRPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fintrace.rewards import SearchStep, Trajectory
from fintrace.rewards.constants import MAX_ROUNDS
from fintrace.rollout import ReActActionKind, ReActParseError, parse_react_turn
from fintrace.tools import RetrievalClient


@dataclass(frozen=True)
class PromptMetadata:
    targets: tuple[str, ...]
    valid_inst: bool


@dataclass(frozen=True)
class SampledChunk:
    text: str
    token_ids: list[int]
    logprobs: list[float]


def extract_sampled_logprobs(token_ids: Sequence[int], logprobs: Sequence[Any] | None) -> list[float]:
    """Extract each sampled token's logprob from vLLM's per-position maps."""
    if logprobs is None or len(token_ids) != len(logprobs):
        raise ValueError("vLLM must return one logprob map for every generated token")

    values: list[float] = []
    for token_id, position in zip(token_ids, logprobs, strict=True):
        if position is None or token_id not in position:
            raise ValueError("vLLM response lacks the sampled token logprob")
        value = position[token_id]
        values.append(float(getattr(value, "logprob", value)))
    return values


class VllmReActGRPORollout:
    """Generate multi-turn trajectories and expose TRL's experimental rollout API.

    The rollout keeps exact token IDs from vLLM for assistant chunks. Tool
    observations are tokenized locally, padded with zero logprobs, and masked
    by ``env_mask`` so GRPO never optimizes environment feedback.
    """

    def __init__(
        self,
        *,
        model_path: str,
        tokenizer: Any,
        retrieval_client: RetrievalClient,
        metadata_by_prompt: Mapping[str, PromptMetadata],
        max_rounds: int = MAX_ROUNDS,
        max_tokens_per_turn: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.7,
        llm: Any | None = None,
        sampling_params_factory: Any | None = None,
    ) -> None:
        if llm is None or sampling_params_factory is None:
            from vllm import LLM, SamplingParams

            llm = llm or LLM(
                model=model_path,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            sampling_params_factory = sampling_params_factory or SamplingParams

        self._llm = llm
        self._sampling_params_factory = sampling_params_factory
        self._tokenizer = tokenizer
        self._retrieval_client = retrieval_client
        self._metadata_by_prompt = metadata_by_prompt
        self._max_rounds = max_rounds
        self._max_tokens_per_turn = max_tokens_per_turn
        self._temperature = temperature
        self._top_p = top_p

    def __call__(self, prompts: list[str], trainer: Any) -> dict[str, Any]:
        num_generations = trainer.num_generations
        prompt_ids: list[list[int]] = []
        completion_ids: list[list[int]] = []
        logprobs: list[list[float]] = []
        env_mask: list[list[int]] = []
        trajectories: list[Trajectory] = []
        targets: list[tuple[str, ...]] = []

        for prompt in prompts:
            metadata = self._metadata_by_prompt.get(prompt)
            if metadata is None:
                raise KeyError("rollout prompt is missing dataset metadata")
            rendered_prompt = self._render_prompt(prompt)
            base_prompt_ids = self._encode(rendered_prompt)
            for _ in range(num_generations):
                sample = self._run_one(base_prompt_ids, metadata)
                prompt_ids.append(base_prompt_ids)
                completion_ids.append(sample[0])
                logprobs.append(sample[1])
                env_mask.append(sample[2])
                trajectories.append(sample[3])
                targets.append(metadata.targets)

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "env_mask": env_mask,
            "trajectories": trajectories,
            "targets": targets,
        }

    def _run_one(
        self,
        base_prompt_ids: list[int],
        metadata: PromptMetadata,
    ) -> tuple[list[int], list[float], list[int], Trajectory]:
        context_ids = list(base_prompt_ids)
        completion_ids: list[int] = []
        completion_logprobs: list[float] = []
        env_mask: list[int] = []
        assistant_text = ""
        search_steps: list[SearchStep] = []
        seen_queries: set[str] = set()
        repeated_query = False

        for round_number in range(1, self._max_rounds + 1):
            chunk = self._generate_chunk(context_ids)
            context_ids.extend(chunk.token_ids)
            completion_ids.extend(chunk.token_ids)
            completion_logprobs.extend(chunk.logprobs)
            env_mask.extend([1] * len(chunk.token_ids))
            assistant_text += chunk.text

            try:
                action = parse_react_turn(chunk.text)
            except ReActParseError:
                return self._finished(
                    assistant_text, metadata, search_steps, round_number, repeated_query,
                    completion_ids, completion_logprobs, env_mask,
                )

            if action.kind is ReActActionKind.ANSWER:
                return self._finished(
                    assistant_text, metadata, search_steps, round_number, repeated_query,
                    completion_ids, completion_logprobs, env_mask,
                )

            normalized_query = action.content.casefold().strip()
            if normalized_query in seen_queries:
                repeated_query = True
                return self._finished(
                    assistant_text, metadata, search_steps, round_number, repeated_query,
                    completion_ids, completion_logprobs, env_mask,
                )
            seen_queries.add(normalized_query)

            try:
                retrieved = self._retrieval_client.search(action.content)
            except Exception:
                return self._finished(
                    assistant_text, metadata, search_steps, round_number, repeated_query,
                    completion_ids, completion_logprobs, env_mask,
                )

            observation = f"<observation>{retrieved.text}</observation>"
            observation_ids = self._encode(observation)
            context_ids.extend(observation_ids)
            completion_ids.extend(observation_ids)
            # observation 不是策略模型采样出的 token：logprob 填零，并通过 env_mask 完全屏蔽 loss。
            completion_logprobs.extend([0.0] * len(observation_ids))
            env_mask.extend([0] * len(observation_ids))
            records = retrieved.metadata.get("records", [])
            search_steps.append(
                SearchStep(
                    query=action.content,
                    retrieved_text=retrieved.text,
                    retrieved_records=records if isinstance(records, list) else [],
                )
            )

        return self._finished(
            assistant_text, metadata, search_steps, self._max_rounds + 1, repeated_query,
            completion_ids, completion_logprobs, env_mask,
        )

    def _generate_chunk(self, context_ids: list[int]) -> SampledChunk:
        params = self._sampling_params_factory(
            n=1,
            max_tokens=self._max_tokens_per_turn,
            temperature=self._temperature,
            top_p=self._top_p,
            logprobs=0,
            stop=["</search>", "</answer>", "<|im_end|>"],
            include_stop_str_in_output=True,
        )
        # 用 token IDs 续写而非重新拼字符串，可保证训练时的 token 序列与采样上下文严格一致。
        outputs = self._llm.generate(
            [{"prompt_token_ids": context_ids}], sampling_params=params, use_tqdm=False
        )
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError("vLLM must return one completion per ReAct turn")
        output = outputs[0].outputs[0]
        token_ids = list(output.token_ids)
        return SampledChunk(
            text=output.text,
            token_ids=token_ids,
            logprobs=extract_sampled_logprobs(token_ids, output.logprobs),
        )

    def _render_prompt(self, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _encode(self, text: str) -> list[int]:
        return list(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    @staticmethod
    def _finished(
        raw_text: str,
        metadata: PromptMetadata,
        search_steps: list[SearchStep],
        num_rounds: int,
        repeated_query: bool,
        completion_ids: list[int],
        completion_logprobs: list[float],
        env_mask: list[int],
    ) -> tuple[list[int], list[float], list[int], Trajectory]:
        answer_start = raw_text.rfind("<answer>")
        answer_end = raw_text.rfind("</answer>")
        final_answer = (
            raw_text[answer_start + len("<answer>") : answer_end].strip()
            if answer_start >= 0 and answer_end > answer_start
            else None
        )
        trajectory = Trajectory(
            raw_text=raw_text,
            ground_truth=metadata.targets[0],
            search_steps=search_steps,
            final_answer=final_answer,
            num_rounds=num_rounds,
            tool_calls_in_single_round=1,
            is_repeated_query=repeated_query,
            valid_inst=metadata.valid_inst,
        )
        return completion_ids, completion_logprobs, env_mask, trajectory
