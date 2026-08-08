"""vLLM implementation of the engine-agnostic generation contract."""

from __future__ import annotations

from typing import Any, Callable, Sequence


class VllmGenerationError(RuntimeError):
    """Raised when vLLM does not return exactly one usable completion."""


class VllmGenerationEngine:
    """Thin vLLM adapter that preserves ReAct stop tags in generated text.

    ``include_stop_str_in_output=True`` is intentional. The rollout state
    machine needs ``</search>`` and ``</answer>`` to parse completed actions.
    """

    def __init__(
        self,
        model_path: str,
        *,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        llm: Any | None = None,
        sampling_params_factory: Callable[..., Any] | None = None,
    ) -> None:
        if llm is None or sampling_params_factory is None:
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:
                raise RuntimeError("vLLM is required for VllmGenerationEngine") from exc
            llm = llm or LLM(
                model=model_path,
                dtype=dtype,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            sampling_params_factory = sampling_params_factory or SamplingParams

        self._llm = llm
        self._sampling_params_factory = sampling_params_factory
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p

    def generate(self, prompt: str, stop_sequences: Sequence[str]) -> str:
        sampling_params = self._sampling_params_factory(
            n=1,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            stop=list(stop_sequences),
            # vLLM 默认会移除 stop 字符串；这里必须保留闭合标签供 ReAct parser 校验。
            include_stop_str_in_output=True,
        )
        outputs = self._llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise VllmGenerationError("vLLM must return one completion per rollout request")
        text = outputs[0].outputs[0].text
        if not isinstance(text, str):
            raise VllmGenerationError("vLLM completion text is not a string")
        return text
