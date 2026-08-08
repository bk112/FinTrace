"""Engine-agnostic multi-turn ReAct rollout state machine."""

from __future__ import annotations

from fintrace.rewards import SearchStep, Trajectory
from fintrace.rewards.constants import (
    MAX_ROUNDS,
    TAG_ANSWER_CLOSE,
    TAG_SEARCH_CLOSE,
)
from fintrace.tools.base import RetrievalClient

from .parser import ReActActionKind, ReActParseError, extract_final_answer, parse_react_turn
from .types import (
    GenerationEngine,
    RolloutResult,
    RolloutTermination,
    TraceSegment,
)

DEFAULT_STOP_SEQUENCES = (TAG_SEARCH_CLOSE, TAG_ANSWER_CLOSE, "<|im_end|>")


class ReActRollout:
    """Generate, retrieve, and resume until an answer or a safe termination."""

    def __init__(
        self,
        engine: GenerationEngine,
        retrieval_client: RetrievalClient,
        *,
        max_rounds: int = MAX_ROUNDS,
        stop_sequences: tuple[str, ...] = DEFAULT_STOP_SEQUENCES,
    ) -> None:
        self._engine = engine
        self._retrieval_client = retrieval_client
        self._max_rounds = max_rounds
        self._stop_sequences = stop_sequences

    def run(self, prompt: str, ground_truth: str, *, valid_inst: bool = True) -> RolloutResult:
        context = prompt
        assistant_text = ""
        search_steps: list[SearchStep] = []
        retrieval_results = []
        segments: list[TraceSegment] = []
        seen_queries: set[str] = set()

        for round_number in range(1, self._max_rounds + 1):
            try:
                # 在动作闭合标签处截断，才能先执行工具、再把 observation 拼回上下文继续生成。
                generated = self._engine.generate(context, self._stop_sequences)
            except Exception:
                return self._result(
                    assistant_text,
                    ground_truth,
                    search_steps,
                    round_number - 1,
                    RolloutTermination.ENGINE_ERROR,
                    valid_inst,
                    segments,
                    retrieval_results,
                )

            if not generated:
                return self._result(
                    assistant_text,
                    ground_truth,
                    search_steps,
                    round_number - 1,
                    RolloutTermination.MALFORMED_ACTION,
                    valid_inst,
                    segments,
                    retrieval_results,
                )

            assistant_text += generated
            context += generated
            # assistant 生成内容参与 GRPO loss；环境返回内容会在下方以 environment 标记。
            segments.append(TraceSegment(owner="assistant", text=generated))

            try:
                action = parse_react_turn(generated)
            except ReActParseError:
                return self._result(
                    assistant_text,
                    ground_truth,
                    search_steps,
                    round_number,
                    RolloutTermination.MALFORMED_ACTION,
                    valid_inst,
                    segments,
                    retrieval_results,
                )

            if action.kind is ReActActionKind.ANSWER:
                return self._result(
                    assistant_text,
                    ground_truth,
                    search_steps,
                    round_number,
                    RolloutTermination.ANSWER,
                    valid_inst,
                    segments,
                    retrieval_results,
                )

            query = action.content

            normalized_query = query.casefold().strip()
            if not normalized_query or normalized_query in seen_queries:
                # 重复查询既浪费外部调用，也会形成可被 reward 利用的无意义轨迹，直接终止。
                return self._result(
                    assistant_text,
                    ground_truth,
                    search_steps,
                    round_number,
                    RolloutTermination.REPEATED_QUERY,
                    valid_inst,
                    segments,
                    retrieval_results,
                )
            seen_queries.add(normalized_query)

            try:
                retrieval_result = self._retrieval_client.search(query)
            except Exception:
                return self._result(
                    assistant_text,
                    ground_truth,
                    search_steps,
                    round_number,
                    RolloutTermination.TOOL_ERROR,
                    valid_inst,
                    segments,
                    retrieval_results,
                )

            observation = f"<observation>{retrieval_result.text}</observation>"
            context += observation
            # observation 是环境反馈，不应驱动模型学习复述检索网页内容。
            segments.append(TraceSegment(owner="environment", text=observation))
            records = retrieval_result.metadata.get("records", [])
            search_steps.append(
                SearchStep(
                    query=query,
                    retrieved_text=retrieval_result.text,
                    retrieved_records=records if isinstance(records, list) else [],
                )
            )
            retrieval_results.append(retrieval_result)

        return self._result(
            assistant_text,
            ground_truth,
            search_steps,
            self._max_rounds,
            RolloutTermination.MAX_ROUNDS,
            valid_inst,
            segments,
            retrieval_results,
        )

    @staticmethod
    def _result(
        raw_text: str,
        ground_truth: str,
        search_steps: list[SearchStep],
        num_rounds: int,
        termination: RolloutTermination,
        valid_inst: bool,
        segments: list[TraceSegment],
        retrieval_results: list,
    ) -> RolloutResult:
        final_answer = extract_final_answer(raw_text)
        trajectory = Trajectory(
            raw_text=raw_text,
            ground_truth=ground_truth,
            search_steps=search_steps,
            final_answer=final_answer,
            num_rounds=num_rounds,
            tool_calls_in_single_round=1,
            is_repeated_query=termination is RolloutTermination.REPEATED_QUERY,
            valid_inst=valid_inst,
        )
        return RolloutResult(
            trajectory=trajectory,
            termination=termination,
            segments=segments,
            retrieval_results=retrieval_results,
        )
