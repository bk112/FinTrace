"""Types shared by ReAct rollout implementations and training adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Protocol, Sequence

from fintrace.rewards import Trajectory
from fintrace.tools.base import RetrievalResult


class GenerationEngine(Protocol):
    """Minimal inference contract required for multi-turn rollout."""

    def generate(self, prompt: str, stop_sequences: Sequence[str]) -> str:
        """Return generated assistant text including the stop tag that ended it."""


class RolloutTermination(str, Enum):
    ANSWER = "answer"
    MAX_ROUNDS = "max_rounds"
    REPEATED_QUERY = "repeated_query"
    MALFORMED_ACTION = "malformed_action"
    ENGINE_ERROR = "engine_error"
    TOOL_ERROR = "tool_error"


@dataclass(frozen=True)
class TraceSegment:
    """A text span whose ownership determines its later loss mask."""

    owner: Literal["assistant", "environment"]
    text: str


@dataclass
class RolloutResult:
    """Completed or terminated rollout with material needed for reward and loss masks."""

    trajectory: Trajectory
    termination: RolloutTermination
    segments: list[TraceSegment] = field(default_factory=list)
    retrieval_results: list[RetrievalResult] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.termination is RolloutTermination.ANSWER
