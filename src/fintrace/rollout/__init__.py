"""Multi-turn ReAct rollout generation and trajectory assembly."""

from .masking import TokenizedTrace, tokenize_trace_with_env_mask
from .parser import ReActAction, ReActActionKind, ReActParseError, parse_react_turn
from .react import ReActRollout
from .types import RolloutResult, RolloutTermination, TraceSegment
from .vllm_engine import VllmGenerationEngine

__all__ = [
    "ReActRollout",
    "RolloutResult",
    "RolloutTermination",
    "TraceSegment",
    "VllmGenerationEngine",
    "TokenizedTrace",
    "tokenize_trace_with_env_mask",
    "ReActAction",
    "ReActActionKind",
    "ReActParseError",
    "parse_react_turn",
]
