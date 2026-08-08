"""Strict parser for one completed ReAct generation turn."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ReActParseError(ValueError):
    """Raised when generated text is not one safe, executable ReAct action."""


class ReActActionKind(str, Enum):
    SEARCH = "search"
    ANSWER = "answer"


@dataclass(frozen=True)
class ReActAction:
    kind: ReActActionKind
    content: str
    thought: str


_TURN_PATTERN = re.compile(
    r"^\s*<think>(?P<thought>[^<]*)</think>\s*"
    r"(?:(?:<search>(?P<query>[^<]*)</search>)|(?:<answer>(?P<answer>[^<]*)</answer>))\s*$",
    re.DOTALL,
)


def parse_react_turn(text: str) -> ReActAction:
    """Parse exactly one completed ``think -> search|answer`` generation turn."""
    match = _TURN_PATTERN.fullmatch(text)
    if match is None:
        raise ReActParseError("turn must contain one closed think block and one closed search or answer block")

    thought = match.group("thought").strip()
    query = match.group("query")
    answer = match.group("answer")
    if not thought:
        raise ReActParseError("think block must be non-empty")
    if query is not None:
        query = query.strip()
        if not query:
            raise ReActParseError("search block must be non-empty")
        return ReActAction(kind=ReActActionKind.SEARCH, content=query, thought=thought)

    if answer is None or not answer.strip():
        raise ReActParseError("answer block must be non-empty")
    return ReActAction(kind=ReActActionKind.ANSWER, content=answer.strip(), thought=thought)


def extract_final_answer(raw_text: str) -> str | None:
    """Extract the last closed answer from a complete assistant trajectory."""
    matches = re.findall(r"<answer>(.*?)</answer>", raw_text, re.DOTALL)
    return matches[-1].strip() if matches and matches[-1].strip() else None
