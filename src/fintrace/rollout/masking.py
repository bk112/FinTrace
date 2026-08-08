"""Token-level loss masks for mixed assistant and environment traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .types import TraceSegment


class OffsetTokenizer(Protocol):
    """Tokenizer capability needed to assign text ownership to token IDs."""

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        """Return input IDs and character offset mappings for a single text."""


@dataclass(frozen=True)
class TokenizedTrace:
    """Completion IDs and the corresponding assistant-only loss mask."""

    text: str
    input_ids: list[int]
    env_mask: list[int]


def tokenize_trace_with_env_mask(
    tokenizer: OffsetTokenizer,
    segments: Sequence[TraceSegment],
) -> TokenizedTrace:
    """Tokenize a trace once and assign 1 to assistant tokens, 0 to environment tokens.

    Tokenizing the full completion once avoids boundary artifacts introduced by
    independently tokenizing each segment. A token crossing an ownership
    boundary is rejected rather than silently assigned an incorrect loss mask.
    """
    text = "".join(segment.text for segment in segments)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if len(input_ids) != len(offsets):
        raise ValueError("tokenizer returned unequal input_ids and offset_mapping lengths")

    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for segment in segments:
        end = cursor + len(segment.text)
        spans.append((cursor, end, 1 if segment.owner == "assistant" else 0))
        cursor = end

    env_mask: list[int] = []
    for start, end in offsets:
        if start == end:
            raise ValueError("zero-width token offset cannot be assigned an ownership mask")
        owners = {owner for span_start, span_end, owner in spans if start >= span_start and end <= span_end}
        if len(owners) != 1:
            raise ValueError("token offset crosses a trace ownership boundary")
        env_mask.append(owners.pop())

    return TokenizedTrace(text=text, input_ids=input_ids, env_mask=env_mask)
