"""Retrieval boundary used by rollout code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RetrievalResult:
    """Normalized result returned by any retrieval implementation."""

    query: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalClient(Protocol):
    """The only retrieval capability required by the ReAct rollout loop."""

    def search(self, query: str) -> RetrievalResult:
        """Run one query against the existing Tool-aware service."""
