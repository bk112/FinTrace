"""In-process knowledge-base retrieval client for RL rollout loops.

Implements the :class:`fintrace.tools.RetrievalClient` protocol so the KB can
be swapped in for the HTTP tool-aware client in train_grpo.py without touching
rollout code. The embedding model + FAISS index load lazily on first search.
"""

from __future__ import annotations

from dataclasses import dataclass

from fintrace.tools.base import RetrievalResult

from . import service as kb_service


@dataclass(frozen=True)
class KbRetrievalClient:
    """RetrievalClient adapter backed by the local FAISS knowledge base."""

    top_k: int = 3

    def search(self, query: str) -> RetrievalResult:
        """Run one query against the local KB.

        Returns:
            RetrievalResult with ``text`` as the formatted observation and
            ``metadata["records"]`` as the full v1.1 structured records
            (used by reward-function exact matching on value_text/value_number).
        """
        records = kb_service.search_records(query, self.top_k)
        text, clean_records = kb_service.format_result(query, records)
        return RetrievalResult(
            query=query,
            text=text,
            metadata={"records": clean_records},
        )
