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
    """RetrievalClient adapter backed by the local FAISS knowledge base.

    ``include_metadata`` is off by default so the policy-visible observation
    stays proofread-minimal; the reward function still receives every v1.1
    field through ``metadata["records"]``.
    """

    top_k: int = 3
    include_metadata: bool = False

    def search(self, query: str) -> RetrievalResult:
        """Run one query against the local KB.

        Returns:
            RetrievalResult with ``text`` as the formatted observation and
            ``metadata["records"]`` as the full v1.1 structured records
            (used by reward-function exact matching on value_text/value_number).
        """
        records = kb_service.search_records(query, self.top_k)
        text, clean_records = kb_service.format_result(
            query, records, include_metadata=self.include_metadata
        )
        return RetrievalResult(
            query=query,
            text=text,
            metadata={"records": clean_records},
        )
