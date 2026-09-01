"""Financial knowledge-base retrieval service (v1.1).

Two entry points per the spec:
  - search_records(query, top_k) → list[dict]   (structured records, for difficulty synthesizer)
  - retrieve(query)              → str (formatted for model context, for RL rollout)

The policy-visible text carries only entity / metric / value_text / source_doc.
Full v1.1 records reach the reward function out of band through
``RetrievalResult.metadata["records"]``; pass ``include_metadata=True`` to inline
them for debugging.

Shared index & records loaded once at module import time.
"""

from .client import KbRetrievalClient
from .service import format_result, retrieve, sample_seed_facts, search_records

__all__ = [
    "KbRetrievalClient",
    "format_result",
    "retrieve",
    "sample_seed_facts",
    "search_records",
]
