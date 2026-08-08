"""Financial knowledge-base retrieval service (v1.1).

Two entry points per the spec:
  - search_records(query, top_k) → list[dict]   (structured records, for difficulty synthesizer)
  - retrieve(query)              → str + metadata (formatted for model context, for RL rollout)

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
