"""Adapters for the existing Tool-aware retrieval service."""

from .base import RetrievalClient, RetrievalResult
from .local_tool_aware import LocalToolAwareConfig, LocalToolAwareRetriever
from .tool_aware import ToolAwareHttpClient, ToolAwareRequestError

__all__ = [
    "RetrievalClient",
    "RetrievalResult",
    "LocalToolAwareConfig",
    "LocalToolAwareRetriever",
    "ToolAwareHttpClient",
    "ToolAwareRequestError",
]
