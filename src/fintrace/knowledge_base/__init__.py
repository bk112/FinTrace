"""Versioned financial knowledge-base schema and retrieval adapters."""

from .schema import KnowledgeRecord, KnowledgeRecordError, parse_knowledge_record
from .faiss_retrieval import FaissKnowledgeBase, FaissRetrievalClient

__all__ = [
    "FaissKnowledgeBase",
    "FaissRetrievalClient",
    "KnowledgeRecord",
    "KnowledgeRecordError",
    "parse_knowledge_record",
]
