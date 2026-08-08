"""FAISS-backed local retrieval adapter for the canonical financial knowledge base."""

from __future__ import annotations

import json
from pathlib import Path

from fintrace.tools.base import RetrievalResult

from .schema import KnowledgeRecord, parse_knowledge_record


class FaissKnowledgeBase:
    """Loads an immutable records snapshot and its order-matched FAISS index."""

    def __init__(self, index_path: Path, records_path: Path, embedding_model_path: str) -> None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("faiss-cpu and sentence-transformers are required for FAISS retrieval") from exc

        self._index = faiss.read_index(str(index_path))
        self._model = SentenceTransformer(embedding_model_path)
        with records_path.open("r", encoding="utf-8") as handle:
            self._records = [parse_knowledge_record(json.loads(line)) for line in handle if line.strip()]
        if self._index.ntotal != len(self._records):
            raise ValueError("FAISS vector count must equal records snapshot length")

    def search_records(self, query: str, top_k: int = 3) -> list[KnowledgeRecord]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        limit = min(top_k, len(self._records))
        embeddings = self._model.encode([query], normalize_embeddings=True)
        _, indices = self._index.search(embeddings.astype("float32"), limit)
        return [self._records[index] for index in indices[0] if index >= 0]


class FaissRetrievalClient:
    """Converts structured FAISS hits into the text-plus-metadata rollout contract."""

    def __init__(self, knowledge_base: FaissKnowledgeBase, top_k: int = 3) -> None:
        self._knowledge_base = knowledge_base
        self._top_k = top_k

    def search(self, query: str) -> RetrievalResult:
        records = self._knowledge_base.search_records(query, self._top_k)
        return RetrievalResult(
            query=query,
            text=self._format_records(records),
            metadata={"records": [record.to_dict() for record in records]},
        )

    @staticmethod
    def _format_records(records: list[KnowledgeRecord]) -> str:
        return "\n\n".join(
            f"[fact {index}] {record.fact}\nsource: {record.source_doc}\nurl: {record.source_url}"
            for index, record in enumerate(records, start=1)
        )
