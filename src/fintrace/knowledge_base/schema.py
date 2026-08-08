"""Auditable schema for financial facts used by synthesis and rollout retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


class KnowledgeRecordError(ValueError):
    """Raised when a source fact cannot safely enter the canonical knowledge base."""


ValueType = Literal["amount", "percentage", "ratio", "rank", "entity", "text"]


@dataclass(frozen=True)
class KnowledgeRecord:
    """One fact with both retrieval text and verifiable, normalized provenance."""

    fact_id: str
    fact: str
    source_doc: str
    source_url: str
    source_type: str
    document_id: str
    published_at: str
    retrieved_at: str
    raw_content_hash: str
    entity: str
    entity_type: str
    metric: str
    value_text: str
    value_type: ValueType
    period: str
    value_number: float | None = None
    unit: str | None = None
    currency: str | None = None
    scale: str | None = None
    industry: str | None = None
    related_entities: tuple[str, ...] = field(default_factory=tuple)
    structured: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["related_entities"] = list(self.related_entities)
        return data


_REQUIRED_TEXT_FIELDS = (
    "fact_id",
    "fact",
    "source_doc",
    "source_type",
    "document_id",
    "published_at",
    "retrieved_at",
    "raw_content_hash",
    "entity",
    "metric",
    "value_text",
    "value_type",
)
_VALUE_TYPES = {"amount", "percentage", "ratio", "rank", "entity", "text"}


def parse_knowledge_record(record: Mapping[str, Any]) -> KnowledgeRecord:
    """Validate both legacy records and the current v1.1 KB record shape."""
    for name in _REQUIRED_TEXT_FIELDS:
        value = record.get(name)
        if not isinstance(value, str) or not value.strip():
            raise KnowledgeRecordError(f"{name} must be a non-empty string")
    source_url = record.get("source_url", "")
    if source_url and not source_url.startswith(("https://", "http://")):
        raise KnowledgeRecordError("source_url must be empty or an HTTP(S) URL")
    if record["value_type"] not in _VALUE_TYPES:
        raise KnowledgeRecordError("value_type is not supported")

    value_number = record.get("value_number")
    if value_number is not None and not isinstance(value_number, (int, float)):
        raise KnowledgeRecordError("value_number must be numeric or null")
    if record["value_type"] in {"amount", "percentage"} and value_number is None:
        raise KnowledgeRecordError("amount and percentage facts require value_number")

    related_entities = record.get("related_entities", [])
    if not isinstance(related_entities, list) or any(
        not isinstance(entity, str) or not entity.strip() for entity in related_entities
    ):
        raise KnowledgeRecordError("related_entities must be a list of non-empty strings")
    structured = record.get("structured", True)
    if not isinstance(structured, bool):
        raise KnowledgeRecordError("structured must be a boolean")

    optional_text = ("unit", "currency", "scale", "industry")
    optional_values = {}
    for name in optional_text:
        value = record.get(name)
        if value is not None and not isinstance(value, str):
            raise KnowledgeRecordError(f"{name} must be a string or null")
        optional_values[name] = value.strip() if isinstance(value, str) else None

    return KnowledgeRecord(
        fact_id=record["fact_id"].strip(),
        fact=record["fact"].strip(),
        source_doc=record["source_doc"].strip(),
        source_url=str(source_url).strip(),
        source_type=record["source_type"].strip(),
        document_id=record["document_id"].strip(),
        published_at=record["published_at"].strip(),
        retrieved_at=record["retrieved_at"].strip(),
        raw_content_hash=record["raw_content_hash"].strip(),
        entity=record["entity"].strip(),
        entity_type=str(record.get("entity_type", "company")).strip() or "company",
        metric=record["metric"].strip(),
        value_text=record["value_text"].strip(),
        value_type=record["value_type"],
        period=str(record.get("period", record.get("date", ""))).strip(),
        value_number=float(value_number) if value_number is not None else None,
        related_entities=tuple(entity.strip() for entity in related_entities),
        structured=structured,
        **optional_values,
    )
