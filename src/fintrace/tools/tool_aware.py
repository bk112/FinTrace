"""HTTP adapter for the existing Tool-aware retrieval service.

The real service contract is intentionally configurable. This module owns only
transport, timeouts, and response normalization; it does not reimplement search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import RetrievalResult


class ToolAwareRequestError(RuntimeError):
    """Raised when the external Tool-aware service cannot provide a result."""


@dataclass(frozen=True)
class ToolAwareHttpClient:
    endpoint: str
    timeout_seconds: float = 30.0
    query_field: str = "query"
    response_text_field: str = "text"

    def search(self, query: str) -> RetrievalResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must be non-empty")

        body = json.dumps({self.query_field: normalized_query}).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise ToolAwareRequestError(f"Tool-aware returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise ToolAwareRequestError(f"Tool-aware request failed: {exc.reason}") from exc

        try:
            payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ToolAwareRequestError("Tool-aware response is not valid JSON") from exc

        text, metadata = self._normalize_payload(payload)
        return RetrievalResult(query=normalized_query, text=text, metadata=metadata)

    def _normalize_payload(self, payload: Any) -> tuple[str, dict[str, Any]]:
        if isinstance(payload, str):
            return payload, {}
        if not isinstance(payload, Mapping):
            raise ToolAwareRequestError("Tool-aware JSON response must be an object or string")

        text = payload.get(self.response_text_field)
        if not isinstance(text, str):
            raise ToolAwareRequestError(
                f"Tool-aware response is missing string field {self.response_text_field!r}"
            )
        return text, {key: value for key, value in payload.items() if key != self.response_text_field}
