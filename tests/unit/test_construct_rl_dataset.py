"""Regression tests for streamed Tencent API parsing without making network calls."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "construct_rl_dataset.py"
SPEC = importlib.util.spec_from_file_location("construct_rl_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


class FakeResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, *, decode_unicode: bool):
        assert not decode_unicode
        return iter(self._lines)


def test_streaming_client_decodes_chinese_sse_as_utf8(monkeypatch) -> None:
    event = {"choices": [{"delta": {"content": "是"}}]}
    lines = [f"data: {json.dumps(event, ensure_ascii=False)}".encode("utf-8"), b"data: [DONE]"]
    monkeypatch.setattr(SCRIPT.requests, "post", lambda *args, **kwargs: FakeResponse(lines))

    assert SCRIPT.DeepSeekFlashClient("test-key", 1).complete("test") == "是"
