#!/usr/bin/env python3
"""Serve the local Tool-aware retriever over the rollout HTTP contract."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fintrace.tools.local_tool_aware import LocalToolAwareConfig, LocalToolAwareRetriever
from fintrace.tools.local_web import WebRetrievalError


def make_handler(retriever: LocalToolAwareRetriever):
    class ToolAwareHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._write_json(HTTPStatus.OK, {"status": "ok"})

        def do_POST(self) -> None:
            if self.path != "/retrieve":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                query = payload["query"]
                if not isinstance(query, str):
                    raise ValueError("query must be a string")
                result = retriever.search(query)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except WebRetrievalError as exc:
                self._write_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
                return

            self._write_json(
                HTTPStatus.OK,
                {
                    "text": result.text,
                    "sources": result.metadata.get("sources", []),
                    "strategy": result.metadata.get("strategy"),
                    "cached": result.metadata.get("cached", False),
                },
            )

        def log_message(self, format: str, *args) -> None:
            return

        def _write_json(self, status: HTTPStatus, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ToolAwareHandler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim/retrieval_cache"))
    parser.add_argument("--search-limit", type=int, default=6)
    parser.add_argument("--document-limit", type=int, default=3)
    parser.add_argument("--preferred-domain", action="append", default=[])
    args = parser.parse_args()

    defaults = LocalToolAwareConfig(cache_dir=args.cache_dir)
    preferred_domains = tuple(args.preferred_domain) or defaults.preferred_domains
    retriever = LocalToolAwareRetriever(
        LocalToolAwareConfig(
            cache_dir=args.cache_dir,
            search_limit=args.search_limit,
            document_limit=args.document_limit,
            preferred_domains=preferred_domains,
        )
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(retriever))
    print(f"Tool-aware listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
