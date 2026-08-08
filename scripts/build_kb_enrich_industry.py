#!/usr/bin/env python3
"""Enrich an existing merged KB jsonl with the `industry` field (idempotent).

Applies infer_industry() to every record — no re-scraping needed. For
FinanceComplexQA records the doc_type is recovered from source_doc
("FinanceComplexQA/<doc_type> ...") so DOC_TYPE_TO_INDUSTRY applies.

Usage: python scripts/build_kb_enrich_industry.py [path-to-jsonl]
Default: data/kb/financial_knowledge_base.jsonl (rewritten in place)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from fintrace.kb.company_pool import infer_industry

DEFAULT = Path(__file__).resolve().parent.parent / "data" / "kb" / "financial_knowledge_base.jsonl"
DOC_TYPE_RE = re.compile(r"FinanceComplexQA/([A-Za-z_]+)")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    assert path.exists(), f"not found: {path}"

    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    enriched = added = 0
    for rec in records:
        doc_type = ""
        m = DOC_TYPE_RE.search(rec.get("source_doc", ""))
        if m:
            doc_type = m.group(1)
        industry = infer_industry(
            rec.get("entity", ""),
            source_type=rec.get("source_type", ""),
            doc_type=doc_type,
        )
        if rec.get("industry") != industry:
            added += 1
        rec["industry"] = industry
        enriched += 1

    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    by_industry = Counter(r["industry"] for r in records)
    print(f"富化完成: {enriched} 条记录, 新增/变更 industry 字段 {added} 条")
    print(f"行业分布: {dict(sorted(by_industry.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
