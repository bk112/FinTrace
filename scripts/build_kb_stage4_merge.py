#!/usr/bin/env python3
"""Stage 4: Merge & deduplicate scraped + opensource records → financial_knowledge_base.jsonl + report.

Data split (v1.1 §5): train / val / test partition on (entity, metric) relation pairs,
NOT on separate knowledge bases.

Output:
  data/kb/financial_knowledge_base.jsonl
  data/kb/data_stats_report.md
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

from fintrace.kb.company_pool import infer_industry

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KB_PATH = OUTPUT_DIR / "financial_knowledge_base.jsonl"
REPORT_PATH = OUTPUT_DIR / "data_stats_report.md"

RAW_PATH = OUTPUT_DIR / "raw_financial_facts.jsonl"
SUPP_PATH = OUTPUT_DIR / "opensource_supplement_facts.jsonl"
ANNO_PATH = OUTPUT_DIR / "raw_announcements_supplement.jsonl"

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
DOC_TYPE_RE = re.compile(r"FinanceComplexQA/([A-Za-z_]+)")

log = logging.getLogger("kb_merge")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())


def _load(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        log.warning("文件不存在: %s", path)
        return records
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("跳过无效 JSON 行: %s", line[:80])
    return records


def _dedup_key(rec: dict) -> str:
    """Return a semantic duplicate key without collapsing conflicting values.

    Financial reports can legitimately publish the same entity/metric/date with
    different values (for example, a restatement or two analyst forecasts). Such
    conflicts must remain observable to retrieval and evaluation.
    """
    entity = rec.get("entity", "")
    metric = rec.get("metric", "")
    date = rec.get("date", "")
    if entity and metric:
        return json.dumps(
            [entity, metric, date, rec.get("value_text", ""), rec.get("unit", "")],
            ensure_ascii=False,
            sort_keys=False,
        )
    # Fallback: hash of fact text
    return hashlib.sha256(rec.get("fact", "").encode()).hexdigest()[:20]


def _fact_text_similarity(a: str, b: str) -> float:
    """Simple Jaccard on character trigrams for near-duplicate detection."""
    if not a or not b:
        return 0.0
    grams_a = {a[i : i + 3] for i in range(len(a) - 2)}
    grams_b = {b[i : i + 3] for i in range(len(b) - 2)}
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)


def _validate(rec: dict) -> list[str]:
    """Check record completeness; return list of field issues."""
    issues: list[str] = []
    required = ["fact_id", "fact", "entity", "source_type"]
    for field in required:
        if not rec.get(field):
            issues.append(f"missing_{field}")
    # v1.1: amount/percentage MUST have value_number
    if rec.get("value_type") in ("amount", "percentage") and rec.get("value_number") is None:
        issues.append("missing_value_number")
    return issues


def _assign_segment(entity: str, metric: str, salt: str = "fintrace_kb_v1") -> str:
    """Deterministic train/val/test split on (entity, metric) relation."""
    key = f"{entity}|{metric}|{salt}"
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    bucket = h % 100
    if bucket < 80:
        return "train"
    elif bucket < 90:
        return "val"
    else:
        return "test"


def _enrich_industry(rec: dict) -> None:
    """Ensure every final record has a stable industry field in the same build pass."""
    if rec.get("industry"):
        return
    match = DOC_TYPE_RE.search(rec.get("source_doc", ""))
    doc_type = match.group(1) if match else ""
    rec["industry"] = infer_industry(rec.get("entity", ""), doc_type=doc_type)


def main() -> None:
    log.info("===== 阶段四：数据合并与去重 =====")

    # Load
    raw = _load(RAW_PATH)
    supp = _load(SUPP_PATH)
    anno = _load(ANNO_PATH)
    log.info("原始记录(scrape): %d 条", len(raw))
    log.info("补充记录(opensource): %d 条", len(supp))
    log.info("公告补充: %d 条", len(anno))

    all_records = raw + supp + anno

    # Phase 1: exact semantic dedup. Value is part of the key to retain conflicts.
    seen_keys: dict[str, dict] = {}
    exact_dups = 0
    for rec in all_records:
        key = _dedup_key(rec)
        if key in seen_keys:
            # Same fact/value from multiple copies: keep the more descriptive text.
            existing = seen_keys[key]
            if len(rec.get("fact", "")) > len(existing.get("fact", "")):
                seen_keys[key] = rec
            exact_dups += 1
        else:
            seen_keys[key] = rec
    log.info("精确去重: 移除 %d 条, 剩余 %d 条", exact_dups, len(seen_keys))

    # Phase 2: remove only byte-identical fact text. Character-similarity based
    # removal is unsafe here: two different financial values often differ by a
    # handful of characters but are highly similar otherwise.
    deduped: list[dict] = list(seen_keys.values())
    near_dup_count = 0
    # Group by entity first to reduce cross-check cost
    by_entity: dict[str, list[int]] = {}
    for idx, rec in enumerate(deduped):
        ent = rec.get("entity", "__unknown__")
        by_entity.setdefault(ent, []).append(idx)

    mask = [True] * len(deduped)
    for ent, indices in by_entity.items():
        for i_idx in range(len(indices)):
            i = indices[i_idx]
            if not mask[i]:
                continue
            fi = deduped[i].get("fact", "")
            for j_idx in range(i_idx + 1, len(indices)):
                j = indices[j_idx]
                if not mask[j]:
                    continue
                fj = deduped[j].get("fact", "")
                if fi == fj:
                    mask[j] = False
                    near_dup_count += 1

    deduped = [rec for idx, rec in enumerate(deduped) if mask[idx]]
    log.info("完全相同文本去重: 移除 %d 条, 剩余 %d 条", near_dup_count, len(deduped))

    # Validate
    valid: list[dict] = []
    validation_issues: Counter = Counter()
    for rec in deduped:
        _enrich_industry(rec)
        issues = _validate(rec)
        if not issues:
            rec["segment"] = _assign_segment(
                rec.get("entity", ""), rec.get("metric", "")
            )
            valid.append(rec)
        else:
            for iss in issues:
                validation_issues[iss] += 1
    log.info("校验: 通过 %d 条, 过滤 %d 条", len(valid), len(deduped) - len(valid))
    if validation_issues:
        log.info("  过滤原因: %s", dict(validation_issues.most_common()))

    # Write
    with open(KB_PATH, "w", encoding="utf-8") as fh:
        for rec in valid:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("输出: %s (%d 条)", KB_PATH, len(valid))

    # -------------------------------------------------------------------
    # Statistics report
    # -------------------------------------------------------------------
    source_counts: Counter = Counter()
    entity_counts: Counter = Counter()
    structured_count = 0
    value_types: Counter = Counter()
    segments: Counter = Counter()

    for rec in valid:
        source_counts[rec.get("source_type", "unknown")] += 1
        entity_counts[rec.get("entity", "unknown")] += 1
        value_types[rec.get("value_type", "text")] += 1
        segments[rec.get("segment", "train")] += 1
        if rec.get("structured", False):
            structured_count += 1

    # Industry inference from company pool (fallback: from entity name)
    industry_map = {}
    for rec in valid:
        ent = rec.get("entity", "")
        if ent not in industry_map:
            industry_map[ent] = set()
        # We'll collect later via the scraping script's company list
        # For now, just count entities

    report_lines = [
        "# 金融知识库数据统计报告",
        "",
        f"生成时间: {__import__('datetime').datetime.now().isoformat()}",
        "",
        "## 总体概况",
        "",
        f"- **总记录数**: {len(valid)}",
        f"- **覆盖实体数**: {len(entity_counts)}",
        f"- **覆盖数据源类型**: {len(source_counts)}",
        f"- **结构化记录占比**: {structured_count}/{len(valid)} ({structured_count*100//max(len(valid),1)}%)",
        "",
        "## 按来源类型分布",
        "",
        "| 来源类型 | 记录数 | 占比 |",
        "|----------|--------|------|",
    ]
    for src, cnt in source_counts.most_common():
        pct = f"{cnt * 100 / len(valid):.1f}%"
        report_lines.append(f"| {src} | {cnt} | {pct} |")

    report_lines += [
        "",
        "## 按 value_type 分布",
        "",
        "| 类型 | 数量 |",
        "|------|------|",
    ]
    for vt, cnt in value_types.most_common():
        report_lines.append(f"| {vt} | {cnt} |")

    report_lines += [
        "",
        "## 数据集切分",
        "",
        "| 切分 | 记录数 | 比例 |",
        "|------|--------|------|",
    ]
    for seg in ["train", "val", "test"]:
        cnt = segments.get(seg, 0)
        pct = f"{cnt * 100 / max(len(valid), 1):.1f}%"
        report_lines.append(f"| {seg} | {cnt} | {pct} |")

    report_lines += [
        "",
        "## Top 20 实体 (按记录数)",
        "",
        "| 实体 | 记录数 |",
        "|------|--------|",
    ]
    for ent, cnt in entity_counts.most_common(20):
        report_lines.append(f"| {ent} | {cnt} |")

    report_lines += [
        "",
        "## 字段完整性",
        "",
        "| 字段 | 完整率 |",
        "|------|--------|",
    ]
    # Check key fields across all records
    field_names = ["fact_id", "fact", "entity", "metric", "value_text",
                   "source_url", "date", "published_at"]
    for field in field_names:
        filled = sum(1 for r in valid if r.get(field))
        pct = f"{filled * 100 // max(len(valid), 1)}%"
        report_lines.append(f"| {field} | {pct} |")

    report_lines += [
        "",
        "## 风险与边界",
        "",
        "- 数据来源：AkShare（东方财富/同花顺聚合） + FinanceComplexQA 开源数据集",
        f"- 抓取时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d')}",
        f"- 覆盖面：{len(entity_counts)} 家公司，{len(source_counts)} 类数据源",
        "- 数据稳定性：AkShare 为爬虫聚合接口，字段偶有不稳定是已知风险",
        "- 数值精度：value_number 字段为自动解析，存在一定的解析错误率，人工抽查时请注意核对",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines) + "\n")
    log.info("报告: %s", REPORT_PATH)

    log.info("===== 阶段四完成 =====")


if __name__ == "__main__":
    main()
