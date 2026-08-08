#!/usr/bin/env python3
"""Stage 3: Download & process FinanceComplexQA → structured supplement records (v1.1 schema).

Output: data/kb/opensource_supplement_facts.jsonl
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import date as dt_date
from pathlib import Path

RETRIEVED_AT = dt_date.today().isoformat()
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "opensource_supplement_facts.jsonl"

log = logging.getLogger("kb_opensource")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Minimal entity + metric + value extraction (best-effort for unstructured text)
# ---------------------------------------------------------------------------

# Crude patterns to extract company mentions
ENTITY_PATTERN = re.compile(
    r"(贵州茅台|五粮液|泸州老窖|招商银行|工商银行|平安银行|恒瑞医药|迈瑞医疗|云南白药"
    r"|海康威视|汇川技术|紫光国微|宁德时代|隆基绿能|TCL中环|比亚迪|长城汽车|长安汽车"
    r"|中国平安|中国人寿|万科|保利发展|美的集团|格力电器|海尔智家|中芯国际|韦尔股份"
    r"|伊利股份|牧原股份|万华化学|龙佰集团|中兴通讯|传音控股|中航沈飞|航天电器"
    r"|长江电力|中国广核|(?:[一-龥]{2,6}(?:集团|股份|控股|科技|银行|保险|汽车|能源|医药|电力|通信)))"
)
METRIC_PATTERNS = [
    (re.compile(r"(?:营业(?:总)?收入|营收)[约达]?[0-9,.]+[万亿元百千万亿]*"), "营业总收入"),
    (re.compile(r"(?:净利润|净利)[约达]?[0-9,.]+[万亿元百千万亿]*"), "净利润"),
    (re.compile(r"(?:同比)?(?:增长|增速|下滑|下降|上升)[0-9,.]+%"), "同比增速"),
    (re.compile(r"(?:毛利率|净利率)[0-9,.]+%"), "利润率"),
    (re.compile(r"(?:市(?:场)?占(?:有)?率|份额)[约达]?[0-9,.]+%"), "市占率"),
    (re.compile(r"(?:PE|市盈率)[约达]?[0-9,.]+倍?"), "市盈率"),
    (re.compile(r"(?:总资产|资产)[约达]?[0-9,.]+[万亿元百千万亿]*"), "总资产"),
    (re.compile(r"(?:ROE|净资产收益率)[0-9,.]+%"), "净资产收益率"),
]
NUMBER_PATTERN = re.compile(r"[0-9,.]+")


def _find_entity(text: str) -> str | None:
    m = ENTITY_PATTERN.search(text)
    return m.group(1) if m else None


def _find_metric_and_value(text: str) -> tuple[str, str] | None:
    for pat, metric_name in METRIC_PATTERNS:
        m = pat.search(text)
        if m:
            raw = m.group(0)
            num_m = NUMBER_PATTERN.search(raw)
            if num_m:
                return metric_name, raw
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("===== 阶段三：开源数据集补充 =====")

    # Try HF mirror first, fall back to official
    hf_endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if not hf_endpoint:
        # Default to official (mirror was unreachable during setup)
        log.info("HF_ENDPOINT not set, using https://huggingface.co")

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets 未安装。请先执行: pip install datasets")
        return

    dataset_name = "Multilingual-Multimodal-NLP/FinanceComplexQA"
    records: list[dict] = []

    try:
        log.info("正在下载 %s ...", dataset_name)
        ds = load_dataset(dataset_name, split="train")
        log.info("数据集下载成功: %d 条", len(ds))
    except Exception as exc:
        log.error("数据集下载失败: %s", exc)
        log.error(
            "如果该数据集需要 HuggingFace 申请 access 权限，请在 https://huggingface.co/datasets/%s 完成申请后重试",
            dataset_name,
        )
        # Write empty output so downstream stages don't crash
        with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
            pass
        return

    for i, sample in enumerate(ds):
        # Filter: Chinese subset only
        lang = sample.get("language", sample.get("lang", ""))
        if isinstance(lang, str) and lang.lower() not in ("zh", "chinese", "cn", ""):
            continue

        # Extract reference_documents
        ref_docs = sample.get("reference_documents")
        if not ref_docs:
            continue

        # Normalise to list
        if isinstance(ref_docs, str):
            ref_docs = [ref_docs]
        elif isinstance(ref_docs, dict):
            ref_docs = list(ref_docs.values())

        question = sample.get("question", sample.get("query", ""))

        for doc_idx, doc_text in enumerate(ref_docs):
            if not isinstance(doc_text, str) or len(doc_text.strip()) < 20:
                continue

            entity = _find_entity(doc_text)

            # Try to extract structured metric+value
            mv = _find_metric_and_value(doc_text)
            metric = mv[0] if mv else ""
            value_text = mv[1] if mv else doc_text[:200]
            structured = bool(mv)

            fact_id = f"fcqa_{_hash(doc_text[:500])}"
            source_doc = f"FinanceComplexQA #{i}"
            if question:
                source_doc += f" (Q: {str(question)[:80]})"

            rec = {
                "fact_id": fact_id,
                "fact": doc_text.strip()[:500],
                "source_doc": source_doc,
                "source_url": f"https://huggingface.co/datasets/{dataset_name}",
                "source_type": "opensource_dataset",
                "document_id": f"fcqa_sample_{i}_doc_{doc_idx}",
                "entity": entity or "未知",
                "metric": metric,
                "value_text": value_text,
                "value_number": None,
                "value_type": "text",
                "unit": "",
                "currency": None,
                "scale": "",
                "date": "",
                "published_at": "",
                "retrieved_at": RETRIEVED_AT,
                "raw_content_hash": _hash(doc_text[:500]),
                "structured": structured,
            }
            records.append(rec)

        if (i + 1) % 2000 == 0:
            log.info("  已处理 %d 条...", i + 1)

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_structured = sum(1 for r in records if r["structured"])
    log.info("===== 阶段三完成 =====")
    log.info("总记录数: %d (其中结构化: %d, 纯文本: %d)", len(records), n_structured, len(records) - n_structured)
    log.info("输出文件: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
