#!/usr/bin/env python3
"""Stage 3 (revised): Process local FinanceComplexQA Chinese subset → v1.1 schema records.

Input:  data/FinanceComplexQA/CN/{scene_categories,task_categories}/**/*.jsonl
Output:
  data/kb/opensource_supplement_facts.jsonl       # 可检索事实，绝不含标准答案
  data/interim/financecomplexqa_gold_labels.jsonl # 训练标签，不进入向量库
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date as dt_date
from pathlib import Path

from fintrace.kb.company_pool import infer_industry

RETRIEVED_AT = dt_date.today().isoformat()
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "FinanceComplexQA" / "CN"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "opensource_supplement_facts.jsonl"
LABELS_PATH = Path(__file__).resolve().parent.parent / "data" / "interim" / "financecomplexqa_gold_labels.jsonl"
LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("kb_fcqa_local")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# ---------------------------------------------------------------------------
# Entity pattern: match company names, including the specific ones from our KB
# ---------------------------------------------------------------------------
ENTITY_RE = re.compile(
    r"("
    r"贵州茅台|五粮液|泸州老窖|招商银行|工商银行|平安银行|"
    r"恒瑞医药|迈瑞医疗|云南白药|海康威视|汇川技术|紫光国微|"
    r"宁德时代|隆基绿能|TCL中环|比亚迪|长城汽车|长安汽车|"
    r"中国平安|中国人寿|万科|保利发展|美的集团|格力电器|海尔智家|"
    r"中芯国际|韦尔股份|伊利股份|牧原股份|万华化学|龙佰集团|"
    r"中兴通讯|传音控股|中航沈飞|航天电器|长江电力|中国广核|"
    r"(?:[一-鿿]{2,6}(?:集团|股份|控股|科技|银行|保险|汽车|能源|医药|电力|通信|公司))"
    r")"
)

# Metric patterns for structured extraction
METRIC_PATTERNS = [
    (re.compile(r"(?:流动比率|速动比率|资产负债率|产权比率|保守速动比率)"), "ratio"),
    (re.compile(r"(?:净资产收益率|ROE|销售净利率|销售毛利率|营业利润率|EBITDA\s*利润率|毛利率|净利率)"), "percentage"),
    (re.compile(r"(?:每股收益|每股净资产|每股经营现金流|每股资本公积金|每股未分配利润)"), "amount"),
    (re.compile(r"(?:营业(?:总)?收入|营收|净利润|扣非净利润|总资产|流动资产|流动负债|存货)"), "amount"),
    (re.compile(r"(?:同比(?:增长|增速|下滑|下降|上升)|增速|增长率)"), "percentage"),
    (re.compile(r"(?:市盈率|PE|市净率|PB|市销率)"), "ratio"),
    (re.compile(r"(?:市(?:场)?占(?:有)?率|份额|排名)"), "percentage"),
]
NUM_RE = re.compile(r"([+-]?\d+\.?\d*)")
PERCENT_RE = re.compile(r"([+-]?\d+\.?\d*)\s*%")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _find_entity(text: str) -> str | None:
    m = ENTITY_RE.search(text)
    return m.group(1) if m else None


def _find_metric(text: str) -> tuple[str | None, str | None, str]:
    """Return (metric_name, value_text, value_type) or (None, None, "text")."""
    for pat, vtype in METRIC_PATTERNS:
        m = pat.search(text)
        if m:
            raw = m.group(0)
            # Try to extract a numeric value nearby
            context = text[m.start():m.start() + 200]
            # Percentage
            pct_m = PERCENT_RE.search(context)
            if pct_m:
                return raw, pct_m.group(0), vtype
            # Plain number
            num_m = NUM_RE.search(context.replace(raw, "", 1))
            if num_m:
                return raw, num_m.group(1), vtype
            return raw, raw, vtype
    return None, None, "text"


def _parse_value(value_text: str | None) -> tuple[float | None, str, str | None, str]:
    """Parse human value text into structured value_number/unit/currency/scale."""
    if not value_text:
        return None, "", None, ""
    text = value_text.strip()

    # Percentage
    if text.endswith("%"):
        try:
            return float(text[:-1]), "%", None, ""
        except ValueError:
            return None, "%", None, ""

    # Chinese units
    for suffix, sc in [("万亿元", "万亿"), ("亿元", "亿"), ("万元", "万"), ("亿", "亿"), ("万", "万")]:
        if text.endswith(suffix):
            num_part = text[: -len(suffix)]
            try:
                return float(re.sub(r"[^0-9.\-]", "", num_part)), suffix, "CNY", sc
            except ValueError:
                return None, suffix, "CNY", sc

    # Plain number
    try:
        return float(text.replace(",", "")), "raw", "CNY", ""
    except ValueError:
        return None, text, "CNY", ""


def main() -> None:
    log.info("===== 阶段三(修订): 本地 FinanceComplexQA 处理 =====")

    # Collect all JSONL files
    all_files = list(DATA_ROOT.rglob("*.jsonl"))
    log.info("找到 %d 个 JSONL 文件", len(all_files))

    records: list[dict] = []
    labels_by_qid: dict[str, dict] = {}
    total_lines = 0
    skipped_parse = 0

    for filepath in all_files:
        with open(filepath, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped_parse += 1
                    continue

                company = obj.get("company_with_period", "")
                finqa_id = obj.get("finqa_id", "")
                question = obj.get("question", "")
                external_knowledge = obj.get("external_knowledge", "")
                gold = obj.get("gold", None)
                doc_type = obj.get("doc_type", "")
                task_type = obj.get("task_type", "")
                # --- Determine entity ---
                entity = _find_entity(company) or company or "未知"
                industry = infer_industry(entity, source_type="opensource_dataset", doc_type=doc_type)

                # -----------------------------------------------------------------
                # Fact 1: Question-derived fact (the implicit context in the question)
                # Many questions embed factual premises like "假如爱美客在存货为X万元, 2022年的流动比率"
                # -----------------------------------------------------------------
                q_clean = question.split("\n\n###")[0]  # strip output format instructions
                if q_clean and len(q_clean) > 20:
                    fact_id = f"fcqa_q_{_hash(finqa_id + '_q')}"
                    metric, val_text, vtype = _find_metric(q_clean)
                    val_num, unit, currency, scale = _parse_value(val_text)

                    rec = {
                        "fact_id": fact_id,
                        "fact": f"FinanceComplexQA金融问答上下文: {q_clean[:400]}",
                        "source_doc": f"FinanceComplexQA/{doc_type} (id: {finqa_id})",
                        "source_url": "",
                        "source_type": "opensource_dataset",
                        "document_id": f"fcqa_{finqa_id}",
                        "entity": entity,
                        "industry": industry,
                        "metric": metric or "金融上下文",
                        "value_text": val_text or q_clean[:150],
                        "value_number": val_num,
                        "value_type": vtype if metric else "text",
                        "unit": unit,
                        "currency": currency,
                        "scale": scale,
                        "date": "",
                        "published_at": "",
                        "retrieved_at": RETRIEVED_AT,
                        "raw_content_hash": _hash(finqa_id + q_clean[:200]),
                        "structured": metric is not None,
                    }
                    records.append(rec)

                # -----------------------------------------------------------------
                # Fact 2: External knowledge (financial formulas & domain knowledge)
                # -----------------------------------------------------------------
                if external_knowledge and len(external_knowledge) > 20:
                    fact_id = f"fcqa_k_{_hash(finqa_id + '_k')}"
                    # Clean LaTeX
                    ek_clean = external_knowledge.replace("\\text{", "").replace("}", "").replace("\\", "")

                    rec = {
                        "fact_id": fact_id,
                        "fact": f"金融公式/领域知识: {ek_clean[:400]}",
                        "source_doc": f"FinanceComplexQA/{doc_type} (id: {finqa_id})",
                        "source_url": "",
                        "source_type": "opensource_dataset",
                        "document_id": f"fcqa_{finqa_id}",
                        "entity": entity,
                        "industry": industry,
                        "metric": "金融公式/知识点",
                        "value_text": ek_clean[:200],
                        "value_number": None,
                        "value_type": "text",
                        "unit": "",
                        "currency": None,
                        "scale": "",
                        "date": "",
                        "published_at": "",
                        "retrieved_at": RETRIEVED_AT,
                        "raw_content_hash": _hash(finqa_id + ek_clean[:200]),
                        "structured": False,
                    }
                    records.append(rec)

                # -----------------------------------------------------------------
                # Fact 3: Ground truth is training supervision, never retrievable evidence.
                # 把答案写入 KB 会让同源题目的检索和 reward 发生标签泄漏。
                # -----------------------------------------------------------------
                if gold is not None and q_clean:
                    targets = gold if isinstance(gold, list) else [gold]
                    targets = [str(target).strip() for target in targets if str(target).strip()]
                    if targets:
                        qid = f"fcqa_{finqa_id}"
                        labels_by_qid[qid] = {
                            "qid": qid,
                            "prompt": q_clean,
                            "ground_truth": {"target": targets},
                            "source": {
                                "source_type": "FinanceComplexQA",
                                "document_id": finqa_id,
                                "doc_type": doc_type,
                                "task_type": task_type,
                            },
                        }

        pass  # progress logged at end

    # Write
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 标签文件位于 data/interim，默认不纳入版本控制，也不会被 Stage 4 合并。
    with open(LABELS_PATH, "w", encoding="utf-8") as fh:
        for label in labels_by_qid.values():
            fh.write(json.dumps(label, ensure_ascii=False) + "\n")

    n_structured = sum(1 for r in records if r["structured"])
    log.info("===== 阶段三(修订) 完成 =====")
    log.info("处理文件: %d, 总行数: %d, 跳过: %d", len(all_files), total_lines, skipped_parse)
    log.info("产出记录: %d (结构化: %d, 纯文本: %d)", len(records), n_structured, len(records) - n_structured)
    log.info("输出: %s", OUTPUT_PATH)
    log.info("训练标签: %s (%d 条，不进入知识库)", LABELS_PATH, len(labels_by_qid))


if __name__ == "__main__":
    main()
