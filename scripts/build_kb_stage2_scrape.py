#!/usr/bin/env python3
"""Stage 2: Scrape financial data via AkShare and output structured records (v1.1 schema).

Data sources:
  - Financial abstracts (财报摘要): ak.stock_financial_abstract_ths()
  - Research reports (研报):      ak.stock_research_report_em()
  - Announcements (公告):         ak.stock_notice_report()

Output: data/kb/raw_financial_facts.jsonl
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from datetime import date as dt_date
from pathlib import Path
from typing import Any

import akshare as ak

from fintrace.kb.company_pool import COMPANIES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "raw_financial_facts.jsonl"

RETRIEVED_AT = dt_date.today().isoformat()
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 1.5  # seconds — be gentle to East Money

log = logging.getLogger("kb_scrape")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# ---------------------------------------------------------------------------
# Company pool — imported from fintrace.kb.company_pool (37 companies, 15 industries)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _parse_number(raw: Any) -> tuple[float | None, str, str | None, str]:
    """Parse a raw value string into (value_number, unit, currency, scale).

    Examples:
       "1.47亿" -> (1.47, "亿", "CNY", "亿")
       "46.84%" -> (46.84, "%", None, None)
       "0.43"   -> (0.43, "ratio", None, None)
       False/NaN/None  -> (None, "", None, "")
    """
    if raw is None:
        return None, "", None, ""
    if isinstance(raw, bool):
        return None, "", None, ""
    if isinstance(raw, (int, float)):
        if math.isnan(raw):
            return None, "", None, ""
        return float(raw), "raw", "CNY", ""

    text = str(raw).strip()
    if not text or text.lower() in ("false", "none", "nan", "n/a"):
        return None, "", None, ""

    # Detect unit
    original = text
    currency = "CNY"  # default, A-share

    # Extract percentage
    if text.endswith("%"):
        num = text[:-1].strip()
        try:
            return float(num), "%", None, ""
        except ValueError:
            return None, "%", None, ""

    # Chinese unit suffixes
    unit = ""
    scale = ""
    for suffix, sc in [("万亿元", "万亿"), ("亿元", "亿"), ("万元", "万"), ("亿", "亿"), ("万", "万")]:
        if text.endswith(suffix):
            unit = suffix
            scale = sc
            text = text[: -len(suffix)]
            break

    # Extract leading number
    m = re.match(r"([+-]?\d+\.?\d*)", text)
    if m:
        try:
            return float(m.group(1)), unit or "raw", currency, scale or ""
        except ValueError:
            pass

    return None, original, currency, ""


def _value_type(unit: str) -> str:
    if unit == "%":
        return "percentage"
    if unit in ("亿", "万", "万亿元", "亿元", "万元"):
        return "amount"
    return "text"


def _build_record(
    entity: str,
    metric: str,
    raw_value: Any,
    source_doc: str,
    source_url: str,
    source_type: str,
    document_id: str,
    date_label: str,
    published_at: str,
    industry: str = "未知",
    extra_context: str = "",
) -> dict | None:
    """Build a single record dict per v1.1 schema, or None if value is invalid."""
    value_number, unit, currency, scale = _parse_number(raw_value)
    vtype = _value_type(unit)

    # v1.1: only amount & percentage MUST have value_number;
    # text/ranking facts are still included without it.
    if vtype in ("amount", "percentage") and value_number is None:
        return None  # mandatory numeric facts without a number → drop

    value_text = str(raw_value) if raw_value is not None else ""
    fact_parts = [f"{entity}{date_label}{metric}为{value_text}"]
    if extra_context:
        fact_parts.append(extra_context)
    fact = "，".join(fact_parts) + "。"

    raw_content = f"{entity}|{source_doc}|{metric}|{raw_value}|{date_label}"
    fact_id = f"{source_type[:4]}_{_hash(raw_content)}"

    return {
        "fact_id": fact_id,
        "fact": fact,
        "source_doc": source_doc,
        "source_url": source_url,
        "source_type": source_type,
        "document_id": document_id,
        "entity": entity,
        "industry": industry,
        "metric": metric,
        "value_text": value_text,
        "value_number": value_number,
        "value_type": vtype,
        "unit": unit,
        "currency": currency,
        "scale": scale,
        "date": date_label,
        "published_at": published_at,
        "retrieved_at": RETRIEVED_AT,
        "raw_content_hash": _hash(raw_content),
        "structured": True,
    }


# ---------------------------------------------------------------------------
# Source 1: Financial abstracts (财报摘要 — 同花顺)
# ---------------------------------------------------------------------------

METRIC_MAP: dict[str, str] = {
    "净利润": "净利润",
    "净利润同比增长率": "净利润同比增速",
    "扣非净利润": "扣非净利润",
    "扣非净利润同比增长率": "扣非净利润同比增速",
    "营业总收入": "营业总收入",
    "营业总收入同比增长率": "营业总收入同比增速",
    "基本每股收益": "基本每股收益",
    "每股净资产": "每股净资产",
    "每股经营现金流": "每股经营现金流",
    "销售毛利率": "销售毛利率",
    "销售净利率": "销售净利率",
    "净资产收益率": "净资产收益率",
    "资产负债率": "资产负债率",
    "流动比率": "流动比率",
    "速动比率": "速动比率",
}


def scrape_financial_abstracts(company: dict) -> list[dict]:
    """Scrape financial indicators for one company."""
    records: list[dict] = []
    code = company["code"]
    name = company["name"]

    for attempt in range(MAX_RETRIES):
        try:
            df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
            break
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(SLEEP_BETWEEN_CALLS * (attempt + 1))
            else:
                log.warning("财务摘要 %s(%s) 获取失败 (已重试%d次)", name, code, MAX_RETRIES)
                return records

    if df is None or df.empty:
        log.info("财务摘要 %s(%s) 返回空", name, code)
        return records

    for _, row in df.iterrows():
        period = str(row.get("报告期", ""))
        if not period:
            continue
        # Normalise date → quarterly label
        date_label = _quarter_label(period)
        published_at = period  # report period is the best proxy we have

        for col, metric_name in METRIC_MAP.items():
            raw_val = row.get(col)
            if raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val)):
                continue
            if isinstance(raw_val, bool):
                continue
            rec = _build_record(
                entity=name,
                industry=company["industry"],
                metric=metric_name,
                raw_value=raw_val,
                source_doc=f"{name}{period}财报摘要(同花顺)",
                source_url=f"https://data.10jqka.com.cn/financial/yjgg/opp/{code}/",
                source_type="financial_report",
                document_id=f"ths_abstract_{code}_{period}",
                date_label=date_label,
                published_at=published_at,
            )
            if rec:
                records.append(rec)
    log.info("  财务摘要 %s: %d 条", name, len(records))
    return records


def _quarter_label(date_str: str) -> str:
    """'2025-06-30' → '2025-Q2', '2025-12-31' → '2025-Q4'"""
    try:
        d = dt_date.fromisoformat(date_str[:10])
        q = (d.month - 1) // 3 + 1
        return f"{d.year}-Q{q}"
    except Exception:
        return date_str


# ---------------------------------------------------------------------------
# Source 2: Research reports (研报 — 东方财富)
# ---------------------------------------------------------------------------


def scrape_research_reports(company: dict) -> list[dict]:
    """Scrape research report metadata + forward estimates."""
    records: list[dict] = []
    code = company["code"]
    name = company["name"]

    for attempt in range(MAX_RETRIES):
        try:
            df = ak.stock_research_report_em(symbol=code)
            break
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(SLEEP_BETWEEN_CALLS * (attempt + 1))
            else:
                log.warning("研报 %s(%s) 获取失败", name, code)
                return records

    if df is None or df.empty:
        log.info("研报 %s(%s) 返回空", name, code)
        return records

    for _, row in df.iterrows():
        report_name = str(row.get("报告名称", ""))
        rating = str(row.get("东财评级", ""))
        org = str(row.get("机构", ""))
        pub_date = str(row.get("日期", ""))
        pdf_url = str(row.get("报告PDF链接", ""))
        # Skip rows that look like metadata-only
        if not report_name or report_name == "nan":
            continue

        # --- Fact 1: Rating ---
        if rating and rating != "nan" and rating != "False":
            fact_id = f"rrpt_{_hash(f'{name}_{pub_date}_{rating}')}"
            records.append({
                "fact_id": fact_id,
                "fact": f"{name}在{pub_date}获{org}研报评级为「{rating}」，报告标题：《{report_name}》。",
                "source_doc": f"研报《{report_name}》({org})",
                "source_url": pdf_url if pdf_url and pdf_url != "nan" else "",
                "source_type": "research_report",
                "document_id": f"em_rpt_{code}_{pub_date}_{_hash(report_name)}",
                "entity": name,
                "industry": company["industry"],
                "metric": "东财评级",
                "value_text": rating,
                "value_number": None,
                "value_type": "text",
                "unit": "",
                "currency": None,
                "scale": "",
                "date": pub_date,
                "published_at": pub_date,
                "retrieved_at": RETRIEVED_AT,
                "raw_content_hash": _hash(f"{name}_{pub_date}_{rating}_{org}"),
                "structured": True,
            })

        # --- Fact 2-3: Forward earnings / PE estimates ---
        for year_suffix, yr in [("2026", "2026"), ("2027", "2027"), ("2028", "2028")]:
            earnings_col = f"{yr}-盈利预测-收益"
            pe_col = f"{yr}-盈利预测-市盈率"

            for col, metric_prefix in [(earnings_col, "预测每股收益"), (pe_col, "预测市盈率")]:
                raw_val = row.get(col)
                if raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val)):
                    continue
                rec = _build_record(
                    entity=name,
                    industry=company["industry"],
                    metric=f"{yr}{metric_prefix}",
                    raw_value=raw_val,
                    source_doc=f"研报《{report_name}》({org})",
                    source_url=pdf_url if pdf_url and pdf_url != "nan" else "",
                    source_type="research_report",
                    document_id=f"em_rpt_{code}_{pub_date}_{_hash(report_name)}",
                    date_label=pub_date,
                    published_at=pub_date,
                    extra_context=f"来源：{org}于{pub_date}发布的研报",
                )
                if rec:
                    records.append(rec)

    log.info("  研报 %s: %d 条", name, len(records))
    return records


# ---------------------------------------------------------------------------
# Source 3: Announcements (公告 — 东方财富数据中心)
# ---------------------------------------------------------------------------

# Pull a few recent trading days to get a meaningful announcement corpus
RECENT_DATES = [
    "20250807",
    "20250806",
    "20250805",
    "20250804",
    "20250801",
    "20250731",
    "20250730",
]
ANNOUNCE_TYPES = ["财务报告", "重大事项", "资产重组", "风险提示", "融资公告"]


def scrape_announcements(company_codes: set[str], code_to_company: dict) -> list[dict]:
    """Scrape all A-share announcements for recent dates, filtered to target companies."""
    records: list[dict] = []
    seen = set()

    for date_str in RECENT_DATES:
        for atype in ANNOUNCE_TYPES:
            for attempt in range(MAX_RETRIES):
                try:
                    df = ak.stock_notice_report(symbol=atype, date=date_str)
                    break
                except Exception:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(SLEEP_BETWEEN_CALLS * (attempt + 1))
                    else:
                        log.warning("公告 %s/%s 获取失败", date_str, atype)
                        continue

            if df is None or df.empty:
                continue

            for _, row in df.iterrows():
                stock_code = str(row.get("代码", ""))
                if stock_code not in company_codes:
                    continue
                title = str(row.get("公告标题", ""))
                url = str(row.get("网址", ""))
                pub_date = str(row.get("公告日期", date_str))
                if not title or title == "nan":
                    continue

                key = (stock_code, title)
                if key in seen:
                    continue
                seen.add(key)

                company = code_to_company.get(stock_code, {})
                name = company.get("name", stock_code)

                fact_id = f"anno_{_hash(f'{stock_code}_{title}')}"
                records.append({
                    "fact_id": fact_id,
                    "fact": f"{name}于{pub_date}发布{atype}公告：《{title}》。",
                    "source_doc": f"公告《{title}》({name})",
                    "source_url": url if url and url != "nan" else "",
                    "source_type": "announcement",
                    "document_id": f"em_notice_{stock_code}_{pub_date}_{_hash(title)}",
                    "entity": name,
                    "industry": company.get("industry", "未知"),
                    "metric": "公告",
                    "value_text": title,
                    "value_number": None,
                    "value_type": "text",
                    "unit": "",
                    "currency": None,
                    "scale": "",
                    "date": pub_date,
                    "published_at": pub_date,
                    "retrieved_at": RETRIEVED_AT,
                    "raw_content_hash": _hash(f"{stock_code}_{title}_{pub_date}"),
                    "structured": True,
                })
            time.sleep(0.5)  # gentle between date×type combos

    log.info("  公告: %d 条 (已去重)", len(records))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("===== 阶段二：AkShare 数据抓取 =====")
    log.info("目标公司数: %d, 行业数: %d", len(COMPANIES), len({c['industry'] for c in COMPANIES}))

    all_records: list[dict] = []
    code_to_company = {c["code"]: c for c in COMPANIES}

    for i, company in enumerate(COMPANIES):
        log.info("[%d/%d] %s (%s)", i + 1, len(COMPANIES), company["name"], company["code"])

        # Financial abstracts
        all_records.extend(scrape_financial_abstracts(company))
        time.sleep(SLEEP_BETWEEN_CALLS)

        # Research reports
        all_records.extend(scrape_research_reports(company))
        time.sleep(SLEEP_BETWEEN_CALLS)

    # Announcements (cross-cutting, date-based)
    log.info("开始抓取公告...")
    all_records.extend(
        scrape_announcements(set(code_to_company.keys()), code_to_company)
    )

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Stats
    sources = {}
    industries = set()
    for r in all_records:
        sources[r["source_type"]] = sources.get(r["source_type"], 0) + 1
        industries.add(r.get("entity", ""))
    log.info("===== 抓取完成 =====")
    log.info("总记录数: %d", len(all_records))
    log.info("按来源: %s", json.dumps(sources, ensure_ascii=False))
    log.info("覆盖实体数: %d", len(industries))
    log.info("输出文件: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
