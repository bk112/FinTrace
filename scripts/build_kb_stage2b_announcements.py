#!/usr/bin/env python3
"""Stage 2 supplement: Improved announcement scraper.

Problem: stock_notice_report(date, type) returns ALL A-share announcements;
         filtering to 37 target companies across 7 dates × 5 types hit only 28 records.

Fix:  60 trading days (last quarter), type="全部", with broader company identifiers.
      Write directly alongside the existing raw_financial_facts.jsonl for Stage 4 merge.

Output: data/kb/raw_announcements_supplement.jsonl
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import akshare as ak

from fintrace.kb.company_pool import infer_industry

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "raw_announcements_supplement.jsonl"

RETRIEVED_AT = date.today().isoformat()
MAX_RETRIES = 3
SLEEP = 1.0

log = logging.getLogger("kb_anno_v2")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# Target companies (same as stage 2) with stock codes + alternate name variations
TARGET_CODES: set[str] = {
    "600519", "000858", "000568",  # 白酒
    "600036", "601398", "000001",  # 银行
    "600276", "300760", "000538",  # 医药
    "002415", "300124", "002049",  # 科技
    "300750", "601012", "002129",  # 新能源
    "002594", "601633", "000625",  # 汽车
    "601318", "601628",            # 保险
    "000002", "600048",            # 地产
    "000333", "000651", "600690",  # 家电
    "688981", "603501",            # 芯片
    "600887", "002714",            # 食品
    "600309", "002601",            # 化工
    "000063", "688036",            # 通信
    "600760", "002025",            # 军工
    "600900", "003816",            # 电力
}

# Name → code mapping for title-based fallback matching
COMPANY_NAMES: dict[str, str] = {
    "贵州茅台": "600519", "五粮液": "000858", "泸州老窖": "000568",
    "招商银行": "600036", "工商银行": "601398", "平安银行": "000001",
    "恒瑞医药": "600276", "迈瑞医疗": "300760", "云南白药": "000538",
    "海康威视": "002415", "汇川技术": "300124", "紫光国微": "002049",
    "宁德时代": "300750", "隆基绿能": "601012", "TCL中环": "002129",
    "比亚迪": "002594", "长城汽车": "601633", "长安汽车": "000625",
    "中国平安": "601318", "中国人寿": "601628",
    "万科": "000002", "保利发展": "600048",
    "美的集团": "000333", "格力电器": "000651", "海尔智家": "600690",
    "中芯国际": "688981", "韦尔股份": "603501",
    "伊利股份": "600887", "牧原股份": "002714",
    "万华化学": "600309", "龙佰集团": "002601",
    "中兴通讯": "000063", "传音控股": "688036",
    "中航沈飞": "600760", "航天电器": "002025",
    "长江电力": "600900", "中国广核": "003816",
}
CODE_TO_NAME: dict[str, str] = {code: name for name, code in COMPANY_NAMES.items()}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _match_company_from_title(title: str) -> str | None:
    for name, code in COMPANY_NAMES.items():
        if name in title:
            return code
    return None


def trading_dates(n: int = 60) -> list[str]:
    """Generate last N calendar days (not strictly trading days, but API handles weekends)."""
    today = date.today()
    return [(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(1, n + 1)]


def main():
    log.info("===== 增强公告抓取（60天, 全部类型）=====")
    records: list[dict] = []
    seen = set()
    dates = trading_dates(60)
    log.info("日期范围: %s ~ %s (%d天)", dates[-1], dates[0], len(dates))

    for idx, dt in enumerate(dates):
        for attempt in range(MAX_RETRIES):
            try:
                df = ak.stock_notice_report(symbol="全部", date=dt)
                break
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(SLEEP * (attempt + 1))
                else:
                    log.debug("公告 %s 获取失败", dt)
                    df = None

        if df is None or df.empty:
            continue

        for _, row in df.iterrows():
            code = str(row.get("代码", ""))
            title = str(row.get("公告标题", ""))
            url = str(row.get("网址", ""))
            pub_date = str(row.get("公告日期", dt))

            if not title or title == "nan":
                continue

            # Match by stock code first, then by company name in title
            matched_code = code if code in TARGET_CODES else _match_company_from_title(title)
            if matched_code is None:
                continue

            key = (matched_code, title)
            if key in seen:
                continue
            seen.add(key)

            name = CODE_TO_NAME.get(matched_code, matched_code)
            fact_id = f"anno2_{_hash(f'{matched_code}_{title}')}"
            records.append({
                "fact_id": fact_id,
                "fact": f"{name}于{pub_date}发布公告：《{title}》。",
                "source_doc": f"公告《{title}》({name})",
                "source_url": url if url and url != "nan" else "",
                "source_type": "announcement",
                "document_id": f"em_notice_{matched_code}_{pub_date}_{_hash(title)}",
                "entity": name,
                "industry": infer_industry(name, source_type="announcement"),
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
                "raw_content_hash": _hash(f"{matched_code}_{title}_{pub_date}"),
                "structured": True,
            })

        if (idx + 1) % 10 == 0:
            log.info("  已处理 %d/%d 天, 累计公告: %d 条", idx + 1, len(dates), len(records))
            time.sleep(SLEEP)  # breathe

    # Write
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("===== 增强公告完成: %d 条 =====", len(records))
    log.info("输出: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
