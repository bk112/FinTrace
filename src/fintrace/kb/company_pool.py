"""Company pool: name → industry mapping shared by KB builders and the difficulty synthesizer.

The 37 scraped companies are mapped explicitly; FinanceComplexQA entities fall back
to doc_type inference or "未知".
"""

from __future__ import annotations

# (code, name, industry) — same pool used by the Stage-2 scraper
COMPANIES: list[dict[str, str]] = [
    # 白酒
    {"code": "600519", "name": "贵州茅台", "industry": "白酒"},
    {"code": "000858", "name": "五粮液", "industry": "白酒"},
    {"code": "000568", "name": "泸州老窖", "industry": "白酒"},
    # 银行
    {"code": "600036", "name": "招商银行", "industry": "银行"},
    {"code": "601398", "name": "工商银行", "industry": "银行"},
    {"code": "000001", "name": "平安银行", "industry": "银行"},
    # 医药
    {"code": "600276", "name": "恒瑞医药", "industry": "医药"},
    {"code": "300760", "name": "迈瑞医疗", "industry": "医药"},
    {"code": "000538", "name": "云南白药", "industry": "医药"},
    # 科技
    {"code": "002415", "name": "海康威视", "industry": "科技"},
    {"code": "300124", "name": "汇川技术", "industry": "科技"},
    {"code": "002049", "name": "紫光国微", "industry": "科技"},
    # 新能源
    {"code": "300750", "name": "宁德时代", "industry": "新能源"},
    {"code": "601012", "name": "隆基绿能", "industry": "新能源"},
    {"code": "002129", "name": "TCL中环", "industry": "新能源"},
    # 汽车
    {"code": "002594", "name": "比亚迪", "industry": "汽车"},
    {"code": "601633", "name": "长城汽车", "industry": "汽车"},
    {"code": "000625", "name": "长安汽车", "industry": "汽车"},
    # 保险
    {"code": "601318", "name": "中国平安", "industry": "保险"},
    {"code": "601628", "name": "中国人寿", "industry": "保险"},
    # 地产
    {"code": "000002", "name": "万科A", "industry": "地产"},
    {"code": "600048", "name": "保利发展", "industry": "地产"},
    # 家电
    {"code": "000333", "name": "美的集团", "industry": "家电"},
    {"code": "000651", "name": "格力电器", "industry": "家电"},
    {"code": "600690", "name": "海尔智家", "industry": "家电"},
    # 芯片
    {"code": "688981", "name": "中芯国际", "industry": "芯片"},
    {"code": "603501", "name": "韦尔股份", "industry": "芯片"},
    # 食品
    {"code": "600887", "name": "伊利股份", "industry": "食品"},
    {"code": "002714", "name": "牧原股份", "industry": "食品"},
    # 化工
    {"code": "600309", "name": "万华化学", "industry": "化工"},
    {"code": "002601", "name": "龙佰集团", "industry": "化工"},
    # 通信
    {"code": "000063", "name": "中兴通讯", "industry": "通信"},
    {"code": "688036", "name": "传音控股", "industry": "通信"},
    # 军工
    {"code": "600760", "name": "中航沈飞", "industry": "军工"},
    {"code": "002025", "name": "航天电器", "industry": "军工"},
    # 电力
    {"code": "600900", "name": "长江电力", "industry": "电力"},
    {"code": "003816", "name": "中国广核", "industry": "电力"},
]

CODE_TO_COMPANY: dict[str, dict[str, str]] = {c["code"]: c for c in COMPANIES}

# name → industry for direct lookup (includes variants like 万科 without suffix)
NAME_TO_INDUSTRY: dict[str, str] = {c["name"]: c["industry"] for c in COMPANIES}
NAME_TO_INDUSTRY["万科"] = "地产"

# FinanceComplexQA doc_type → industry signal (best-effort)
DOC_TYPE_TO_INDUSTRY: dict[str, str] = {
    "Bank_Financial_Statements": "银行",
    # Corporate_Financial_Report / Research_Report / etc. carry no reliable industry signal
}

UNKNOWN_INDUSTRY = "未知"


def infer_industry(
    entity: str,
    *,
    source_type: str = "",
    doc_type: str = "",
) -> str:
    """Best-effort industry inference for an entity name.

    Priority: exact name match → 万科 prefix variant → FCQA doc_type → 未知.
    """
    if not entity or entity == "未知":
        return UNKNOWN_INDUSTRY
    if entity in NAME_TO_INDUSTRY:
        return NAME_TO_INDUSTRY[entity]
    # 万科A vs 万科 variants
    if "万科" in entity:
        return "地产"
    if doc_type and doc_type in DOC_TYPE_TO_INDUSTRY:
        return DOC_TYPE_TO_INDUSTRY[doc_type]
    return UNKNOWN_INDUSTRY
