#!/usr/bin/env python3
"""Announcement scraper — 30-day sweep with '全部' type, filter to target 37 companies."""
from __future__ import annotations
import akshare as ak, json, hashlib, time
from datetime import date, timedelta
from pathlib import Path

from fintrace.kb.company_pool import infer_industry

TARGET = {'600519','000858','000568','600036','601398','000001','600276','300760','000538',
          '002415','300124','002049','300750','601012','002129','002594','601633','000625',
          '601318','601628','000002','600048','000333','000651','600690','688981','603501',
          '600887','002714','600309','002601','000063','688036','600760','002025','600900','003816'}
NAMES = {'600519':'贵州茅台','000858':'五粮液','000568':'泸州老窖','600036':'招商银行',
         '601398':'工商银行','000001':'平安银行','600276':'恒瑞医药','300760':'迈瑞医疗',
         '000538':'云南白药','002415':'海康威视','300124':'汇川技术','002049':'紫光国微',
         '300750':'宁德时代','601012':'隆基绿能','002129':'TCL中环','002594':'比亚迪',
         '601633':'长城汽车','000625':'长安汽车','601318':'中国平安','601628':'中国人寿',
         '000002':'万科A','600048':'保利发展','000333':'美的集团','000651':'格力电器',
         '600690':'海尔智家','688981':'中芯国际','603501':'韦尔股份','600887':'伊利股份',
         '002714':'牧原股份','600309':'万华化学','002601':'龙佰集团','000063':'中兴通讯',
         '688036':'传音控股','600760':'中航沈飞','002025':'航天电器','600900':'长江电力','003816':'中国广核'}
def h(s): return hashlib.sha256(s.encode()).hexdigest()[:16]

out = Path(__file__).resolve().parent.parent / "data" / "kb" / "raw_announcements_supplement.jsonl"
seen = set(); records = []
dates = [(date.today()-timedelta(days=i)).strftime('%Y%m%d') for i in range(1,31)]
retrieved = date.today().isoformat()

for di, d in enumerate(dates):
    try: df = ak.stock_notice_report(symbol='全部', date=d)
    except: continue
    if df is None or df.empty: continue
    for _, r in df.iterrows():
        code = str(r.get('代码','')); title = str(r.get('公告标题',''))
        url = str(r.get('网址','')); pub = str(r.get('公告日期', d))
        if not title or title == 'nan': continue
        if code not in TARGET:
            matched = None
            for k, v in NAMES.items():
                if v in title: matched = k; break
            if not matched: continue
            code = matched
        key = (code, title)
        if key in seen: continue
        seen.add(key)
        name = NAMES.get(code, code)
        records.append(dict(
            fact_id=f'anno2_{h(f"{code}_{title}")}',
            fact=f'{name}于{pub}发布公告：《{title}》。',
            source_doc=f'公告《{title}》({name})', source_url=url if url != 'nan' else '',
            source_type='announcement', document_id=f'em_notice_{code}_{d}_{h(title)}',
            entity=name, industry=infer_industry(name, source_type='announcement'), metric='公告',
            value_text=title, value_number=None,
            value_type='text', unit='', currency=None, scale='', date=pub,
            published_at=pub, retrieved_at=retrieved,
            raw_content_hash=h(f'{code}_{title}_{d}'), structured=True))
    if (di+1) % 10 == 0:
        print(f'  {di+1}/30 天, {len(records)} 条', flush=True)
    time.sleep(0.3)

with open(out, 'w', encoding='utf-8') as f:
    for rec in records: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
print(f'完成: {len(records)} 条公告, 输出: {out}')
