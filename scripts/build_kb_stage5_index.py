#!/usr/bin/env python3
"""Stage 5: Build FAISS index from financial_knowledge_base.jsonl using Qwen3-Embedding-0.6B.

Output:
  data/kb/fintrace_kb.index   (FAISS IndexFlatIP)
  data/kb/records.jsonl       (records aligned with index order)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KB_PATH = OUTPUT_DIR / "financial_knowledge_base.jsonl"
INDEX_PATH = OUTPUT_DIR / "fintrace_kb.index"
RECORDS_PATH = OUTPUT_DIR / "records.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "fintrace_kb.manifest.json"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_CACHE = "/media/xdhpc/data/whr/models"
BATCH_SIZE = 128

log = logging.getLogger("kb_index")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    log.info("===== 阶段五：FAISS 索引构建 =====")

    # Load records
    if not KB_PATH.exists():
        log.error("知识库文件不存在: %s", KB_PATH)
        raise SystemExit(1)

    records: list[dict] = []
    with open(KB_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("加载记录: %d 条", len(records))

    texts = [r["fact"] for r in records]

    # Load model
    t0 = time.time()
    log.info("加载模型: %s (cache: %s)", MODEL_NAME, MODEL_CACHE)
    model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE, trust_remote_code=True)
    dim = model.get_embedding_dimension()
    log.info("模型已加载, embedding 维度: %d", dim)

    # Encode
    log.info("开始编码 %d 条文本...", len(texts))
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )
    t_encode = time.time() - t0
    log.info("编码完成, 耗时: %.1f 秒 (%.1f 条/秒)", t_encode, len(texts) / t_encode)

    # Build index
    log.info("构建 FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    log.info("索引构建完成, 向量数量: %d", index.ntotal)

    # Save
    faiss.write_index(index, str(INDEX_PATH))
    log.info("索引文件: %s", INDEX_PATH)

    # Save records (ingestion-sorted)
    with open(RECORDS_PATH, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
    log.info("记录文件: %s (%d 条)", RECORDS_PATH, len(records))

    # 记录索引、模型和 records snapshot 的绑定关系，防止只更新其中一个文件。
    manifest = {
        "schema_version": "fintrace-kb-v1.1",
        "embedding_model": MODEL_NAME,
        "embedding_dimension": dim,
        "normalize_embeddings": True,
        "record_count": len(records),
        "records_sha256": _sha256(RECORDS_PATH),
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    log.info("索引清单: %s", MANIFEST_PATH)

    # Verify
    loaded = faiss.read_index(str(INDEX_PATH))
    assert loaded.ntotal == len(records), f"索引记录数不匹配: {loaded.ntotal} vs {len(records)}"
    log.info("索引验证通过 (faiss.read_index 成功, 向量数量: %d)", loaded.ntotal)

    # Quick sanity
    log.info("快速检索测试...")
    q = model.encode(["银行营收增速"], normalize_embeddings=True)
    scores, indices = loaded.search(q.astype("float32"), 3)
    log.info("  查询 '银行营收增速' → Top 3:")
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0:
            r = records[idx]
            log.info("    [%.4f] %s", score, r["fact"][:80])

    log.info("===== 阶段五完成 =====")


if __name__ == "__main__":
    main()
