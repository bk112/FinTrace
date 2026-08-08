"""Financial knowledge-base retrieval service (v1.1).

Lazy-loads FAISS index + records on first call; thread-safe after initialisation.

Two interfaces (per v1.1 §3):
  - search_records(query, top_k) → list[dict]   (structured records)
  - retrieve(query)              → str with metadata.records (model-consumable text)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

log = logging.getLogger("fintrace.kb")

# ---------------------------------------------------------------------------
# Configurable paths — override via environment or direct assignment
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
KB_DIR = Path(os.environ.get("FINTRACE_KB_DIR", PROJECT_ROOT / "data" / "kb"))
INDEX_PATH = Path(os.environ.get("FINTRACE_KB_INDEX_PATH", KB_DIR / "fintrace_kb.index"))
RECORDS_PATH = Path(os.environ.get("FINTRACE_KB_RECORDS_PATH", KB_DIR / "records.jsonl"))
MANIFEST_PATH = Path(os.environ.get("FINTRACE_KB_MANIFEST_PATH", KB_DIR / "fintrace_kb.manifest.json"))
MODEL_NAME = os.environ.get("FINTRACE_KB_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
MODEL_CACHE = os.environ.get("FINTRACE_KB_MODEL_CACHE", "/media/xdhpc/data/whr/models")

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_model: SentenceTransformer | None = None
_index: faiss.IndexFlatIP | None = None
_records: list[dict] = []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_loaded() -> None:
    global _model, _index, _records
    if _index is not None:
        return
    with _lock:
        if _index is not None:  # double-check
            return
        log.info("加载知识库: index=%s, records=%s", INDEX_PATH, RECORDS_PATH)

        if not MANIFEST_PATH.exists():
            raise RuntimeError(f"知识库清单不存在: {MANIFEST_PATH}; 请重新运行 Stage 5")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("embedding_model") != MODEL_NAME:
            raise RuntimeError("知识库 embedding 模型与 manifest 不一致")

        _model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE, trust_remote_code=True)

        _index = faiss.read_index(str(INDEX_PATH))

        _records = []
        with open(RECORDS_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    _records.append(json.loads(line))

        assert _index.ntotal == len(_records), \
            f"索引与记录数量不一致: {_index.ntotal} vs {len(_records)}"
        if manifest.get("record_count") != len(_records):
            raise RuntimeError("知识库记录数与 manifest 不一致")
        if manifest.get("embedding_dimension") != _index.d:
            raise RuntimeError("知识库向量维度与 manifest 不一致")
        if manifest.get("records_sha256") != _sha256(RECORDS_PATH):
            raise RuntimeError("records.jsonl 已变化，请重新运行 Stage 5")
        log.info("知识库加载完成: %d 条记录", len(_records))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_records(query: str, top_k: int = 3) -> list[dict]:
    """Search knowledge base and return structured records.

    Args:
        query: Natural-language search query.
        top_k: Number of results to return.

    Returns:
        List of record dicts (v1.1 schema), sorted by relevance descending.

    Edge cases:
        - Empty query → []
        - top_k > total records → returns all records
    """
    q = query.strip()
    if not q:
        return []

    _ensure_loaded()
    assert _model is not None and _index is not None

    k = min(top_k, len(_records))
    if k <= 0:
        return []

    q_emb = _model.encode([q], normalize_embeddings=True)
    scores, indices = _index.search(q_emb.astype("float32"), k)

    results: list[dict] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        rec = dict(_records[idx])  # shallow copy to avoid mutation
        rec["_score"] = float(score)
        results.append(rec)
    return results


def format_result(query: str, records: list[dict]) -> tuple[str, list[dict]]:
    """Format search records into (model-consumable text, clean record list).

    Shared by ``retrieve()`` and :class:`KbRetrievalClient` so both produce
    byte-identical observation text and metadata records (one encode per query).
    """
    if not records:
        return f"[search] 未找到与 \"{query}\" 相关的结果。", []

    lines = [f"搜索结果（共 {len(records)} 条）：\n"]
    for i, rec in enumerate(records, 1):
        source = rec.get("source_doc", "")
        entity = rec.get("entity", "")
        metric = rec.get("metric", "")
        value = rec.get("value_text", "")
        lines.append(f"{i}. [{entity}] {metric}: {value} （来源: {source}）")

    # Embed metadata for reward function
    records_clean = [{k: v for k, v in r.items() if k != "_score"} for r in records]
    metadata_json = json.dumps({"records": records_clean}, ensure_ascii=False)
    lines.append(f"\n<!-- metadata: {metadata_json} -->")

    return "\n".join(lines), records_clean


def retrieve(query: str, top_k: int = 3) -> str:
    """Search and format results as model-consumable text.

    Returns formatted text string with structured records embedded
    as metadata.records in a JSON block that the reward function can parse.

    Args:
        query: Natural-language search query.
        top_k: Number of results to return.

    Returns:
        Markdown-formatted search results with embedded metadata JSON.
    """
    records = search_records(query, top_k)
    text, _ = format_result(query, records)
    return text


def sample_seed_facts(n: int = 1, entity_filter: str | None = None) -> list[dict]:
    """Randomly sample N structured records as seed facts for the difficulty synthesizer.

    Args:
        n: Number of seed facts to sample.
        entity_filter: Optional entity name to restrict sampling to.

    Returns:
        List of record dicts.
    """
    _ensure_loaded()

    pool = _records
    if entity_filter:
        pool = [r for r in _records if r.get("entity") == entity_filter]

    if not pool:
        return []

    n = min(n, len(pool))
    sampled = random.sample(pool, n)
    return [dict(r) for r in sampled]
