#!/usr/bin/env python3
"""Construct a FinTrace RL candidate set using the documented five-step workflow.

Requires CODEBUDDY_API_KEY in .env. It makes streamed DeepSeek V4 Flash calls
sequentially because the provider's safe concurrency limit is unknown.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from fintrace.data.agentic_synthesis import (
    answer_targets,
    build_entity_groups,
    eligible_record,
    output_record,
    parse_candidate_choice,
    render_candidates,
    unique_verdict_is_yes,
)
from fintrace.rewards.reward_functions import compute_f1
from fintrace.rewards.constants import F1_CORRECT_THRESHOLD

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORDS = PROJECT_ROOT / "data" / "kb" / "records.jsonl"
DEFAULT_ACCEPTED = PROJECT_ROOT / "data" / "interim" / "rl_synthesis_accepted.jsonl"
DEFAULT_REJECTED = PROJECT_ROOT / "data" / "interim" / "rl_synthesis_rejected.jsonl"
API_URL = "https://copilot.tencent.com/v2/chat/completions"
API_MODEL = "deepseek-v4-flash-ioa"

SELECT_PROMPT = """给定根实体“{entity}”及候选事实，选出最适合与当前事实组成两到三跳金融检索问题的一条。
优先不同报告期的同一指标、或同行业不同公司的同一指标。只返回候选序号，不要其他文字。
当前事实：{current_fact}
候选：
{candidates}"""

BRAINSTORM_PROMPT = """已有实体：{entities}。从候选中选择一条能引入新实体、且与已有事实形成金融对比或推理链的事实。
只返回候选序号，不要其他文字。
候选：
{candidates}"""

COMPOSE_PROMPT = """根据以下已验证金融事实，写一个中文金融检索问题。
问题必须需要结合全部事实才能确定最后一条事实的数值，且不得直接泄露公司名、日期、具体数值或答案。
只输出问题本身，不要解释、答案、标签或编号。
事实：
{facts}"""

FUZZ_PROMPT = """将以下金融问题中的具体公司名、日期和数字改写为必须检索才能确定的描述性短语。
不得改变推理逻辑，不得使问题无答案，也不得输出答案。只输出改写后的问题。
原问题：{question}"""

UNIQUE_PROMPT = """判断模糊化问题是否仍能唯一确定原问题的答案，不存在关键实体、时期或指标歧义。
只回答“是”或“否”。
原问题：{original}
模糊化问题：{fuzzed}"""


class RemoteLLMError(RuntimeError):
    """The remote generation API failed or returned no usable streamed content."""


class DeepSeekFlashClient:
    """Minimal OpenAI-compatible streaming client; never logs the API key."""

    def __init__(self, api_key: str, timeout_seconds: int) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def complete(self, prompt: str) -> str:
        payload = {
            "model": API_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json", "X-Api-Key": self._api_key}
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(10, self._timeout_seconds),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RemoteLLMError(f"DeepSeek API request failed: {exc}") from exc

        parts: list[str] = []
        # 腾讯接口未稳定声明 SSE 字符集。手动按 UTF-8 解码，避免中文被误判为 latin-1。
        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace")
            else:
                line = str(raw_line)
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
        content = "".join(parts).strip()
        if not content:
            raise RemoteLLMError("DeepSeek API returned an empty streamed response")
        return content


class BlindBaseModel:
    """Local no-tool base model used only for the documented 0/3 difficulty gate."""

    def __init__(self, model_path: str, max_model_len: int, backend: str) -> None:
        self._backend = backend
        if backend == "vllm":
            # vLLM 0.26 的 EngineCore 多进程在当前环境不稳定，调用方需显式选择。
            from vllm import LLM

            self._llm = LLM(
                model=model_path,
                dtype="bfloat16",
                max_model_len=max_model_len,
                gpu_memory_utilization=0.65,
                enforce_eager=True,
                enable_lora=False,
            )
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=True,
        ).to(self._device).eval()

    def guesses(self, prompt: str, trials: int) -> list[str]:
        if self._backend == "vllm":
            from vllm import SamplingParams

            outputs = self._llm.generate(
                [prompt],
                SamplingParams(n=trials, temperature=1.0, top_p=0.95, max_tokens=256),
                use_tqdm=False,
            )
            return [output.text for output in outputs[0].outputs]

        # 盲猜阶段不注入工具或检索上下文，只保留普通对话模板。
        rendered_prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(rendered_prompt, return_tensors="pt")
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]
        pad_token_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **encoded,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                max_new_tokens=256,
                num_return_sequences=trials,
                pad_token_id=pad_token_id,
            )
        return self._tokenizer.batch_decode(
            output_ids[:, prompt_length:], skip_special_tokens=True
        )


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _append_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _status(message: str) -> None:
    """输出构造阶段状态；flush 保证直接运行 Python 时立即可见。"""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", file=sys.stdout, flush=True)


def _failure_payload(
    attempt: int, root: dict[str, Any], reason: str, detail: str = ""
) -> dict[str, Any]:
    """保留根事实定位，避免失败样本只剩下不可复现的错误文本。"""
    return {
        "attempt": attempt,
        "reason": reason,
        "detail": detail,
        "root_record": {
            "fact_id": root.get("fact_id", ""),
            "entity": root.get("entity", ""),
            "metric": root.get("metric", ""),
            "date": root.get("date", ""),
        },
    }


def _clean_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip("\"'")


def _contains_answer_leak(question: str, targets: list[str]) -> bool:
    normalized = question.replace(" ", "")
    return any(len(target.replace(" ", "")) >= 2 and target.replace(" ", "") in normalized for target in targets)


def _select_record(
    *,
    current: dict[str, Any],
    query: str,
    prompt_template: str,
    llm: DeepSeekFlashClient,
    exclude_ids: set[str],
    known_entities: list[str] | None = None,
) -> dict[str, Any] | None:
    from fintrace.kb import search_records

    candidates = [
        record
        for record in search_records(query, top_k=5)
        if record.get("fact_id") not in exclude_ids and eligible_record(record)
    ]
    if not candidates:
        return None
    prompt = prompt_template.format(
        entity=current["entity"],
        current_fact=current["fact"],
        entities="、".join(known_entities or [str(current["entity"])]),
        candidates=render_candidates(candidates),
    )
    choice = parse_candidate_choice(llm.complete(prompt), len(candidates))
    return candidates[choice] if choice is not None else None


def _fuzz_with_uniqueness_check(
    question: str, llm: DeepSeekFlashClient, retries: int = 2
) -> tuple[str | None, str]:
    last_fuzzed = ""
    last_verdict = ""
    for _ in range(retries):
        fuzzed = _clean_question(llm.complete(FUZZ_PROMPT.format(question=question)))
        verdict = llm.complete(UNIQUE_PROMPT.format(original=question, fuzzed=fuzzed)).strip()
        if unique_verdict_is_yes(verdict):
            return fuzzed, ""
        last_fuzzed = fuzzed
        last_verdict = verdict
    detail = f"last_uniqueness_response={last_verdict[:160]!r}; last_fuzzed={last_fuzzed[:240]!r}"
    return None, detail


def _difficulty_passes(question: str, targets: list[str], base_model: BlindBaseModel, trials: int) -> bool:
    """0/3 gate: accept only if every no-tool base-model guess is wrong by reward F1."""
    for guess in base_model.guesses(question, trials):
        if any(compute_f1(guess, target) >= F1_CORRECT_THRESHOLD for target in targets):
            return False
    return True


@dataclass(frozen=True)
class AttemptFailure:
    reason: str
    detail: str = ""


def _construct_one(root: dict[str, Any], llm: DeepSeekFlashClient, max_loops: int) -> tuple[dict[str, Any] | None, AttemptFailure | None]:
    """Run SELECT -> optional BRAINSTORM -> FUZZ; EXIT is the fixed loop bound."""
    involved = [root]
    selected = _select_record(
        current=root,
        query=f"{root['entity']} 相关 {root['metric']}",
        prompt_template=SELECT_PROMPT,
        llm=llm,
        exclude_ids={str(root["fact_id"])},
    )
    if selected is None:
        return None, AttemptFailure("select_failed")
    involved.append(selected)

    # 第三跳只在配置允许时尝试；没有合适新实体时保留已验证的两跳链。
    if max_loops >= 3:
        brainstorm = _select_record(
            current=selected,
            query=f"{selected['entity']} 同行业 对比 {selected['metric']}",
            prompt_template=BRAINSTORM_PROMPT,
            llm=llm,
            exclude_ids={str(record["fact_id"]) for record in involved},
            known_entities=[str(record["entity"]) for record in involved],
        )
        if brainstorm is not None and brainstorm.get("entity") not in {record["entity"] for record in involved}:
            involved.append(brainstorm)

    facts = "\n".join(
        f"- [{record['fact_id']}] {record['fact']}" for record in involved
    )
    draft = _clean_question(llm.complete(COMPOSE_PROMPT.format(facts=facts)))
    fuzzed, fuzz_failure_detail = _fuzz_with_uniqueness_check(draft, llm)
    if fuzzed is None:
        return None, AttemptFailure("fuzz_not_unique", fuzz_failure_detail)
    if _contains_answer_leak(fuzzed, answer_targets(involved[-1])):
        return None, AttemptFailure("answer_leak")
    return output_record(
        qid="",
        prompt=fuzzed,
        root=root,
        involved=involved,
        actions=(
            ["SELECT", "BRAINSTORM", "FUZZ", "EXIT"]
            if len(involved) == 3
            else ["SELECT", "FUZZ", "EXIT"]
        ),
    ), None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--candidates", type=int, default=100, help="number of candidate attempts before 0/3 filtering")
    parser.add_argument("--max-loops", type=int, default=3, choices=(2, 3))
    parser.add_argument("--difficulty-trials", type=int, default=3, choices=(3,))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model", default="/media/xdhpc/data/whr/models/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--difficulty-backend",
        choices=("transformers", "vllm"),
        default="transformers",
        help="local no-tool model backend; transformers is the stable default",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--accepted-output", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--rejected-output", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("CODEBUDDY_API_KEY", "").strip()
    if not api_key:
        print("CODEBUDDY_API_KEY is missing. Fill it in .env before running this script.", file=sys.stderr)
        return 2
    if args.candidates < 1:
        print("--candidates must be positive", file=sys.stderr)
        return 2
    if not args.records.is_file():
        print(f"records file does not exist: {args.records}", file=sys.stderr)
        return 2
    for output in (args.accepted_output, args.rejected_output):
        if output.exists() and not args.overwrite:
            print(f"refusing to overwrite existing output: {output}; pass --overwrite", file=sys.stderr)
            return 2
        output.parent.mkdir(parents=True, exist_ok=True)
    if args.accepted_output.resolve() == args.rejected_output.resolve():
        print("accepted and rejected outputs must be different files", file=sys.stderr)
        return 2

    _status(f"加载知识库记录: {args.records}")
    groups = build_entity_groups(_load_records(args.records))
    if not groups:
        print("no eligible KB records", file=sys.stderr)
        return 1
    eligible_count = sum(len(records) for records in groups.values())
    _status(
        f"根事实池就绪: {eligible_count} 条结构化数值事实，"
        f"覆盖 {len(groups)} 个实体；计划构造 {args.candidates} 条候选"
    )
    rng = random.Random(args.seed)
    entities = list(groups)
    rng.shuffle(entities)
    llm = DeepSeekFlashClient(api_key, args.timeout_seconds)
    base_model: BlindBaseModel | None = None
    accepted = rejected = 0
    _status(f"远程构造模型: {API_MODEL}；API 严格单并发")
    if not sys.stdout.isatty():
        _status("当前标准输出不是终端；请使用 conda run --no-capture-output 以实时查看进度")

    # 供应商并发上限未知，故严格单并发；tqdm 会实时显示构造和筛选进度。
    with args.accepted_output.open("w", encoding="utf-8") as accepted_fh, args.rejected_output.open(
        "w", encoding="utf-8"
    ) as rejected_fh, tqdm(
        total=args.candidates,
        desc="构造候选",
        unit="题",
        file=sys.stdout,
        mininterval=0.3,
        dynamic_ncols=True,
    ) as progress:
        for attempt in range(args.candidates):
            entity = entities[attempt % len(entities)]
            root = rng.choice(groups[entity])
            try:
                candidate, failure = _construct_one(root, llm, args.max_loops)
            except RemoteLLMError as exc:
                _append_jsonl(
                    rejected_fh,
                    _failure_payload(attempt, root, "remote_llm_error", str(exc)),
                )
                rejected += 1
                progress.update(1)
                continue
            except Exception as exc:
                _append_jsonl(
                    rejected_fh,
                    _failure_payload(attempt, root, "construction_error", str(exc)),
                )
                rejected += 1
                progress.update(1)
                continue

            if failure is not None:
                _append_jsonl(
                    rejected_fh,
                    _failure_payload(attempt, root, failure.reason, failure.detail),
                )
                rejected += 1
            else:
                assert candidate is not None
                if base_model is None:
                    progress.clear()
                    _status(
                        f"首条候选通过构造，加载本地难度模型 "
                        f"({args.difficulty_backend}): {args.base_model}"
                    )
                    base_model = BlindBaseModel(
                        args.base_model, args.max_model_len, args.difficulty_backend
                    )
                    _status("本地难度模型加载完成，开始执行 0/3 无工具盲猜筛选")
                targets = candidate["ground_truth"]["target"]
                if _difficulty_passes(candidate["prompt"], targets, base_model, args.difficulty_trials):
                    candidate["qid"] = f"ft_{accepted + 1:05d}"
                    _append_jsonl(accepted_fh, candidate)
                    accepted += 1
                else:
                    _append_jsonl(
                        rejected_fh,
                        _failure_payload(
                            attempt,
                            root,
                            "difficulty_failed_base_model_guessed",
                        ) | {"meta": candidate["meta"]},
                    )
                    rejected += 1
            progress.update(1)
            progress.set_postfix(accepted=accepted, rejected=rejected)

    print(
        json.dumps(
            {
                "attempted": args.candidates,
                "accepted": accepted,
                "rejected": rejected,
                "accepted_output": str(args.accepted_output),
                "rejected_output": str(args.rejected_output),
                "api_concurrency": 1,
                "difficulty_trials": args.difficulty_trials,
                "difficulty_backend": args.difficulty_backend,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
