#!/usr/bin/env python3
"""Construct a FinTrace RL candidate set using the documented five-step workflow.

Requires CODEBUDDY_API_KEY in .env. It makes streamed DeepSeek V4 Flash calls
with a small, caller-controlled concurrency level.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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
    build_anchored_lookup_question,
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

SELECT_PROMPT = """给定根实体“{entity}”及候选事实，选择一条与当前事实属于同一公司、同一指标、不同报告期且数值不同的候选。
该选择会被代码模板化为检索问题，因此不能引入另一家公司、比较关系或任何额外事实。只返回候选序号，不要其他文字。
当前事实：{current_fact}
候选：
{candidates}"""

BRAINSTORM_PROMPT = """已有实体：{entities}。从候选中选择一条能引入新实体、且与已有事实形成金融对比或推理链的事实。
只返回候选序号，不要其他文字。
候选：
{candidates}"""

COMPOSE_PROMPT = """根据以下已验证金融事实，写一个中文金融检索问题。
答案必须严格等于最后一条事实的原始数值；前面的事实只能用于定位该最后事实，不能参与数值计算。
禁止询问差值、增减幅度、百分点、比例、倍数、最大值、最小值或任何需要计算两个事实的结果。
问题必须需要结合全部事实才能定位最后一条事实，且不得直接泄露公司名、日期、具体数值或答案。
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

TARGET_ALIGNMENT_PROMPT = """根据事实和问题判断 target 是否严格正确。
只有在问题要求直接查出“目标事实”的原始数值，且其他事实仅用于定位该事实时回答“是”。
若问题要求比较、相减、相除、求百分点、比例、倍数、最大最小值，或询问实体/时期而 target 是数值，则回答“否”。
只回答“是”或“否”。
目标事实：{target_fact}
全部事实：
{facts}
问题：{question}"""

RELATION_GROUNDING_PROMPT = """核验问题中的每一个事实关系是否都能由给定结构化事实严格支持。
禁止把同一实体说成“另一家/另一公司”，禁止把不同数值说成“相同/一致/相等”，禁止虚构报告期、指标、比较关系或因果关系。
问题只能将事实改写为描述性定位线索；若任一限定条件与事实冲突、事实不足以支持，或无法确认，回答“否”。
只有全部关系均真实且 target 仍直接询问最后一条事实的原始数值时，才回答“是”。
只回答“是”或“否”。
目标事实：{target_fact}
全部事实：
{facts}
问题：{question}"""


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
    """Local no-tool base model used only for the documented difficulty gate."""

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
    require_same_entity_metric: bool = False,
) -> dict[str, Any] | None:
    from fintrace.kb import search_records

    candidates = [
        record
        for record in search_records(query, top_k=5)
        if record.get("fact_id") not in exclude_ids and eligible_record(record)
    ]
    if require_same_entity_metric:
        candidates = [
            record
            for record in candidates
            if record["entity"] == current["entity"]
            and record["metric"] == current["metric"]
            and record["date"] != current["date"]
            and record["value_text"] != current["value_text"]
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


def _target_alignment_passes(
    question: str,
    final_record: dict[str, Any],
    facts: str,
    llm: DeepSeekFlashClient,
) -> bool:
    verdict = llm.complete(
        TARGET_ALIGNMENT_PROMPT.format(
            target_fact=final_record["fact"],
            facts=facts,
            question=question,
        )
    )
    return unique_verdict_is_yes(verdict)


def _relation_grounding_passes(
    question: str,
    final_record: dict[str, Any],
    facts: str,
    llm: DeepSeekFlashClient,
) -> bool:
    """Reject questions whose descriptive relations contradict their source records."""
    verdict = llm.complete(
        RELATION_GROUNDING_PROMPT.format(
            target_fact=final_record["fact"],
            facts=facts,
            question=question,
        )
    )
    return unique_verdict_is_yes(verdict)


def _difficulty_passes(question: str, targets: list[str], base_model: BlindBaseModel, trials: int) -> bool:
    """0/N gate: accept only if every no-tool base-model guess is wrong by reward F1."""
    for guess in base_model.guesses(question, trials):
        if any(compute_f1(guess, target) >= F1_CORRECT_THRESHOLD for target in targets):
            return False
    return True


@dataclass(frozen=True)
class AttemptFailure:
    reason: str
    detail: str = ""


def _construct_one(root: dict[str, Any], llm: DeepSeekFlashClient, max_loops: int) -> tuple[dict[str, Any] | None, AttemptFailure | None]:
    """Run a record-constrained SELECT -> TEMPLATE -> EXIT synthesis flow."""
    if max_loops != 2:
        return None, AttemptFailure("unsupported_max_loops", "deterministic template requires --max-loops 2")
    involved = [root]
    selected = _select_record(
        current=root,
        query=f"{root['entity']} {root['metric']}",
        prompt_template=SELECT_PROMPT,
        llm=llm,
        exclude_ids={str(root["fact_id"])},
        require_same_entity_metric=True,
    )
    if selected is None:
        return None, AttemptFailure("select_failed")
    involved.append(selected)

    try:
        question = build_anchored_lookup_question(root, selected)
    except ValueError as error:
        return None, AttemptFailure("template_constraint_failed", str(error))
    output = output_record(
        qid="",
        prompt=question,
        root=root,
        involved=involved,
        actions=["SELECT", "TEMPLATE", "EXIT"],
    )
    output["meta"]["question_policy"] = "same_entity_metric_template_v1"
    return output, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--candidates", type=int, default=100, help="number of candidate attempts before difficulty filtering")
    parser.add_argument(
        "--max-loops",
        type=int,
        default=2,
        choices=(2,),
        help="fixed at 2: root fact plus target fact are rendered by a deterministic template",
    )
    parser.add_argument("--difficulty-trials", type=int, default=6, choices=(3, 6))
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=1,
        help="simultaneous remote construction requests; use 2 first and increase only after a clean run",
    )
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
    if not 1 <= args.api_concurrency <= 4:
        print("--api-concurrency must be between 1 and 4", file=sys.stderr)
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
    _status(f"远程构造模型: {API_MODEL}；API 并发={args.api_concurrency}")
    if not sys.stdout.isatty():
        _status("当前标准输出不是终端；请使用 conda run --no-capture-output 以实时查看进度")

    # 远端请求可小并发；本地 GPU 盲猜与 JSONL 写入必须串行，避免显存争用和写入竞争。
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
        next_attempt = 0
        pending: dict[Future[tuple[dict[str, Any] | None, AttemptFailure | None]], tuple[int, dict[str, Any]]] = {}

        def submit_attempt(executor: ThreadPoolExecutor, attempt: int) -> None:
            entity = entities[attempt % len(entities)]
            root = rng.choice(groups[entity])
            future = executor.submit(_construct_one, root, llm, args.max_loops)
            pending[future] = (attempt, root)

        with ThreadPoolExecutor(max_workers=args.api_concurrency, thread_name_prefix="remote-construct") as executor:
            while next_attempt < args.candidates and len(pending) < args.api_concurrency:
                submit_attempt(executor, next_attempt)
                next_attempt += 1

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    attempt, root = pending.pop(future)
                    try:
                        candidate, failure = future.result()
                    except RemoteLLMError as exc:
                        _append_jsonl(
                            rejected_fh,
                            _failure_payload(attempt, root, "remote_llm_error", str(exc)),
                        )
                        rejected += 1
                    except Exception as exc:
                        _append_jsonl(
                            rejected_fh,
                            _failure_payload(attempt, root, "construction_error", str(exc)),
                        )
                        rejected += 1
                    else:
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
                                _status(
                                    f"本地难度模型加载完成，开始执行 "
                                    f"0/{args.difficulty_trials} 无工具盲猜筛选"
                                )
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
                    if next_attempt < args.candidates:
                        submit_attempt(executor, next_attempt)
                        next_attempt += 1

    print(
        json.dumps(
            {
                "attempted": args.candidates,
                "accepted": accepted,
                "rejected": rejected,
                "accepted_output": str(args.accepted_output),
                "rejected_output": str(args.rejected_output),
                "api_concurrency": args.api_concurrency,
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
