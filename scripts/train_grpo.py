#!/usr/bin/env python3
"""First-pass LoRA-GRPO training entry point for the local financial ReAct agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from fintrace.data.jsonl import iter_financial_qa_samples
from fintrace.rewards.constants import MAX_TOOL_CALLS_PER_ROUND
from fintrace.training import (
    PromptMetadata,
    TransformersReActGRPORollout,
    TrajectoryAuditReward,
    VllmReActGRPORollout,
)
from fintrace.tools import ToolAwareHttpClient


def build_retrieval_client(config: dict, endpoint: str | None):
    """Construct the retrieval client selected by config["retrieval"]["adapter"]."""
    retrieval_config = config["retrieval"]
    adapter = retrieval_config.get("adapter", "tool_aware")
    if adapter == "kb":
        from fintrace.kb import KbRetrievalClient
        return KbRetrievalClient(top_k=retrieval_config.get("top_k", 3))
    if adapter == "tool_aware":
        if not endpoint:
            print("Set FINTRACE_TOOL_AWARE_ENDPOINT or pass --endpoint", file=sys.stderr)
            raise SystemExit(2)
        return ToolAwareHttpClient(
            endpoint=endpoint,
            timeout_seconds=retrieval_config["timeout_seconds"],
        )
    raise ValueError(f"unknown retrieval adapter: {adapter!r} (expected 'kb' or 'tool_aware')")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be an object")
    return value


def build_records(dataset_path: Path) -> tuple[list[dict], dict[str, PromptMetadata]]:
    records: list[dict] = []
    metadata_by_prompt: dict[str, PromptMetadata] = {}
    for sample in iter_financial_qa_samples(dataset_path):
        if sample.prompt in metadata_by_prompt:
            raise ValueError("duplicate prompt values are unsupported by the first rollout implementation")
        metadata_by_prompt[sample.prompt] = PromptMetadata(sample.targets, sample.valid_inst)
        records.append(
            {
                "qid": sample.qid,
                "prompt": sample.prompt,
                "source": sample.source,
                "targets": list(sample.targets),
                "valid_inst": sample.valid_inst,
            }
        )
    if not records:
        raise ValueError("training dataset has no valid records")
    return records, metadata_by_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/grpo_lora.example.yaml"))
    parser.add_argument("--endpoint", default=os.getenv("FINTRACE_TOOL_AWARE_ENDPOINT"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoints/grpo_first_pass"))
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="append each sampled ReAct trajectory and reward breakdown to this new JSONL file",
    )
    parser.add_argument(
        "--append-trace",
        action="store_true",
        help="allow appending to an existing --trace-output JSONL instead of rejecting it",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="override training.max_steps from the YAML configuration",
    )
    parser.add_argument("--run", action="store_true", help="perform optimizer updates")
    args = parser.parse_args()

    if args.trace_output is not None and not args.run:
        print("--trace-output requires --run", file=sys.stderr)
        return 2
    if args.trace_output is not None and args.trace_output.exists() and not args.append_trace:
        print(
            f"refusing to append to existing audit file: {args.trace_output} "
            "(choose a new path or pass --append-trace)",
            file=sys.stderr,
        )
        return 2

    config = load_config(args.config)
    model_config = config["model"]
    rollout_config = config["rollout"]
    training_config = config["training"]
    max_steps = args.max_steps if args.max_steps is not None else training_config.get("max_steps", 1)
    if max_steps < 1:
        print("max_steps must be at least 1", file=sys.stderr)
        return 2
    rollout_engine = rollout_config.get("engine", "vllm")
    # vLLM EngineCore 不会自动接收每步更新后的 LoRA 权重，禁止产生误导性的多步训练。
    if rollout_engine == "vllm" and max_steps != 1:
        print(
            "rollout.engine=vllm only permits max_steps=1 because LoRA weight synchronization "
            "between optimizer steps is not implemented. Use rollout.engine=transformers for a multi-step pilot.",
            file=sys.stderr,
        )
        return 2
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    generation_batch_size = (
        training_config["per_device_train_batch_size"]
        * training_config.get("gradient_accumulation_steps", 1)
        * world_size
    )
    if generation_batch_size % rollout_config["num_generations"] != 0:
        print(
            "generation batch size must be divisible by num_generations: "
            f"{generation_batch_size} % {rollout_config['num_generations']} != 0. "
            "Adjust training.per_device_train_batch_size, gradient_accumulation_steps, "
            "or rollout.num_generations.",
            file=sys.stderr,
        )
        return 2
    max_rounds = rollout_config["max_rounds"]
    max_tool_calls_per_round = rollout_config.get(
        "max_tool_calls_per_round", MAX_TOOL_CALLS_PER_ROUND
    )
    if max_tool_calls_per_round != 1:
        # ReAct parser 每轮只接受一个 action，rollout 侧无法产生更多工具调用。
        # 这里只提示，不阻断：历史配置沿用了大于 1 的取值，语义上等价于 1。
        print(
            f"warning: rollout.max_tool_calls_per_round={max_tool_calls_per_round} is inert; "
            "the ReAct parser accepts exactly one action per turn, so each round "
            "issues at most one search.",
            file=sys.stderr,
        )

    records, metadata_by_prompt = build_records(Path(config["data"]["train_path"]))

    retrieval_client = build_retrieval_client(config, args.endpoint)

    summary = {
        "mode": "train" if args.run else "dry-run",
        "records": len(records),
        "model": model_config["base_model_path"],
        "retrieval_adapter": config["retrieval"].get("adapter", "tool_aware"),
        "rollout_engine": rollout_engine,
        "endpoint": args.endpoint,
        "num_generations": rollout_config["num_generations"],
        "max_steps": max_steps,
        # 奖励的终止判据必须与 rollout 用同一组阈值，否则审计里的归零原因会失真。
        "max_rounds": max_rounds,
        "max_tool_calls_per_round": max_tool_calls_per_round,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.run:
        return 0

    # 延迟导入重依赖，使 dry-run 可在未加载 GPU 模型时完成数据与配置校验。
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    tokenizer = AutoTokenizer.from_pretrained(model_config["base_model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model_path"],
        dtype=torch.bfloat16,
    )
    # lora微调参数
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_config["lora_rank"],
        lora_alpha=model_config["lora_alpha"],
        lora_dropout=model_config["lora_dropout"],
        target_modules="all-linear",
    )
    rollout_kwargs = {
        "tokenizer": tokenizer,
        "retrieval_client": retrieval_client,
        "metadata_by_prompt": metadata_by_prompt,
        "max_rounds": max_rounds,
        "max_tokens_per_turn": rollout_config.get("max_tokens_per_turn", 1024),
        "temperature": rollout_config.get("temperature", 1.0),
        "top_p": rollout_config.get("top_p", 1.0),
    }
    if rollout_engine == "transformers":
        rollout = TransformersReActGRPORollout(**rollout_kwargs)
    elif rollout_engine == "vllm":
        rollout = VllmReActGRPORollout(
            model_path=model_config["base_model_path"],
            dtype=model_config["torch_dtype"],
            **rollout_kwargs,
        )
    else:
        raise ValueError("rollout.engine must be 'transformers' or 'vllm'")

    # wandb 可视化：TRL 只消费已存在的 wandb.run（见 GRPOTrainer.log），故自行初始化。
    report_to = "none"
    try:
        import wandb
    except ImportError:
        print("wandb not installed; metrics fall back to report_to=none", file=sys.stderr)
        wandb = None  # type: ignore[assignment]

    if wandb is not None:
        run_section = config.get("run", {})
        wandb.init(
            project=run_section.get("project", "fintrace"),
            name=run_section.get("name") or args.output_dir.name,
            config={
                "summary": summary,
                "rollout": rollout_config,
                "retrieval": config["retrieval"],
                "training": training_config,
            },
        )
        report_to = "wandb"

    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"trajectory_audit": str(args.trace_output)}, ensure_ascii=False))

    # rl训练核心配置
    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=training_config["learning_rate"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        max_steps=max_steps,
        num_generations=rollout_config["num_generations"],
        max_completion_length=rollout_config["max_response_tokens"],
        beta=training_config["kl_coefficient"],
        epsilon=training_config.get("epsilon", 0.2),
        bf16=True,
        use_vllm=False,
        remove_unused_columns=False,
        report_to=report_to,
        save_strategy="steps",
        save_steps=training_config.get("save_steps", 1),
        save_total_limit=training_config.get("save_total_limit"),
        logging_strategy="steps",
        logging_steps=training_config.get("logging_steps", 1),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=TrajectoryAuditReward(
            args.trace_output,
            max_rounds=max_rounds,
            max_tool_calls_per_round=max_tool_calls_per_round,
        ),
        args=grpo_config,
        train_dataset=Dataset.from_list(records),
        processing_class=tokenizer,
        peft_config=lora_config,
        rollout_func=rollout,
    )
    try:
        trainer.train()
        trainer.save_model(str(args.output_dir))
    finally:
        if report_to == "wandb":
            wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
