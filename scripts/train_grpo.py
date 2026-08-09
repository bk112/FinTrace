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
from fintrace.training import (
    PromptMetadata,
    TransformersReActGRPORollout,
    VllmReActGRPORollout,
    financial_trajectory_reward,
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
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--run", action="store_true", help="perform the one-step optimizer update")
    args = parser.parse_args()

    if args.max_steps != 1:
        print(
            "First-pass script only permits --max-steps 1: vLLM LoRA weight synchronization "
            "between optimizer steps is not implemented yet.",
            file=sys.stderr,
        )
        return 2

    config = load_config(args.config)
    model_config = config["model"]
    rollout_config = config["rollout"]
    training_config = config["training"]
    records, metadata_by_prompt = build_records(Path(config["data"]["train_path"]))

    retrieval_client = build_retrieval_client(config, args.endpoint)

    summary = {
        "mode": "train" if args.run else "dry-run",
        "records": len(records),
        "model": model_config["base_model_path"],
        "retrieval_adapter": config["retrieval"].get("adapter", "tool_aware"),
        "rollout_engine": rollout_config.get("engine", "vllm"),
        "endpoint": args.endpoint,
        "num_generations": rollout_config["num_generations"],
        "max_steps": args.max_steps,
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
        torch_dtype=torch.bfloat16,
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
        "max_rounds": rollout_config["max_rounds"],
        "max_tokens_per_turn": rollout_config.get("max_tokens_per_turn", 1024),
        "temperature": rollout_config.get("temperature", 1.0),
        "top_p": rollout_config.get("top_p", 1.0),
    }
    rollout_engine = rollout_config.get("engine", "vllm")
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

    # rl训练核心配置
    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=training_config["learning_rate"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        max_steps=args.max_steps,
        num_generations=rollout_config["num_generations"],
        max_completion_length=rollout_config["max_response_tokens"],
        beta=training_config["kl_coefficient"],
        epsilon=training_config.get("epsilon", 0.2),
        bf16=True,
        use_vllm=False,
        remove_unused_columns=False,
        report_to="none",
        save_strategy="steps",
        save_steps=1,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=financial_trajectory_reward,
        args=grpo_config,
        train_dataset=Dataset.from_list(records),
        processing_class=tokenizer,
        peft_config=lora_config,
        rollout_func=rollout,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
