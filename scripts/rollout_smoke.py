#!/usr/bin/env python3
"""Run one real multi-turn ReAct rollout against the local model and retrieval service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from fintrace.rewards import compute_total_reward
from fintrace.rollout import ReActRollout, VllmGenerationEngine
from fintrace.tools import ToolAwareHttpClient


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be an object")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/grpo_lora.example.yaml"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--endpoint", default=os.getenv("FINTRACE_TOOL_AWARE_ENDPOINT"))
    parser.add_argument("--response-text-field", default="text")
    args = parser.parse_args()

    if not args.endpoint:
        print("Missing retrieval endpoint: set FINTRACE_TOOL_AWARE_ENDPOINT or --endpoint", file=sys.stderr)
        return 2

    config = load_config(args.config)
    model_config = config["model"]
    rollout_config = config["rollout"]
    retrieval_config = config["retrieval"]

    engine = VllmGenerationEngine(
        model_config["base_model_path"],
        max_tokens=rollout_config["max_response_tokens"],
        dtype=model_config["torch_dtype"],
    )
    client = ToolAwareHttpClient(
        endpoint=args.endpoint,
        timeout_seconds=retrieval_config["timeout_seconds"],
        response_text_field=args.response_text_field,
    )
    rollout = ReActRollout(
        engine,
        client,
        max_rounds=rollout_config["max_rounds"],
        stop_sequences=tuple(rollout_config["stop_sequences"]),
    )
    result = rollout.run(args.prompt, args.ground_truth)
    # 此处只做推理链路验证，reward 用于观察，不会执行反向传播或更新 LoRA 权重。
    breakdown = compute_total_reward(result.trajectory)

    print(
        json.dumps(
            {
                "completed": result.completed,
                "termination": result.termination.value,
                "reward": breakdown.total,
                "searches": [step.query for step in result.trajectory.search_steps],
                "assistant_trace": result.trajectory.raw_text,
                "environment_characters": sum(
                    len(segment.text) for segment in result.segments if segment.owner == "environment"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
