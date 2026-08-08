# FinTrace

FinTrace trains the execution layer of a financial multi-hop retrieval agent.
The existing planner and Tool-aware retrieval interface remain outside this
repository's training scope. The target agent follows a ReAct trajectory:

`think -> search -> observation -> answer`

## Scope

- Base model: local `Qwen2.5-3B-Instruct`.
- Training: LoRA-based GRPO.
- Rollout: custom multi-turn ReAct loop using vLLM stop sequences.
- Reward: the supplied five-dimensional reward implementation.
- Retrieval: the existing Tool-aware service, through an adapter to be added.

## Layout

- `src/fintrace/rewards/`: validated reward domain module.
- `src/fintrace/rollout/`: ReAct generation and trajectory assembly.
- `src/fintrace/tools/`: Tool-aware retrieval adapter.
- `src/fintrace/training/`: GRPO orchestration.
- `src/fintrace/data/`: dataset schemas and preparation.
- `configs/`: versioned, non-secret experiment configuration.
- `data/`: local datasets; contents are intentionally ignored by Git.
- `artifacts/`: checkpoints, trajectories, and reports; ignored by Git.
- `tests/`: unit and integration tests.

## Local Model Convention

The default base model is read from `configs/train/grpo_lora.example.yaml`:

`/media/xdhpc/data/whr/models/Qwen2.5-3B-Instruct`

The model directory is external input and must not be modified by project
scripts. Copy the example configuration for a real run and keep machine- or
credential-specific overrides outside version control.

## Initial Verification

The supplied reward tests have no third-party dependency:

```bash
PYTHONPATH=src python tests/unit/test_rewards.py
```

Install the training dependencies only in a Python 3.10-3.12 environment:

```bash
python -m pip install -e '.[train,dev]'
```
