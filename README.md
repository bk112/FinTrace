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

The local reproduction baseline is Python 3.11, Torch 2.11.0+cu130, TRL 1.9,
and vLLM 0.26. The project uses its own vLLM rollout adapter because TRL 1.9
warns that its built-in vLLM integration is verified only through vLLM 0.25.1.

## Initial Verification

The supplied reward tests have no third-party dependency:

```bash
PYTHONPATH=src python tests/unit/test_rewards.py
```

Install the training dependencies only in a Python 3.10-3.12 environment:

```bash
python -m pip install -e '.[train,dev]'
```

## Smoke Rollout

Once the Tool-aware endpoint contract is configured, run one real local-model
rollout without an optimizer update:

```bash
PYTHONPATH=src python scripts/rollout_smoke.py \
  --prompt '...' \
  --ground-truth '...' \
  --endpoint "$FINTRACE_TOOL_AWARE_ENDPOINT"
```

## Local Tool-aware Service

Run the local retrieval service in one terminal. It uses no API key, fetches
public pages, records citable source URLs, and caches each query result.

```bash
PYTHONPATH=src python scripts/serve_tool_aware.py
```

Then use `http://127.0.0.1:8765/retrieve` as the rollout endpoint. For
training and evaluation, preserve the generated retrieval cache with the run
report so a later replay uses the same evidence.

## First GRPO Step

The first training entry validates data and configuration by default. Add
`--run` only after the local retrieval service is healthy. This first version
intentionally permits one optimizer step, because vLLM LoRA synchronization
between steps is the next implementation item.

```bash
PYTHONPATH=src python scripts/train_grpo.py \
  --endpoint http://127.0.0.1:8765/retrieve

PYTHONPATH=src python scripts/train_grpo.py \
  --endpoint http://127.0.0.1:8765/retrieve \
  --run
```
