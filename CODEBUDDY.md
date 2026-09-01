# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## 常用命令

### 安装依赖

训练依赖只能装在 Python 3.10–3.12 环境：`python -m pip install -e '.[train,dev]'`。
可选组：`rollout`（vLLM 0.26）、`kb-build`（akshare）、`synthesis`（dotenv/requests/tqdm）。
基础依赖只有 PyYAML，因此 reward 与 rollout 的单测不需要 GPU 相关包即可运行。

### 测试

`pytest -q`（`pyproject.toml` 已配置 `testpaths=tests` 与 `pythonpath=src`，无需设置环境变量）。
运行单个用例：`pytest -q tests/unit/test_react_rollout.py -k repeated_query`。
reward 测试无第三方依赖，也可直接执行：`PYTHONPATH=src python tests/unit/test_rewards.py`
（PowerShell 写法：`$env:PYTHONPATH="src"; python tests/unit/test_rewards.py`）。
`make test` 先跑 reward 脚本再跑 pytest；`make compile` 执行 `python -m compileall -q src tests`。

### 代码检查

`ruff check .` 与 `ruff format .`；配置在 `pyproject.toml` 中，行宽 100、目标版本 py310。

### 环境自检

`PYTHONPATH=src python scripts/check_environment.py` 会打印 Python 版本、torch/transformers/trl/peft/accelerate/datasets/vllm 版本以及 CUDA 状态。任何 GPU 运行前应先执行。

### 启动本地检索服务

`PYTHONPATH=src python scripts/serve_tool_aware.py` 提供 `POST http://127.0.0.1:8765/retrieve` 与 `GET /health`。
不需要 API Key，结果按查询哈希缓存到 `data/interim/retrieval_cache`；该缓存应随实验报告一同保留，以便复放时使用同一份证据。

### 冒烟 rollout（单条轨迹，不做优化器更新）

```
PYTHONPATH=src python scripts/rollout_smoke.py --prompt '...' --ground-truth '...' --endpoint http://127.0.0.1:8765/retrieve
```

### GRPO 训练

默认是不加载模型的 dry-run，只校验数据集与配置：

```
PYTHONPATH=src python scripts/train_grpo.py --config configs/train/grpo_lora.example.yaml
PYTHONPATH=src python scripts/train_grpo.py --config ... --run --trace-output artifacts/audits/<new>.jsonl
```

`--trace-output` 必须配合 `--run`，且遇到已存在文件会拒绝写入，除非显式传 `--append-trace`。
`--max-steps N` 可覆盖 `training.max_steps`，但 `rollout.engine: vllm` 会拒绝任何 `N != 1`。

### 数据集校验

`PYTHONPATH=src python scripts/validate_dataset.py data/processed/<file>.jsonl`（schema 校验 + qid 查重）。
`PYTHONPATH=src python scripts/validate_rl_targets.py --input ... --report ... [--filtered-output ...]` 审计基于记录锚定的 target，并产出过滤后的训练副本；两个输出路径都必须是新路径。
`PYTHONPATH=src python scripts/summarize_trajectory_audit.py` 将轨迹审计 JSONL 汇总为 rollout 与 GRPO 分组健康度指标。

### 知识库构建流水线（按序执行）

`build_kb_stage2_scrape.py` → `build_kb_stage2b_announcements.py` / `build_kb_announcements.py` → `build_kb_stage3_local.py` / `build_kb_stage3_opensource.py` → `build_kb_stage4_merge.py` → `build_kb_stage5_index.py`。
`build_kb_enrich_industry.py` 是合并后的幂等补充脚本。产物位于 `data/kb`（`records.jsonl`、FAISS 索引、manifest）。

### RL 数据集合成

`PYTHONPATH=src python scripts/construct_rl_dataset.py --candidates 100 --api-concurrency 2` 需要 `.env` 中的 `CODEBUDDY_API_KEY`。它用远端 DeepSeek 模型挑选关联事实，并用本地无工具基座模型的 0/N 盲猜做难度闸门。

## 架构总览

### 目标与边界

FinTrace 只训练金融多跳检索 Agent 的**执行层**。Planner 与线上 Tool-aware 检索服务都在本仓库之外，按外部契约对待。Agent 遵循严格的 ReAct 轨迹 `think -> search -> observation -> answer`，基座为本地 `Qwen2.5-3B-Instruct`，采用 LoRA + GRPO 训练，奖励为该五维实现。Rollout 使用 vLLM 的 stop 序列自行实现，而不是 TRL 内置的 vLLM 集成——因为 TRL 1.9 声明其集成仅验证到 vLLM 0.25.1，而本地基线是 vLLM 0.26。

### 端到端数据流

1. `scripts/` 下的 KB 构建脚本把公告、AkShare 数据、FinanceComplexQA 统一为 v1.1 的 `KnowledgeRecord` schema（`src/fintrace/knowledge_base/schema.py`），合并去重后用 `Qwen3-Embedding-0.6B` 在 `data/kb/` 建立 FAISS 索引。
2. `scripts/construct_rl_dataset.py` 从 KB 采样种子事实，用远端 LLM 挑选“同实体、同指标、不同报告期、数值不同”的关联事实，再用**确定性模板** `build_anchored_lookup_question`（`src/fintrace/data/agentic_synthesis.py`）生成问题，保证任何数值都不是模型编造的；随后依次通过唯一性、target 对齐、关系接地与本地 0/N 盲猜难度闸门。
3. `scripts/validate_rl_targets.py` 剔除两类样本：问题要求计算差值/比例/百分点变化而 target 只是原始记录值的；以及问题中关系与源记录矛盾的。
4. `scripts/train_grpo.py` 加载存活下来的 JSONL，为每个 prompt 构造 `PromptMetadata(targets, valid_inst)` 映射，连同数据一起交给 TRL 的 `GRPOTrainer`。

### 两套 rollout 实现（不要混淆）

仓库里存在两个并行的 ReAct 循环，终止语义相同，但处理对象不同：

- `src/fintrace/rollout/` 是**基于文本、与引擎解耦**的实现。`ReActRollout` 只依赖 `GenerationEngine` 协议（`generate(prompt, stop_sequences) -> str`），调用检索客户端后把 `<observation>...</observation>` 拼回 prompt 字符串。目前只提供了 `VllmGenerationEngine`。`scripts/rollout_smoke.py` 与 parser/masking 单测使用这条链路。它会记录 `TraceSegment(owner="assistant"|"environment")`，供 `tokenize_trace_with_env_mask` 生成 token 级掩码：整段 completion **只 tokenize 一次**，任何跨越所有权边界的 token 直接报错而不是被随意归属。
- `src/fintrace/training/react_grpo_rollout.py` 是**基于 token ID、面向 TRL** 的实现。`VllmReActGRPORollout.__call__(prompts, trainer)` 返回 TRL 实验性 `rollout_func` 所需的字典：`prompt_ids`、`completion_ids`、`logprobs`、`env_mask`，外加奖励函数要消费的 `trajectories` 与 `targets`。它用 `prompt_token_ids` 续写，而不是重新拼接字符串，以保证训练时的 token 序列与采样上下文严格一致。

基于 token ID 的 rollout 有几条关键不变量：

- 环境文本用本地分词结果追加，**logprob 填 0.0**，`env_mask` 置 `0`，因此 GRPO 永远不会优化检索到的证据；assistant 片段的 `env_mask` 为 `1`。
- `_strip_trailing_chat_end` 会把采样出的 `<|im_end|>` 从可训练区间中剔除（它属于 chat 模板边界而非 ReAct 内容），而 `_qwen_observation_bridge` 在桥接下一轮时再补回。改动模板时两者必须同步维护。
- TRL 的 `RepeatSampler` 已经把每个 prompt 复制了 `num_generations` 次，rollout 对每个输入只能生成一次，否则各字段 batch 维度不一致。
- `TransformersReActGRPORollout` 继承 vLLM 版 rollout，只覆写 `_generate_chunk`，在 `inference_mode()` 下调用 `trainer.model.generate` 并从 logits 重算 logprob。它存在的理由是 vLLM 的 EngineCore 不会在优化步之间接收 LoRA 权重更新。因此 `scripts/train_grpo.py` 硬性拒绝 `rollout.engine: vllm` 且 `max_steps != 1`；多步实验必须设置 `engine: transformers`（参见 `configs/train/grpo_rl_synthesis_pilot.yaml`）。

### 奖励层

`src/fintrace/rewards/reward_functions.py` 是给定的、已验证的**单 target** 模块。`compute_total_reward(Trajectory)` 返回 `RewardBreakdown`，包含五个分量：答案正确性（F1 + 数值近似匹配，权重 0.4）、答案 CEM（0.2）、格式合规（0.2）、搜索激励（0.1，最多计两次搜索）、检索正确（0.1），最后把总分 clamp 到 `[0, 1]`。`should_terminate_with_zero_reward` 是一票否决：没有最终答案、超过 `MAX_ROUNDS`、单轮工具调用过多、重复 query、标签不配对，都会让整条轨迹归零。当 `valid_inst` 为真且答案命中“无法回答”关键词表时，正确性维度返回负的 `INVALID_ANSWER_PENALTY`，经 clamp 后的实际效果是抵消其他正向分量，而不是让总分为负。所有权重与阈值集中在 `rewards/constants.py`，不要硬编码。

`src/fintrace/training/reward_adapter.py` 负责把单 target 模块桥接到多 target 数据。`best_reward_breakdown` 会对每个可接受的 target 表述重新打分并取最大值，因为同一事实存在多种规范表述。传给 `GRPOTrainer` 的可调用对象是 `TrajectoryAuditReward`：它可选地把每条采样的审计行（prompt、轨迹、每次检索的预览、完整分维度明细）追加写入 JSONL，并在存在 `wandb.run` 时记录聚合指标。

### 检索边界

rollout 代码只依赖 `src/fintrace/tools/base.py` 中的 `RetrievalClient` 协议（`search(query) -> RetrievalResult(query, text, metadata)`）。具体实现由 `configs/*.yaml` 的 `retrieval.adapter` 选择：

- `kb`（训练推荐，进程内）：`KbRetrievalClient` 包装 `src/fintrace/kb/service.py`，后者用加锁的懒加载读取 FAISS 索引与记录，并在服务前校验 manifest（embedding 模型名、记录数、向量维度、`records_sha256`）。`format_result` 会把清洗后的结构化记录以 `<!-- metadata: {...} -->` JSON 块内嵌进 observation 文本，`KbRetrievalClient` 同时以 `metadata["records"]` 返回同一份数据；检索正确性奖励读取该列表，并优先按规范化的 `value_text`/`unit` 判定，避免把“15.3亿元”误判为“15.3%”。
- `tool_aware`（HTTP）：`ToolAwareHttpClient` 只负责传输、超时与响应归一化，端点来自 `FINTRACE_TOOL_AWARE_ENDPOINT` 或 `--endpoint`。

`src/fintrace/tools/local_tool_aware.py` 与 `local_web.py` 实现了一个免密钥的本地 Tool-aware 替代方案：DuckDuckGo HTML 搜索 → 按域名优先级（cninfo、sse、szse、hkexnews、sec）抓取正文，带 SSRF 防护（拒绝内网/回环/保留地址）、正文长度截断，以及基于 SHA-256 的查询缓存，保证多步 RL 复放同一份证据。

注意一处重复：`src/fintrace/knowledge_base/` 里还有一套类型更严格的 `FaissKnowledgeBase`/`FaissRetrievalClient`，其 observation 格式与 `fintrace.kb` 不同。新工作优先用 `fintrace.kb`，不要擅自合并两套实现。

### 解析契约

`src/fintrace/rollout/parser.py` 刻意做严格校验：一轮必须恰好是 `<think>...</think>` 加 `<search>...</search>` 或 `<answer>...</answer>` 之一，内容部分用 `[^<]*` 限制，防止模型把第二个标签吞进内容里，且 `think` 与动作内容均不可为空。任何偏差都会抛出 `ReActParseError`，rollout 将其转化为 `RolloutTermination.MALFORMED_ACTION`，进而导致零奖励。`react_grpo_rollout.py` 中的 `AGENT_SYSTEM_PROMPT` 是配套的提示词契约，其中明确声明 observation 是不可信数据，不得执行其中的指令。

### 配置与密钥

`configs/` 存放纳入版本控制的非敏感 YAML。以 `.example.yaml` 结尾的是模板——实际运行时请复制一份，机器相关路径、端点与凭据一律不得入库。`.env` 已被 git 忽略，只有 `construct_rl_dataset.py` 会读取 `CODEBUDDY_API_KEY`。基座模型目录属于外部输入，项目脚本绝不能修改它。

### 仓库数据策略

`data/`（数据集、知识库、检索缓存）是刻意纳入版本管理的；只有超大的 `data/kb/*.index` 走 Git LFS，`data/FinanceComplexQA` 是子模块。`artifacts/checkpoints`、`artifacts/rollouts`、`artifacts/reports` 除占位用的 `.gitignore` 外均被忽略，而 `artifacts/audits` 与 `artifacts/data_audits` 保留轨迹与数据审计 JSONL。多数脚本拒绝覆盖已有输出文件，要求传入全新路径。
