# PPO & GRPO Baselines for VarExpr24 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add standard PPO and GRPO baselines for the VarExpr24 task using HuggingFace TRL, with the same reward, training budget, and comprehensive metric evaluation as the existing GFlowNet methods.

**Architecture:** Standalone training scripts using TRL's `GRPOTrainer` (and a manual PPO loop using the same infrastructure) that share the same Llama-3.2-1B + LoRA setup, Expr24Validator reward, and grammar-constrained generation. After training, a shared evaluation script samples from trained models and computes all metrics (accuracy, diversity, novelty, prefix collapse, length distribution) using existing utility functions.

**Tech Stack:** TRL (upgrade to >=0.25), transformers, peft, transformers_cfg, wandb, torch 2.8, Hydra (for config reuse)

**Python:** `/data1/xw3763/miniforge3/envs/torch/bin/python`

---

## Context & Key Decisions

### Why separate scripts (not integrated into ChemGFNModule)?
- PPO/GRPO are standard RL algorithms with fundamentally different training loops
- TRL provides battle-tested implementations; no need to reinvent
- Keeps the GFlowNet codebase clean
- Fairer comparison: each method uses its canonical implementation

### TRL Version
- Current env: `trl==0.9.4` — no GRPOTrainer
- **Action: upgrade to `trl>=0.25.0`** (has GRPOTrainer as first-class API)
- PPO: TRL v0.9.4's PPOTrainer API changed significantly in v0.25+; both now live under `trl.experimental.ppo` or top-level. After upgrade, we'll use the new API.
- 需要确认 upgrade 后不会 break transformers_cfg 或其他依赖

### Reward Consistency
- Both PPO and GRPO use **exactly the same** `Expr24Validator(scorer="hit24_dense")` reward
- `global_score`: float in [0,1], 1.0 iff expression equals 24
- This is the **same** reward function used in GFlowNet training (the task-specific component)

### Grammar Constraints
- Use `transformers_cfg.GrammarConstrainedLogitsProcessor` during generation
- Grammar: `assets/24_grammars/var_length.ebnf`
- Legal tokens: `assets/token_list/24_points/general` (digits 0-9, +-*/)
- Min length: 3 tokens, Max length: 9 tokens
- 需要确认 GRPOTrainer 支持自定义 `logits_processor` 参数

### Training Budget (match GFlowNet configs)
- 5000 training steps
- 32 samples per step (n_samples=32)
- Gradient accumulation: 4
- Learning rate: 1e-4 (AdamW)
- Precision: bf16
- LoRA: rank=16, alpha=16, targets=[q,k,v,o,gate,down,up]_proj

### Prompt
- File: `data/24_points/prompts.txt` — a long system prompt for the 24 Game
- Tokenized and used as prefix for all generations

### Metrics to Compute (post-training evaluation)
1. **Accuracy**: hit24 rate (fraction of expressions that equal 24)
2. **Token-level diversity**: Per-position entropy averaged across positions
3. **Diversity by length**: Diversity grouped by expression length
4. **Text embedding diversity**: SentenceTransformer cosine distance
5. **Top-K Levenshtein diversity**: Mean pairwise edit distance of top-K valid expressions
6. **Top-K Levenshtein novelty**: Mean min edit distance to training buffer
7. **Prefix collapse**: Per-position token frequency concentration (top1_mass, entropy)
8. **Length distribution**: Count and score aggregated by sequence length
9. **Log to wandb** with comparable metric names

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scripts/train_grpo_expr24.py` | GRPO training using TRL GRPOTrainer |
| `scripts/train_ppo_expr24.py` | PPO training using TRL PPOTrainer (manual loop) |
| `scripts/eval_rl_baseline.py` | Shared evaluation: load LoRA model, sample, compute all metrics |
| `scripts/run_rl_baselines.sh` | Launch script for training + eval |
| `configs/experiment/VarExpr24/VarExpr24_PPO.yaml` | Hydra config (for eval compatibility) |
| `configs/experiment/VarExpr24/VarExpr24_GRPO.yaml` | Hydra config (for eval compatibility) |
| `tests/test_rl_baselines.py` | Smoke tests for reward, grammar, train loop |

### Existing files reused (read-only)
- `chemgfn/models/validators.py` — `Expr24Validator`
- `chemgfn/utils/sequence_metrics.py` — `levenshtein_diversity`, `levenshtein_novelty`, `select_topk`
- `chemgfn/utils/diversity.py` — `SequenceDiversity`
- `chemgfn/utils/prefix_metrics.py` — `prefix_collapse_by_position`, `prefix_collapse_by_k`
- `chemgfn/utils/gfn_utils.py` — `prepare_token_mask`, `calculate_diversity_by_length`
- `assets/24_grammars/var_length.ebnf` — Grammar file
- `assets/token_list/24_points/general` — Legal token list
- `data/24_points/prompts.txt` — Prompt
- `data/24_points/buffer_24_non_zero.pt` — Buffer for novelty comparison

---

## Task Breakdown

### Task 0: Environment Setup & TRL Upgrade

**Files:** None (environment only)

- [ ] **Step 0.1: Check current TRL and upgrade**

```bash
/data1/xw3763/miniforge3/envs/torch/bin/pip install "trl>=0.25.0" --upgrade
```

- [ ] **Step 0.2: Verify GRPOTrainer is available**

```bash
/data1/xw3763/miniforge3/envs/torch/bin/python -c "from trl import GRPOTrainer, GRPOConfig; print('OK')"
```

- [ ] **Step 0.3: Verify existing dependencies still work**

```bash
/data1/xw3763/miniforge3/envs/torch/bin/python -c "
import transformers_cfg; print('transformers_cfg:', transformers_cfg.__version__)
import peft; print('peft:', peft.__version__)
import torch; print('torch:', torch.__version__)
from chemgfn.models.validators import Expr24Validator; print('Expr24Validator: OK')
"
```

- [ ] **Step 0.4: If incompatibilities, pin TRL to a compatible version**

---

### Task 1: GRPO Training Script

**Files:**
- Create: `scripts/train_grpo_expr24.py`

核心逻辑:
- GRPOTrainer 需要一个 `Dataset` (带 `"prompt"` 列), 一个 `reward_funcs`, 和 `GRPOConfig`
- Prompt 从 `data/24_points/prompts.txt` 加载
- Reward function 用 Expr24Validator 打分
- 需要 grammar constraint 作为 logits_processor 注入到 generation 中
- LoRA config 与 GFlowNet 实验一致

- [ ] **Step 1.1: Write reward function wrapper**

Reward function 需要符合 TRL 的签名: `reward_func(completions, **kwargs) -> list[float]`

```python
# GRPOTrainer passes completions as list[str] (decoded text)
# We need to parse each expression and score it
def expr24_reward(completions, **kwargs):
    scores = []
    for text in completions:
        # Strip prompt, whitespace, special tokens
        expr = text.strip()
        is_valid, score, value = validator._score_expression(expr)
        scores.append(float(score))
    return scores
```

注意: GRPOTrainer 的 reward_func 接收的 `completions` 格式取决于 TRL 版本:
- 新版 (v0.25+): `completions` 是 `list[str]` (decoded completion text, 不含 prompt)
- 需要查看文档确认是 chat format `list[list[dict]]` 还是 plain text `list[str]`

- [ ] **Step 1.2: Write grammar-constrained generation setup**

```python
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
```

确认 GRPOTrainer 是否支持 `generation_config` 或 `logits_processor` 参数。如果不直接支持，可能需要:
- Override model 的 `generate()` 方法
- 或者在 `GRPOConfig` 中通过 `generation_kwargs` 传递

- [ ] **Step 1.3: Write the complete training script**

关键参数映射 (GFlowNet → GRPO):
```
GFlowNet                    GRPO
---------                   ----
max_steps: 5000             num_train_epochs: 计算等价值
n_samples: 32               num_generations: 32
accumulate_grad_batches: 4  gradient_accumulation_steps: 4
lr: 1e-4                    learning_rate: 1e-4
precision: bf16             bf16: True
LoRA r=16, alpha=16         peft_config: LoraConfig(r=16, lora_alpha=16, ...)
```

GRPOConfig 特有参数:
```
beta: 0.05          # KL penalty coefficient (相当于 GFlowNet 的 reference scaling)
num_generations: 32  # Group size for GRPO advantage normalization
max_completion_length: 9  # max_sentence_len
```

- [ ] **Step 1.4: Add wandb logging**

```python
GRPOConfig(
    report_to="wandb",
    run_name="VarExpr24_GRPO",
    ...
)
```

- [ ] **Step 1.5: Test script runs for 10 steps without error**

```bash
/data1/xw3763/miniforge3/envs/torch/bin/python scripts/train_grpo_expr24.py \
    --max_steps 10 --output_dir ./logs/test_grpo
```

- [ ] **Step 1.6: Commit**

---

### Task 2: PPO Training Script

**Files:**
- Create: `scripts/train_ppo_expr24.py`

核心逻辑:
- TRL v0.25+ 的 PPOTrainer API 可能与 v0.9.4 不同
- PPO 需要 value head (AutoModelForCausalLMWithValueHead)
- 训练循环: generate → compute reward → PPO step
- 同样需要 grammar constraints

注意 TRL v0.25+ 中 PPO 的 API 变化:
- 可能已改为 `OnlineDPOTrainer` 或保留 `PPOTrainer`
- 需要查文档确认

- [ ] **Step 2.1: Write PPO training script with manual loop**

核心循环:
```python
for step in range(max_steps):
    # 1. Generate with grammar constraints
    query_tensors = tokenize_prompt(prompt)
    response_tensors = ppo_trainer.generate(query_tensors, **gen_kwargs)

    # 2. Compute reward
    decoded = tokenizer.batch_decode(response_tensors)
    rewards = [compute_expr24_reward(text) for text in decoded]
    reward_tensors = [torch.tensor(r) for r in rewards]

    # 3. PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

    # 4. Log to wandb
    ppo_trainer.log_stats(stats, batch, rewards)
```

- [ ] **Step 2.2: Match training budget**

```python
PPOConfig(
    batch_size=32,
    mini_batch_size=8,
    ppo_epochs=4,  # PPO inner epochs
    learning_rate=1e-4,
    gradient_accumulation_steps=4,
    max_grad_norm=0.5,
    log_with="wandb",
)
```

- [ ] **Step 2.3: Add grammar constraints to generation**

```python
gen_kwargs = {
    "max_new_tokens": 9,
    "min_new_tokens": 3,
    "do_sample": True,
    "temperature": 1.0,
    "logits_processor": LogitsProcessorList([grammar_processor]),
}
```

- [ ] **Step 2.4: Test script runs for 10 steps**

```bash
/data1/xw3763/miniforge3/envs/torch/bin/python scripts/train_ppo_expr24.py \
    --max_steps 10 --output_dir ./logs/test_ppo
```

- [ ] **Step 2.5: Commit**

---

### Task 3: Evaluation Script (Shared)

**Files:**
- Create: `scripts/eval_rl_baseline.py`

This script loads a trained LoRA checkpoint, generates samples with grammar constraints, and computes all metrics matching the GFlowNet evaluation pipeline.

- [ ] **Step 3.1: Write model loading and generation**

```python
def load_model_and_generate(
    model_path: str,
    prompt_path: str,
    grammar_path: str,
    legal_tokens_path: str,
    n_samples: int = 6400,  # 200 batches * 32 samples, matching GFlowNet eval
    batch_size: int = 32,
    max_len: int = 9,
    min_len: int = 3,
) -> tuple[list[str], list[torch.Tensor]]:
    """Load model, generate samples with grammar constraints, return decoded strings and token tensors."""
```

Generation 使用 `model.generate()` with:
- `GrammarConstrainedLogitsProcessor` from transformers_cfg
- Legal token mask (只允许 0-9, +-*/, EOS)
- Temperature 1.0

- [ ] **Step 3.2: Write metrics computation**

```python
def compute_all_metrics(
    generated_texts: list[str],
    generated_tokens: torch.Tensor,
    tokenizer,
    training_buffer_path: str | None = None,
) -> dict[str, float]:
```

Metrics to compute (reusing existing utility functions):

```python
from chemgfn.models.validators import Expr24Validator
from chemgfn.utils.sequence_metrics import levenshtein_diversity, levenshtein_novelty, select_topk
from chemgfn.utils.diversity import SequenceDiversity
from chemgfn.utils.prefix_metrics import prefix_collapse_by_position
from chemgfn.utils.gfn_utils import calculate_diversity_by_length

# 1. Accuracy
validator = Expr24Validator(scorer="hit24_dense")
result = validator(token_tensor, tokenizer)
accuracy = result["global_score"].mean().item()

# 2. Token-level diversity (entropy per position)
# Reuse _calculate_diversity_ragged logic

# 3. Text embedding diversity
seq_div = SequenceDiversity("sequence_embedding")
text_diversity = seq_div(valid_texts)

# 4. Top-K Levenshtein diversity & novelty
topk_seqs, topk_scores = select_topk(valid_texts, valid_scores, k=100)
topk_diversity = levenshtein_diversity(topk_seqs)
topk_novelty = levenshtein_novelty(topk_seqs, training_set)

# 5. Prefix collapse
prefix_stats = prefix_collapse_by_position(token_tensor, eos_id)

# 6. Length distribution
length_counts = {}  # count by expression length

# 7. Diversity by length
div_by_len = calculate_diversity_by_length(all_token_ids, eos_id)
```

- [ ] **Step 3.3: Write wandb logging and CSV/JSON output**

```python
def log_results(metrics: dict, exp_name: str, output_dir: str):
    """Log to wandb and save CSV/JSON matching GFlowNet eval format."""
    wandb.init(project="ChemGFN", name=exp_name)
    wandb.log(metrics)

    # Save JSON with all metrics
    with open(f"{output_dir}/eval_metrics_{exp_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save CSV with per-sample data
    # columns: expression, score, length, valid
```

- [ ] **Step 3.4: Write CLI interface**

```python
# Usage:
# python scripts/eval_rl_baseline.py \
#     --model_path ./logs/grpo_expr24/final \
#     --exp_name VarExpr24_GRPO \
#     --n_samples 6400 \
#     --output_dir ./logs/eval_grpo
```

Arguments:
- `--model_path`: Path to trained LoRA adapter (or merged model)
- `--base_model`: Base model name (default: meta-llama/Llama-3.2-1B)
- `--exp_name`: Experiment name for wandb
- `--n_samples`: Total samples to generate (default: 6400)
- `--batch_size`: Generation batch size (default: 32)
- `--output_dir`: Where to save results
- `--seed`: Random seed (default: 42)
- `--test_repeats`: Number of evaluation repeats (default: 3)

- [ ] **Step 3.5: Test eval script with untrained model (should show ~0% accuracy)**

```bash
/data1/xw3763/miniforge3/envs/torch/bin/python scripts/eval_rl_baseline.py \
    --model_path meta-llama/Llama-3.2-1B \
    --exp_name test_base --n_samples 64 --output_dir ./logs/test_eval
```

- [ ] **Step 3.6: Commit**

---

### Task 4: Launch Script & Experiment Configs

**Files:**
- Create: `scripts/run_rl_baselines.sh`
- Create: `configs/experiment/VarExpr24/VarExpr24_PPO.yaml`
- Create: `configs/experiment/VarExpr24/VarExpr24_GRPO.yaml`

- [ ] **Step 4.1: Write launch script**

```bash
#!/usr/bin/env bash
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
GPU=${1:-0}
OUTPUT_ROOT=./logs/rl_baselines

# === Training ===
echo "=== Training GRPO ==="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_grpo_expr24.py \
    --output_dir $OUTPUT_ROOT/grpo_expr24 \
    --max_steps 5000 \
    --seed 42

echo "=== Training PPO ==="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_ppo_expr24.py \
    --output_dir $OUTPUT_ROOT/ppo_expr24 \
    --max_steps 5000 \
    --seed 42

# === Evaluation ===
echo "=== Evaluating GRPO ==="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/eval_rl_baseline.py \
    --model_path $OUTPUT_ROOT/grpo_expr24/final \
    --exp_name VarExpr24_GRPO \
    --n_samples 6400 \
    --test_repeats 3 \
    --output_dir $OUTPUT_ROOT/eval_grpo

echo "=== Evaluating PPO ==="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/eval_rl_baseline.py \
    --model_path $OUTPUT_ROOT/ppo_expr24/final \
    --exp_name VarExpr24_PPO \
    --n_samples 6400 \
    --test_repeats 3 \
    --output_dir $OUTPUT_ROOT/eval_ppo
```

- [ ] **Step 4.2: Write Hydra experiment configs (for documentation/reference)**

这些 config 主要用于记录实验参数，使其与 GFlowNet 实验可比较。Eval 脚本不直接使用 Hydra，但 config 文件保持参数文档化。

- [ ] **Step 4.3: Commit**

---

### Task 5: Smoke Tests

**Files:**
- Create: `tests/test_rl_baselines.py`

- [ ] **Step 5.1: Test reward function consistency**

```python
def test_expr24_reward_consistency():
    """Verify PPO/GRPO reward matches GFlowNet's Expr24Validator."""
    from chemgfn.models.validators import Expr24Validator
    validator = Expr24Validator(scorer="hit24_dense")

    # Known expressions
    # "1+2+3*6" = 1+2+18 = 21 ≠ 24
    # "4*5+6-2" = 20+6-2 = 24 ✓

    # Tokenize and validate
    tokens = tokenizer.encode("4*5+6-2", add_special_tokens=False)
    # ... assert global_score == 1.0
```

- [ ] **Step 5.2: Test grammar constraint produces valid expressions**

```python
def test_grammar_produces_valid_expressions():
    """Verify grammar constraint produces syntactically valid expressions."""
    # Generate 10 samples with grammar
    # Assert all match regex: ^[0-9][+\-*/][0-9]([+\-*/][0-9])*$
```

- [ ] **Step 5.3: Test metrics computation produces expected keys**

```python
def test_metrics_keys():
    """Verify compute_all_metrics returns all required metric keys."""
    # Expected keys:
    required = {"accuracy", "diversity", "diversity_valid", "topk_diversity",
                "topk_novelty", "topk_performance", "text_diversity",
                "prefix_top1_auc", "length_distribution"}
```

- [ ] **Step 5.4: Commit**

---

## Risk & Open Questions

1. **TRL upgrade compatibility**: Upgrading from 0.9.4 to 0.25+ is a major jump. May need to pin specific version. Test transformers_cfg compatibility after upgrade.

2. **Grammar constraints in GRPOTrainer**: GRPOTrainer 的内部 generate 可能不直接支持 `logits_processor`。备选方案:
   - 通过 `generation_config` 传递
   - Override model 的 `generate` 方法
   - 自己实现 generation loop (不用 GRPOTrainer 的内置 generate)

3. **Prompt format**: GRPOTrainer 通常期望 chat-format prompts (`list[dict]`)，而 VarExpr24 用的是 raw text prompt。需要适配。

4. **Token-level vs string-level reward**: GRPOTrainer 的 reward function 接收 decoded strings。但 Expr24Validator 的 `__call__` 接收 token tensors。需要在 reward function 中处理 string → score 的转换。

5. **Effective training steps**: GRPOTrainer 用 `num_train_epochs` 而不是 `max_steps`。需要计算:
   - Dataset size = 1 (single prompt, repeated)
   - Steps = dataset_size / (batch_size * grad_accum * num_gpus) * num_epochs
   - 需要调整使得总 gradient steps ≈ 5000

6. **PPO in TRL v0.25+**: PPOTrainer 的 API 可能已经大改。需要查看最新文档。如果新版 PPOTrainer 不好用，备选方案是用 `OnlineDPOTrainer` 或手写 PPO loop。

---

## Execution Order

```
Task 0 (env setup) → Task 1 (GRPO) → Task 2 (PPO) → Task 3 (eval) → Task 4 (launch) → Task 5 (tests)
```

Task 1 和 Task 2 可以并行开发（独立脚本），但 Task 3 (eval) 依赖于了解 Task 1/2 的输出格式。
