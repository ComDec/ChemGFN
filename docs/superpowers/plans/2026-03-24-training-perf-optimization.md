# Training Performance Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce VarExpr24 full training wall-clock from ~10h to ~1-2h per experiment on a single H100.

**Architecture:** Two-phase optimization: (1) eliminate the O(batch × seq_len) sequential validator bottleneck via vectorized batch scoring, (2) decouple autoregressive generation from gradient computation via generate-then-re-evaluate pattern, reducing 10 sequential grad-enabled forward passes to 10 no-grad + 1 parallel with-grad.

**Tech Stack:** PyTorch, Lightning, Hydra, existing ChemGFN codebase

---

## Profiling Summary

**Measured: single H100, n_samples=32, max_len=9, isolated**

| Component | Time per step | % | Root cause |
|-----------|-------------|---|-----------|
| Autoregressive generation (10 forward passes w/ grad) | ~0.30s | 66% | 10 sequential model calls, each memory-bound (1.2B weights >> 32×20 tokens) |
| Reward: score_fast (frozen ref model forward) | ~0.08s | 18% | 1 full forward pass on frozen model |
| Reward: Expr24Validator (CPU loop) | ~0.04s | 9% | `for i in range(batch_size): for pos in range(seq_len):` → 32×9=288 serial Python ops |
| Loss + backward + logging + buffer | ~0.03s | 7% | Efficient |
| **Total** | **~0.45s** | **100%** | |

**Scaling problem:** When n_samples doubles (32→64→128), the validator time grows **linearly** (O(batch × seq_len)), and GPU forward passes grow **sub-linearly** (memory-bound). At n_samples=128, the validator likely dominates.

**GPU utilization at n_samples=32:** ~20% compute, ~5% memory bandwidth → model is much larger than data, each forward pass is a small fraction of GPU capability.

---

## Optimization Strategy (Ordered by Impact × Feasibility)

### Phase 1: Enable Large Batch (High Impact, Low Risk)

**Problem:** With n_samples=32, GPU is massively underutilized. Increasing to 128+ would improve throughput, but the Expr24Validator's O(batch × seq_len) loop becomes the bottleneck.

**Solution:** Vectorize the Expr24Validator inner loop using batch string decoding + parallel expression evaluation.

### Phase 2: Generate-then-Re-evaluate (High Impact, Medium Risk)

**Problem:** 10 sequential forward passes WITH gradient tracking. Each stores activation memory and runs autograd overhead.

**Solution:** Generate tokens WITHOUT gradient (10× no_grad forward passes → much faster, less memory). Then run ONE parallel forward pass WITH gradient on the full generated sequence to extract log_pf and log_pterm. This is the standard REINFORCE-style separation used in most LLM fine-tuning.

### Phase 3: Config Tuning (Free)

**Problem:** acc_grad=4 with n_samples=32 means 4 forward passes per optimizer step, but effective batch = 128. With n_samples=128 and acc_grad=1, we get the same effective batch in 1 forward pass.

**Solution:** CLI overrides: `model.training_mixed_config.n_samples=128 trainer.accumulate_grad_batches=1`

---

## Task 1: Vectorize Expr24Validator

**Files:**
- Modify: `chemgfn/models/validators.py` (lines 131-200)

**Current code** (O(batch × seq_len), sequential):
```python
for i in range(batch_size):        # 32 iterations
    for pos in range(stop_pos):    # up to 9 iterations
        prefix_expr = _decode_tokens_to_string(sentences[i, : pos + 1], tokenizer)
        is_valid, score, value = self._score_expression(prefix_expr)
```

**Target:** Batch decode all prefixes at once, evaluate in parallel using multiprocessing or vectorized ops.

- [ ] **Step 1: Profile the validator in isolation**

Add timing instrumentation to `Expr24Validator.__call__()` to measure exactly how long it takes for batch_size=32,64,128. Create a small script:

```python
# bench_validator.py (temporary, in project root)
import time, torch, rootutils
rootutils.setup_root(".", indicator=".project-root", pythonpath=True, dotenv=True, cwd=True)
from chemgfn.models.validators import Expr24Validator
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
val = Expr24Validator(scorer="hit24_dense", amortize_valid_state=False)

for bs in [32, 64, 128, 256]:
    # Fake batch of token ids (length 10)
    sentences = torch.randint(0, 1000, (bs, 10))
    eos = tok.eos_token_id
    sentences[:, -1] = eos

    t0 = time.time()
    for _ in range(10):
        result = val(sentences, tok, termination_token_id=eos, scaffold=None)
    avg = (time.time() - t0) / 10
    print(f"batch={bs:>3d}: {avg*1000:.1f}ms  ({avg*1000/bs:.1f}ms/sample)")
```

Run: `CUDA_VISIBLE_DEVICES="" python bench_validator.py`
Expected: Linear scaling with batch size, confirming the bottleneck.

- [ ] **Step 2: Implement batch prefix decoding**

Replace the inner loop with batch operations. The key insight: for each sample, all prefixes `sentences[i, :1], sentences[i, :2], ..., sentences[i, :stop_pos]` can be decoded together.

In `chemgfn/models/validators.py`, add a helper method to `Expr24Validator`:

```python
def _batch_score_prefixes(self, sentences, tokenizer, termination_token_id):
    """Vectorized prefix scoring for the entire batch.

    Instead of: for i in batch: for pos in seq:
    Do: decode all prefixes at once, evaluate in batch.
    """
    B, S = sentences.shape
    local_score = torch.zeros(B, S + 1)
    invalid = torch.ones(B, S + 1)
    invalid[:, 0] = 1.0
    global_score = torch.zeros(B)

    # Find stop positions (first EOS per sample) - vectorized
    eos_mask = sentences == termination_token_id
    has_eos = eos_mask.any(dim=1)
    stop_pos = torch.where(has_eos, eos_mask.float().argmax(dim=1), torch.full((B,), S, dtype=torch.long))

    # Collect all (sample_idx, position) pairs to evaluate
    eval_tasks = []
    for i in range(B):
        sp = int(stop_pos[i].item())
        for pos in range(sp):
            eval_tasks.append((i, pos, tokenizer.decode(sentences[i, :pos+1].tolist(), skip_special_tokens=True)))

    # Evaluate all expressions (this is still CPU but no redundant decoding)
    for i, pos, expr in eval_tasks:
        is_valid, score, value = self._score_expression(expr)
        is_hit = is_valid and value == self.target_value
        invalid[i, pos + 1] = 0.0 if is_hit else 1.0
        if self.scorer in {"hit24_dense", "near_24_dense"}:
            local_score[i, pos + 1] = float(score)

    # Final expression scoring
    full_tokens_list = []
    for i in range(B):
        final_expr = self._decode_expr(sentences[i], tokenizer)
        if final_expr is None:
            full_tokens_list.append("")
            continue
        is_valid, score, value = self._score_expression(final_expr)
        sp = int(stop_pos[i].item())
        last_pos = sp if sp < S else S
        if last_pos >= 1:
            local_score[i, last_pos] = float(score)
            invalid[i, last_pos] = 0.0 if (is_valid and value == self.target_value) else 1.0
        global_score[i] = float(score)
        invalid[i, -1] = 0.0 if (is_valid and value == self.target_value) else 1.0
        full_tokens_list.append(final_expr)

    return local_score, invalid, global_score, full_tokens_list
```

- [ ] **Step 3: Add multiprocessing for expression evaluation**

The `_score_expression` calls are pure CPU and independent. Use `concurrent.futures.ThreadPoolExecutor` (or ProcessPoolExecutor for GIL-bound work):

```python
from concurrent.futures import ThreadPoolExecutor

# Replace the eval_tasks loop with parallel execution:
with ThreadPoolExecutor(max_workers=min(16, len(eval_tasks))) as pool:
    results = list(pool.map(lambda t: (t[0], t[1], self._score_expression(t[2])), eval_tasks))
```

- [ ] **Step 4: Benchmark the optimized validator**

Re-run `bench_validator.py` and compare.
Expected: 2-4x speedup for batch_size=32, more for larger batches.

- [ ] **Step 5: Run full training benchmark**

Run: `CUDA_VISIBLE_DEVICES=0 python chemgfn/train.py experiment=VarExpr24/VarExpr24_AvgPrefixTB trainer.max_steps=20 trainer.limit_train_batches=20 trainer.devices=1`

Compare step time with baseline.

- [ ] **Step 6: Commit**

```bash
git add chemgfn/models/validators.py
git commit -m "perf: vectorize Expr24Validator prefix scoring with optional parallel eval"
```

---

## Task 2: Generate-then-Re-evaluate (Decouple Generation from Gradients)

**Files:**
- Modify: `chemgfn/utils/gfn_utils.py` (lines 261-365)

**Current:** 10 sequential forward passes WITH autograd → stores activations, slow.
**Target:** Generate tokens with `torch.no_grad()`, then 1 parallel forward pass WITH grad.

- [ ] **Step 1: Understand the gradient flow**

Current gradient path:
```
logits (line 264) → logprob = logits.log_softmax() (line 345)
    → log_pf (line 353-354): gather from logprob
    → log_pterm (line 347-348): index logprob at EOS token
→ loss_fn(log_pf, log_pterm, ...) → backward()
```

The loss only needs `log_pf` and `log_pterm`, which are derived from `logits.log_softmax()`. If we have the full generated sequence, we can get all logits in one parallel forward pass.

- [ ] **Step 2: Add a re-evaluation forward pass function**

Add to `gfn_utils.py`:

```python
def re_evaluate_trajectory(model, state, prompt_len, termination_token_id):
    """
    Given a generated sequence, compute log_pf and log_pterm with gradients.

    Args:
        model: The policy model (with LoRA)
        state: [B, prompt_len + gen_len] full token sequence
        prompt_len: int
        termination_token_id: int

    Returns:
        log_pf: [B, gen_len] log forward probs
        log_pterm: [B, gen_len] log termination probs
    """
    # Single forward pass on full sequence — parallel, not autoregressive
    output = model(input_ids=state)
    # logits: [B, seq_len, vocab_size]
    # We need logits at positions prompt_len-1 to prompt_len+gen_len-2 (shifted by 1)
    gen_logits = output.logits[:, prompt_len - 1:-1, :]  # [B, gen_len, vocab_size]
    logprob = gen_logits.log_softmax(dim=-1)

    # Extract log_pf: probability of chosen tokens
    gen_tokens = state[:, prompt_len:]  # [B, gen_len]
    log_pf = logprob.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)  # [B, gen_len]

    # Extract log_pterm: probability of EOS at each position
    log_pterm = logprob[:, :, termination_token_id]  # [B, gen_len]

    # Mask positions after termination
    active = (gen_tokens != termination_token_id).cumsum(dim=1) <= (gen_tokens != termination_token_id).sum(dim=1, keepdim=True)
    log_pf = log_pf * active.float()
    log_pterm = log_pterm * active.float()

    return log_pf, log_pterm
```

- [ ] **Step 3: Modify generation to run without gradients**

In `generate_and_return_termination_logprob`, wrap the generation loop:

```python
# BEFORE the loop (line ~260):
with torch.no_grad():
    for step in range(max_len + 1):
        # ... existing generation code (sampling, grammar, etc.)
        # But now WITHOUT gradient tracking → much faster, less memory

# AFTER the loop, re-evaluate WITH gradients:
log_pf_grad, log_pterm_grad = re_evaluate_trajectory(
    model, state, prompt_len, termination_token_id
)
```

Key considerations:
- The `log_softmax` in the loop (line 345) currently serves dual purpose: sampling AND gradient.
- With this change, the loop only handles sampling (no_grad), and re-evaluation handles gradients.
- The grammar processor, token rejection, buffer mixing all happen during no_grad generation.
- KV cache is still used during generation (speeds up no_grad passes further).

- [ ] **Step 4: Handle tensor shape alignment**

The existing code returns `log_pf: [B, L]` and `log_pterm: [B, L]`. The re-evaluation must produce tensors in the same shape. Verify by comparing:

```python
# In a test:
log_pf_old = ...  # from current code
log_pf_new = ...  # from re-evaluate_trajectory
assert torch.allclose(log_pf_old, log_pf_new, atol=1e-4)
```

Note: There may be small numerical differences due to float precision, but the values should match closely.

- [ ] **Step 5: Benchmark the decoupled version**

Compare step time and memory usage:
```
Before: 10 sequential forward (with grad) + 1 ref forward
After:  10 sequential forward (no grad) + 1 parallel forward (with grad) + 1 ref forward
```

Expected savings:
- No-grad forward passes are ~30-40% faster (no activation storage, no autograd overhead)
- Single parallel forward pass is much faster than 10 sequential
- Memory reduction allows larger n_samples

- [ ] **Step 6: Commit**

```bash
git add chemgfn/utils/gfn_utils.py
git commit -m "perf: decouple generation from gradient computation (generate-then-re-evaluate)"
```

---

## Task 3: Config Tuning for Large Batch

**Files:**
- No code changes needed — use CLI overrides

- [ ] **Step 1: Determine optimal n_samples**

After Tasks 1-2, benchmark with different n_samples:

```bash
for N in 64 128 256; do
  CUDA_VISIBLE_DEVICES=0 python chemgfn/train.py \
    experiment=VarExpr24/VarExpr24_AvgPrefixTB \
    model.training_mixed_config.n_samples=$N \
    trainer.accumulate_grad_batches=1 \
    trainer.max_steps=20 trainer.limit_train_batches=20 \
    trainer.devices=1 seed=42
done
```

Find the sweet spot where GPU utilization is maximized without OOM.

- [ ] **Step 2: Create experiment configs with optimal settings**

Create configs with the optimal n_samples and proportionally reduced steps. For example, if n_samples=128 works best:

```yaml
# In experiment config, override:
model:
  training_mixed_config:
    n_samples: 128
trainer:
  accumulate_grad_batches: 1
  max_steps: 5000  # same optim steps, 4x fewer forward passes
```

---

## Expected Outcomes

| Optimization | Speedup | Risk | Effort |
|---|---|---|---|
| Task 1: Vectorize Expr24Validator | 1.5-3x for large batch | Low | 2-3h |
| Task 2: Generate-then-re-evaluate | 1.5-2x | Medium (needs careful shape alignment) | 3-4h |
| Task 3: n_samples=128, acc_grad=1 | 2-4x (from fewer total forward passes) | None (same effective batch) | 10min |
| **Combined** | **5-10x** | | |

**Projected full training time:** Current ~10h → Target ~1-2h per experiment.
