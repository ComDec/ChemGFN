#!/usr/bin/env python
"""Evaluate a trained LoRA (or base) model on VarExpr24 with grammar-constrained generation.

Loads the model, generates N samples in batches, computes all metrics matching the
GFlowNet evaluation pipeline, and logs to wandb + saves JSON/CSV.

Usage:
    python scripts/eval_rl_baseline.py \
        --model_path ./logs/grpo_expr24/final \
        --base_model meta-llama/Llama-3.2-1B \
        --exp_name VarExpr24_GRPO \
        --n_samples 6400 \
        --batch_size 32 \
        --output_dir ./logs/eval_grpo \
        --seed 42 \
        --test_repeats 3 \
        --wandb_project ChemGFN
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList
from transformers_cfg.generation.logits_process import (
    GrammarIncrementalLogitsProcessorGeneral,
)
from transformers_cfg.parser import parse_ebnf

# ---------------------------------------------------------------------------
# Ensure project root on sys.path so chemgfn is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from chemgfn.models.validators import Expr24Validator
from chemgfn.utils.diversity import SequenceDiversity
from chemgfn.utils.gfn_utils import calculate_diversity_by_length, prepare_token_mask
from chemgfn.utils.prefix_metrics import prefix_collapse_by_position
from chemgfn.utils.sequence_metrics import (
    levenshtein_diversity,
    levenshtein_novelty,
    select_topk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_lora_adapter(path: str) -> bool:
    """Return True if *path* looks like a PEFT LoRA adapter directory."""
    p = Path(path)
    return p.is_dir() and (p / "adapter_config.json").exists()


def load_model(base_model: str, model_path: str, device: str = "cuda"):
    """Load base Llama + optional LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map=device,
    )

    if model_path and _is_lora_adapter(model_path):
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, model_path)
        print(f"[INFO] Loaded LoRA adapter from {model_path}")
    else:
        model = base
        if model_path and model_path != base_model:
            # Might be a full fine-tuned checkpoint directory
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    device_map=device,
                )
                print(f"[INFO] Loaded full model from {model_path}")
            except Exception:
                print(f"[WARN] Could not load {model_path} as full model; using base model.")
        else:
            print(f"[INFO] Using base model {base_model} directly (no adapter).")

    model.eval()
    return model, tokenizer


class WrappedGrammarProcessor:
    """Wraps transformers_cfg grammar processor to return a plain tensor."""

    def __init__(self, processor):
        self.processor = processor

    def init_for_prompt(self, prompt_len: int):
        """Reset state and set prompt length before each generation batch."""
        self.processor.reset()
        self.processor.set_prompt_length(prompt_len)

    def __call__(self, input_ids, scores):
        result = self.processor(input_ids, scores)
        if isinstance(result, dict):
            return result["masked_logits"]
        return result

    def reset(self):
        if hasattr(self.processor, "reset"):
            self.processor.reset()


def build_grammar_processor(
    grammar_path: str, legal_tokens_path: str, tokenizer, device: str = "cuda"
):
    """Build the same grammar processor used in GFlowNet VarExpr24 experiments."""
    grammar_str = Path(grammar_path).read_text()
    try:
        parsed_grammar = parse_ebnf(grammar_str)
    except Exception as e:
        print(f"[WARN] Grammar parsing failed ({e}); grammar constraints disabled.")
        return None

    with open(legal_tokens_path) as f:
        legal_tokens = [line.strip() for line in f if line.strip()]
    legal_token_ids = []
    for tok in legal_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            print(f"[WARN] Token '{tok}' -> {ids}, skipping")
            continue
        legal_token_ids.append(ids[0])
    legal_token_ids.append(tokenizer.eos_token_id)

    raw_processor = GrammarIncrementalLogitsProcessorGeneral(
        parsed_grammar,
        tokenizer=tokenizer,
        nice_token_ids_list=legal_token_ids,
        execution_mode="limited",
    )
    return WrappedGrammarProcessor(raw_processor)


def calculate_diversity_ragged(token_ids_list: list[list[int]], eos_id: int) -> float:
    """Token-level diversity: average entropy per position across all sequences."""
    if len(token_ids_list) <= 1:
        return 0.0
    max_len = max(len(seq) for seq in token_ids_list)
    entropies = []
    for t in range(max_len):
        tokens_at_t = []
        for seq in token_ids_list:
            if t < len(seq) and seq[t] != eos_id:
                tokens_at_t.append(seq[t])
        if len(tokens_at_t) < 2:
            continue
        counts: dict[int, int] = {}
        for tok in tokens_at_t:
            counts[tok] = counts.get(tok, 0) + 1
        n = len(tokens_at_t)
        entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
        entropies.append(entropy)
    return sum(entropies) / len(entropies) if entropies else 0.0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    grammar_processor,
    legal_token_mask: torch.Tensor | None,
    illegal_token_mask: torch.Tensor | None,
    batch_size: int,
    max_len: int,
    temperature: float,
    device: str,
) -> torch.Tensor:
    """Generate a single batch of sequences with grammar constraints.

    Returns tensor of shape [batch_size, max_len] containing only the generated
    tokens (prompt stripped), padded with eos_token_id.
    """
    eos_id = tokenizer.eos_token_id
    prompt_len = prompt_ids.shape[1]

    # Expand prompt for batch
    state = prompt_ids.expand(batch_size, -1).clone().to(device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Initialize grammar processor for this prompt length
    if grammar_processor is not None and hasattr(grammar_processor, "init_for_prompt"):
        grammar_processor.init_for_prompt(prompt_len)

    generated_tokens: list[torch.Tensor] = []

    for step in range(max_len + 1):
        outputs = model(input_ids=state)
        logits = outputs.logits[:, -1, :]

        scores = logits.clone()

        # Apply legal token mask
        if illegal_token_mask is not None:
            scores[:, illegal_token_mask] = -torch.inf

        # Apply grammar constraints
        if grammar_processor is not None:
            scores = grammar_processor(state, scores)

        # Force EOS at last step
        if step >= max_len:
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[:, eos_id] = False
            scores[mask] = -torch.inf
            scores[:, eos_id] = 0.0

        # Inactive sequences always pick EOS
        if (~active).any():
            scores[~active] = -torch.inf
            scores[~active, eos_id] = 0.0

        # Handle rows with no valid tokens
        no_valid = ~torch.isfinite(scores).any(dim=-1)
        if no_valid.any():
            scores[no_valid] = -torch.inf
            scores[no_valid, eos_id] = 0.0

        probs = (scores / temperature).softmax(dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)  # [B, 1]

        # Force inactive to EOS
        next_tokens = torch.where(
            active.unsqueeze(-1),
            next_tokens,
            torch.full_like(next_tokens, eos_id),
        )

        active = active & (next_tokens.squeeze(-1) != eos_id)
        generated_tokens.append(next_tokens.squeeze(-1))
        state = torch.cat([state, next_tokens], dim=-1)

    gen = torch.stack(generated_tokens, dim=1)  # [B, max_len+1]
    return gen


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    all_gen_tokens: torch.Tensor,
    tokenizer,
    validator: Expr24Validator,
    seq_div: SequenceDiversity | None,
    buffer_path: str | None,
    topk_k: int = 100,
) -> dict:
    """Compute all evaluation metrics from generated token tensor.

    Args:
        all_gen_tokens: [N, seq_len] tensor of generated tokens (prompt stripped).
        tokenizer: the tokenizer.
        validator: Expr24Validator instance.
        seq_div: SequenceDiversity instance (or None to skip).
        buffer_path: path to training buffer .pt for novelty computation.
        topk_k: number of top-k sequences to select.

    Returns:
        Dict of metric_name -> value.
    """
    eos_id = tokenizer.eos_token_id
    N = all_gen_tokens.shape[0]
    device = all_gen_tokens.device

    # --- A. Accuracy via validator ---
    result = validator(all_gen_tokens, tokenizer)
    global_scores = result["global_score"]  # [N]
    full_tokens = result["full_tokens"]  # list[str]

    accuracy = global_scores.mean().item()
    valid_mask = global_scores > 0
    valid_rate = valid_mask.float().mean().item()
    n_valid = int(valid_mask.sum().item())

    # Decode all sequences to text
    decoded_texts = []
    for i in range(N):
        seq = all_gen_tokens[i].tolist()
        text = tokenizer.decode(
            [t for t in seq if t != eos_id],
            skip_special_tokens=True,
        )
        decoded_texts.append(text.strip())

    # Token ID lists (trimmed at EOS) for diversity computation
    token_ids_list: list[list[int]] = []
    for i in range(N):
        seq = all_gen_tokens[i].tolist()
        trimmed = []
        for t in seq:
            if t == eos_id:
                break
            trimmed.append(t)
        token_ids_list.append(trimmed)

    scores_list = global_scores.tolist()

    # --- B. Token-level diversity (ragged) ---
    diversity_all = calculate_diversity_ragged(token_ids_list, eos_id)

    valid_token_ids = [token_ids_list[i] for i in range(N) if valid_mask[i]]
    diversity_valid = (
        calculate_diversity_ragged(valid_token_ids, eos_id) if valid_token_ids else 0.0
    )

    # --- C. Diversity by length ---
    div_by_len = calculate_diversity_by_length(all_gen_tokens, eos_id)

    # --- D. Text embedding diversity ---
    text_diversity = None
    text_diversity_valid = None
    if seq_div is not None:
        try:
            text_diversity = seq_div(decoded_texts)
        except Exception as e:
            print(f"[WARN] text diversity failed: {e}")
            text_diversity = None
        valid_texts_for_div = [decoded_texts[i] for i in range(N) if valid_mask[i]]
        if valid_texts_for_div:
            try:
                text_diversity_valid = seq_div(valid_texts_for_div)
            except Exception as e:
                print(f"[WARN] text diversity (valid) failed: {e}")
                text_diversity_valid = None

    # --- E. Top-K Levenshtein metrics ---
    valid_texts = [decoded_texts[i] for i in range(N) if scores_list[i] > 0]
    valid_scores = [scores_list[i] for i in range(N) if scores_list[i] > 0]

    topk_seqs, topk_scores = select_topk(valid_texts, valid_scores, k=topk_k)
    topk_diversity = levenshtein_diversity(topk_seqs) if len(topk_seqs) >= 2 else 0.0
    topk_performance = sum(topk_scores) / len(topk_scores) if topk_scores else 0.0

    # Novelty against training buffer
    topk_novelty = 0.0
    if buffer_path and Path(buffer_path).exists() and topk_seqs:
        try:
            buffer = torch.load(buffer_path, map_location="cpu", weights_only=False)
            if isinstance(buffer, dict):
                # Handle dict-format buffers
                buffer = buffer.get("samples", buffer.get("data", list(buffer.values())[0]))
            if isinstance(buffer, torch.Tensor):
                training_set = []
                for row in buffer:
                    text = tokenizer.decode(
                        [t.item() for t in row if t.item() != eos_id],
                        skip_special_tokens=True,
                    ).strip()
                    if text:
                        training_set.append(text)
            elif isinstance(buffer, list):
                training_set = [str(s) for s in buffer]
            else:
                training_set = []

            if training_set:
                topk_novelty = levenshtein_novelty(topk_seqs, training_set)
        except Exception as e:
            print(f"[WARN] Could not load buffer for novelty: {e}")

    # --- F. Prefix collapse ---
    # Build sequences as list-of-lists and active_before masks
    seqs_for_prefix: list[list[int]] = []
    active_before_list: list[list[bool]] = []
    invalid_flags: list[bool] = []
    for i in range(N):
        seq = all_gen_tokens[i].tolist()
        seqs_for_prefix.append(seq)
        active = []
        seen_eos = False
        for t in seq:
            if seen_eos:
                active.append(False)
            else:
                active.append(True)
                if t == eos_id:
                    seen_eos = True
        active_before_list.append(active)
        invalid_flags.append(not valid_mask[i].item())

    prefix_stats = prefix_collapse_by_position(
        seqs_for_prefix,
        active_before=active_before_list,
        invalid=invalid_flags,
    )

    # --- G. Length distribution ---
    length_counts: dict[int, int] = {}
    score_sum_by_len: dict[int, float] = {}
    score_count_by_len: dict[int, int] = {}
    for i in range(N):
        length = len(token_ids_list[i])
        length_counts[length] = length_counts.get(length, 0) + 1
        score_sum_by_len[length] = score_sum_by_len.get(length, 0) + scores_list[i]
        score_count_by_len[length] = score_count_by_len.get(length, 0) + 1

    score_by_length = {k: score_sum_by_len[k] / score_count_by_len[k] for k in score_sum_by_len}

    # --- Assemble results ---
    metrics = {
        "accuracy": accuracy,
        "valid_rate": valid_rate,
        "n_samples": N,
        "n_valid": n_valid,
        "diversity": diversity_all,
        "diversity_valid": diversity_valid,
        "text_diversity": text_diversity,
        "text_diversity_valid": text_diversity_valid,
        "topk_performance": topk_performance,
        "topk_diversity": topk_diversity,
        "topk_novelty": topk_novelty,
        "prefix_top1_auc": prefix_stats.top1_auc,
        "prefix_collapse_depth": prefix_stats.collapse_depth,
        "prefix_top1_auc_correct": prefix_stats.top1_auc_correct,
        "prefix_collapse_depth_correct": prefix_stats.collapse_depth_correct,
        "length_distribution": {str(k): v for k, v in sorted(length_counts.items())},
        "score_by_length": {str(k): round(v, 4) for k, v in sorted(score_by_length.items())},
        "diversity_by_length": {str(k): round(v, 4) for k, v in sorted(div_by_len.items())},
    }

    # Per-sample data for CSV
    per_sample = []
    for i in range(N):
        per_sample.append(
            {
                "expression": decoded_texts[i],
                "score": scores_list[i],
                "length": len(token_ids_list[i]),
                "valid": bool(valid_mask[i].item()),
            }
        )

    return metrics, per_sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_single_eval(
    model,
    tokenizer,
    grammar_processor,
    legal_token_mask,
    illegal_token_mask,
    validator,
    seq_div,
    args,
    seed: int,
) -> tuple[dict, list[dict]]:
    """Run a single round of generation + evaluation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = args.device
    eos_id = tokenizer.eos_token_id

    # Tokenize prompt
    prompt_text = Path(args.prompt_path).read_text().strip()
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    n_batches = math.ceil(args.n_samples / args.batch_size)
    all_gen: list[torch.Tensor] = []

    for batch_idx in tqdm(range(n_batches), desc=f"Generating (seed={seed})"):
        current_bs = min(args.batch_size, args.n_samples - batch_idx * args.batch_size)
        if current_bs <= 0:
            break

        gen = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            grammar_processor=grammar_processor,
            legal_token_mask=legal_token_mask,
            illegal_token_mask=illegal_token_mask,
            batch_size=current_bs,
            max_len=args.max_len,
            temperature=args.temperature,
            device=device,
        )
        all_gen.append(gen.cpu())

    all_gen_tokens = torch.cat(all_gen, dim=0)[: args.n_samples]

    metrics, per_sample = compute_metrics(
        all_gen_tokens=all_gen_tokens,
        tokenizer=tokenizer,
        validator=validator,
        seq_div=seq_div,
        buffer_path=args.buffer_path,
        topk_k=args.topk_k,
    )

    return metrics, per_sample


def aggregate_repeats(all_metrics: list[dict]) -> dict:
    """Compute mean +/- std over multiple repeats for scalar metrics."""
    scalar_keys = [
        k
        for k in all_metrics[0]
        if isinstance(all_metrics[0][k], (int, float)) and all_metrics[0][k] is not None
    ]
    agg = {}
    for k in scalar_keys:
        vals = [m[k] for m in all_metrics if m[k] is not None]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        else:
            agg[f"{k}_mean"] = None
            agg[f"{k}_std"] = None
    # Also keep non-scalar from last repeat
    for k in all_metrics[-1]:
        if k not in scalar_keys:
            agg[k] = all_metrics[-1][k]
    return agg


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL baseline on VarExpr24")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to LoRA adapter or model"
    )
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--exp_name", type=str, default="VarExpr24_RL_baseline")
    parser.add_argument("--n_samples", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./logs/eval_rl_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_repeats", type=int, default=1)
    parser.add_argument("--topk_k", type=int, default=100)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--grammar_path",
        type=str,
        default=str(_PROJECT_ROOT / "assets" / "24_grammars" / "var_length.ebnf"),
    )
    parser.add_argument(
        "--legal_tokens_path",
        type=str,
        default=str(_PROJECT_ROOT / "assets" / "token_list" / "24_points" / "general"),
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "24_points" / "prompts.txt"),
    )
    parser.add_argument(
        "--buffer_path",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "24_points" / "buffer_24_non_zero.pt"),
    )
    parser.add_argument("--scorer", type=str, default="hit24_dense")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--skip_text_diversity", action="store_true", help="Skip embedding diversity"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    print(f"Loading model: base={args.base_model}, adapter={args.model_path}")
    model, tokenizer = load_model(args.base_model, args.model_path, device=args.device)

    # --- Build grammar processor ---
    print(f"Loading grammar from {args.grammar_path}")
    grammar_processor = build_grammar_processor(
        args.grammar_path, args.legal_tokens_path, tokenizer, device=args.device
    )

    # --- Build legal token mask ---
    legal_token_mask, illegal_token_mask, _ = prepare_token_mask(tokenizer, args.legal_tokens_path)
    legal_token_mask = legal_token_mask.to(args.device)
    illegal_token_mask = illegal_token_mask.to(args.device)

    # --- Build validator ---
    validator = Expr24Validator(scorer=args.scorer)

    # --- Build text diversity ---
    seq_div = None
    if not args.skip_text_diversity:
        try:
            seq_div = SequenceDiversity("sequence_embedding")
        except Exception as e:
            print(f"[WARN] Could not load SequenceDiversity: {e}")

    # --- Wandb ---
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.exp_name,
                config=vars(args),
            )
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")

    # --- Run evaluations ---
    all_metrics: list[dict] = []
    all_per_sample: list[list[dict]] = []

    for repeat_idx in range(args.test_repeats):
        seed = args.seed + repeat_idx
        print(f"\n{'='*60}")
        print(f"Repeat {repeat_idx + 1}/{args.test_repeats} (seed={seed})")
        print(f"{'='*60}")

        t0 = time.time()
        metrics, per_sample = run_single_eval(
            model=model,
            tokenizer=tokenizer,
            grammar_processor=grammar_processor,
            legal_token_mask=legal_token_mask,
            illegal_token_mask=illegal_token_mask,
            validator=validator,
            seq_div=seq_div,
            args=args,
            seed=seed,
        )
        elapsed = time.time() - t0
        metrics["wall_time_seconds"] = elapsed
        metrics["seed"] = seed
        metrics["repeat_idx"] = repeat_idx

        all_metrics.append(metrics)
        all_per_sample.append(per_sample)

        # Print summary for this repeat
        print(f"\n--- Repeat {repeat_idx + 1} Results ---")
        for k in [
            "accuracy",
            "valid_rate",
            "n_valid",
            "diversity",
            "diversity_valid",
            "text_diversity",
            "text_diversity_valid",
            "topk_performance",
            "topk_diversity",
            "topk_novelty",
            "prefix_top1_auc",
            "prefix_collapse_depth",
        ]:
            print(f"  {k}: {metrics.get(k)}")
        print(f"  wall_time: {elapsed:.1f}s")

        # Log to wandb
        if wandb_run is not None:
            log_dict = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v is not None:
                    log_dict[f"test/{k}"] = v
            wandb_run.log(log_dict, step=repeat_idx)

        # Save per-repeat CSV
        csv_path = Path(args.output_dir) / f"samples_repeat{repeat_idx}.csv"
        with open(csv_path, "w") as f:
            f.write("expression,score,length,valid\n")
            for row in per_sample:
                expr = row["expression"].replace('"', '""')
                f.write(f'"{expr}",{row["score"]},{row["length"]},{row["valid"]}\n')
        print(f"  Saved per-sample CSV: {csv_path}")

    # --- Aggregate results ---
    if args.test_repeats > 1:
        agg = aggregate_repeats(all_metrics)
        print(f"\n{'='*60}")
        print(f"Aggregated Results ({args.test_repeats} repeats)")
        print(f"{'='*60}")
        for k in [
            "accuracy",
            "valid_rate",
            "n_valid",
            "diversity",
            "diversity_valid",
            "text_diversity",
            "text_diversity_valid",
            "topk_performance",
            "topk_diversity",
            "topk_novelty",
            "prefix_top1_auc",
            "prefix_collapse_depth",
        ]:
            mean_val = agg.get(f"{k}_mean")
            std_val = agg.get(f"{k}_std")
            if mean_val is not None:
                print(f"  {k}: {mean_val:.4f} +/- {std_val:.4f}")
        final_output = {
            "exp_name": args.exp_name,
            "test_repeats": args.test_repeats,
            "aggregated": agg,
            "per_repeat": all_metrics,
        }
    else:
        final_output = {
            "exp_name": args.exp_name,
            **all_metrics[0],
        }

    # Save JSON
    json_path = Path(args.output_dir) / "eval_results.json"

    def _json_default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return str(obj)

    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=2, default=_json_default)
    print(f"\nSaved results JSON: {json_path}")

    # Final wandb summary
    if wandb_run is not None:
        if args.test_repeats > 1:
            for k, v in agg.items():
                if isinstance(v, (int, float)) and v is not None:
                    wandb_run.summary[f"test/{k}"] = v
        else:
            for k, v in all_metrics[0].items():
                if isinstance(v, (int, float)) and v is not None:
                    wandb_run.summary[f"test/{k}"] = v
        wandb_run.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
