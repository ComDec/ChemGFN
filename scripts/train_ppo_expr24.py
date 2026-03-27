#!/usr/bin/env python
"""PPO training script for the VarExpr24 task.

Trains a LoRA-adapted Llama-3.2-1B to generate arithmetic expressions that
evaluate to 24, using a manual PPO loop with grammar-constrained decoding.

Grammar constraint uses transformers_cfg (same as the GFlowNet experiments).
"""

import argparse
import os
import sys
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.generation.logits_process import (
    GrammarIncrementalLogitsProcessorGeneral,
)
from transformers_cfg.parser import parse_ebnf

# ---------------------------------------------------------------------------
# Add project root to path so we can import from chemgfn
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chemgfn.models.validators import Expr24Validator  # noqa: E402


# ---------------------------------------------------------------------------
# Grammar processor setup using transformers_cfg (matching GFlowNet config)
# ---------------------------------------------------------------------------
class WrappedGrammarProcessor:
    """Wraps transformers_cfg grammar processor to return a plain tensor.

    GrammarIncrementalLogitsProcessorGeneral returns a dict
    {'masked_logits': Tensor, 'acceptance': Tensor}. This wrapper extracts
    'masked_logits' so it can be used in manual generation loops.

    IMPORTANT: Must call init_for_prompt(prompt_len) before each generation batch.
    """

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


def build_grammar_processor(tokenizer, grammar_path, legal_tokens_path):
    """Build the same grammar processor used in GFlowNet VarExpr24 experiments."""
    with open(grammar_path) as f:
        grammar_str = f.read()
    parsed_grammar = parse_ebnf(grammar_str)

    with open(legal_tokens_path) as f:
        legal_tokens = [line.strip() for line in f if line.strip()]
    legal_token_ids = []
    for tok in legal_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        assert len(ids) == 1, f"Token '{tok}' -> {ids}, expected single token"
        legal_token_ids.append(ids[0])
    legal_token_ids.append(tokenizer.eos_token_id)

    raw_processor = GrammarIncrementalLogitsProcessorGeneral(
        parsed_grammar,
        tokenizer=tokenizer,
        nice_token_ids_list=legal_token_ids,
        execution_mode="limited",
    )
    return WrappedGrammarProcessor(raw_processor)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for VarExpr24")
    parser.add_argument("--output_dir", type=str, default="./logs/ppo_expr24")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_coeff", type=float, default=0.05)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="ChemGFN")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min_new_tokens", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=9)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    return parser.parse_args()


def compute_rewards(generated_ids: torch.Tensor, tokenizer, validator: Expr24Validator):
    """Compute per-sample scalar rewards using Expr24Validator.

    Returns:
        rewards: [B] reward for each sample
        hit24: [B] 1.0 if expression == 24, else 0.0
        expressions: list of decoded expression strings
    """
    batch_size = generated_ids.shape[0]
    rewards = torch.zeros(batch_size, device=generated_ids.device)
    hit24 = torch.zeros(batch_size, device=generated_ids.device)
    expressions = []

    for i in range(batch_size):
        tokens = generated_ids[i]
        pieces = []
        for t in tokens:
            tid = t.item()
            if tid == tokenizer.eos_token_id:
                break
            pieces.append(tokenizer.decode(t, skip_special_tokens=False))
        expr_str = "".join("".join(pieces).split())
        expressions.append(expr_str)

        if not expr_str:
            continue

        is_valid, score, value = validator._score_expression(expr_str)
        rewards[i] = score
        if is_valid and value is not None and value == Fraction(24):
            hit24[i] = 1.0

    return rewards, hit24, expressions


def build_token_mask(generated: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Build mask: 1.0 for real tokens up to and including first EOS, 0.0 after."""
    is_eos = generated == eos_token_id
    cumsum_eos = is_eos.cumsum(dim=1)
    return (cumsum_eos <= 1).float()


def autoregressive_generate(
    model,
    prompt_ids: torch.Tensor,
    grammar_processor,
    n_samples: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    eos_token_id: int,
):
    """Manual autoregressive generation with grammar constraints.

    Uses transformers_cfg's grammar processor which reads grammar state from
    the full input_ids sequence. This matches the GFlowNet generation pipeline.

    Returns:
        all_ids: [B, prompt_len + gen_len] full sequences
        gen_ids: [B, gen_len] generated part only
        gen_log_probs: [B, gen_len] log-prob of each generated token under the policy
    """
    device = prompt_ids.device
    batch_prompt = prompt_ids.expand(n_samples, -1)  # [B, P]
    prompt_len = batch_prompt.shape[1]

    # Initialize grammar processor for this prompt
    grammar_processor.init_for_prompt(prompt_len)

    # Working sequence starts as just the prompt
    current_ids = batch_prompt.clone()  # [B, P]
    gen_token_ids = []
    gen_token_lps = []
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)

    for t in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(current_ids)
            logits = outputs.logits[:, -1, :]  # [B, V]

        # Apply grammar constraints (transformers_cfg reads state from input_ids)
        constrained_logits = grammar_processor(current_ids, logits.clone())

        # Temperature
        if temperature != 1.0:
            constrained_logits = constrained_logits / temperature

        # Handle edge case: all logits are -inf (should not happen with valid grammar)
        all_neg_inf = (constrained_logits == float("-inf")).all(dim=-1)
        if all_neg_inf.any():
            constrained_logits[all_neg_inf, eos_token_id] = 0.0

        # Clamp logits to avoid NaN/Inf in softmax (policy may collapse)
        constrained_logits = constrained_logits.clamp(min=-1e4, max=1e4)
        constrained_logits = torch.nan_to_num(constrained_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        # Sample
        log_probs = F.log_softmax(constrained_logits, dim=-1)
        probs = log_probs.exp().clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # re-normalize
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

        # For finished sequences, force EOS
        next_tokens = torch.where(
            finished, torch.full_like(next_tokens, eos_token_id), next_tokens
        )

        # Record
        token_lps = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)  # [B]
        gen_token_ids.append(next_tokens)
        gen_token_lps.append(token_lps)

        # Update finished
        finished = finished | (next_tokens == eos_token_id)

        # Append to sequence
        current_ids = torch.cat([current_ids, next_tokens.unsqueeze(1)], dim=1)

        # Early stop if all finished and min length reached
        if finished.all() and (t + 1) >= min_new_tokens:
            break

    gen_ids = torch.stack(gen_token_ids, dim=1)  # [B, T]
    gen_log_probs = torch.stack(gen_token_lps, dim=1)  # [B, T]
    all_ids = current_ids  # [B, P + T]

    return all_ids, gen_ids, gen_log_probs


def compute_log_probs_for_tokens(model, full_ids, generated, prompt_len):
    """Compute log-probs of generated tokens under model.

    Returns:
        token_log_probs: [B, T]
        all_log_probs: [B, T, V] (for entropy computation)
    """
    logits = model(full_ids).logits  # [B, P + T, V]
    gen_len = generated.shape[1]
    relevant_logits = logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]  # [B, T, V]
    all_log_probs = F.log_softmax(relevant_logits, dim=-1)  # [B, T, V]
    token_log_probs = all_log_probs.gather(2, generated.unsqueeze(-1)).squeeze(-1)  # [B, T]
    return token_log_probs, all_log_probs


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------
    # wandb
    # -----------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name="ppo_expr24", config=vars(args))

    # -----------------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------
    # Prompt
    # -----------------------------------------------------------------------
    prompt_path = PROJECT_ROOT / "data" / "24_points" / "prompts.txt"
    prompt_text = prompt_path.read_text().strip()
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=True).to(
        device
    )
    prompt_len = prompt_ids.shape[1]
    print(f"Prompt length: {prompt_len} tokens")

    # -----------------------------------------------------------------------
    # Grammar processor (transformers_cfg, same as GFlowNet)
    # -----------------------------------------------------------------------
    grammar_path = PROJECT_ROOT / "assets" / "24_grammars" / "var_length.ebnf"
    legal_tokens_path = PROJECT_ROOT / "assets" / "token_list" / "24_points" / "general"
    grammar_processor = build_grammar_processor(tokenizer, grammar_path, legal_tokens_path)
    print(f"Grammar constraint enabled (transformers_cfg, {grammar_path.name})")

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------
    print(f"Loading base model: {args.model_name}")
    model_kwargs = dict(dtype=torch.bfloat16)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, attn_implementation="flash_attention_2", **model_kwargs
        )
    except Exception:
        print("Flash attention not available, using default attention")
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # LoRA config matching GFlowNet experiments
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # Frozen reference model
    print("Loading reference model...")
    try:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, attn_implementation="flash_attention_2", **model_kwargs
        )
    except Exception:
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # -----------------------------------------------------------------------
    # Validator & Optimizer
    # -----------------------------------------------------------------------
    validator = Expr24Validator(scorer="hit24", target_value=24)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(
        f"\nStarting PPO training for {args.max_steps} steps, "
        f"{args.n_samples} samples/step, {args.ppo_epochs} PPO epochs/step"
    )

    model.train()

    for step in range(args.max_steps):
        # -------------------------------------------------------------------
        # 1. Generate samples with grammar constraints
        # -------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            all_ids, gen_ids, old_gen_log_probs = autoregressive_generate(
                model=model,
                prompt_ids=prompt_ids,
                grammar_processor=grammar_processor,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                eos_token_id=tokenizer.eos_token_id,
            )
        model.train()

        gen_len = gen_ids.shape[1]
        if gen_len == 0:
            print(f"Step {step}: no tokens generated, skipping")
            continue

        # -------------------------------------------------------------------
        # 2. Compute rewards
        # -------------------------------------------------------------------
        rewards, hit24, expressions = compute_rewards(gen_ids, tokenizer, validator)

        # Token mask
        token_mask = build_token_mask(gen_ids, tokenizer.eos_token_id)

        # -------------------------------------------------------------------
        # 3. Reference log-probs (no grad)
        # -------------------------------------------------------------------
        with torch.no_grad():
            ref_token_log_probs, _ = compute_log_probs_for_tokens(
                ref_model, all_ids, gen_ids, prompt_len
            )

        # -------------------------------------------------------------------
        # 4. Old policy log-probs (detached, from generation)
        # -------------------------------------------------------------------
        old_token_log_probs = old_gen_log_probs.detach()

        # -------------------------------------------------------------------
        # 5. Advantage (reward whitening, broadcast to token level)
        # -------------------------------------------------------------------
        with torch.no_grad():
            if rewards.std() > 1e-8:
                advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                advantage = rewards - rewards.mean()
            advantage_tokens = advantage.unsqueeze(1).expand(-1, gen_len) * token_mask

        # -------------------------------------------------------------------
        # 6. PPO optimization epochs
        # -------------------------------------------------------------------
        accum_pg_loss = 0.0
        accum_kl = 0.0
        accum_entropy = 0.0
        accum_loss = 0.0

        optimizer.zero_grad()

        for ppo_epoch in range(args.ppo_epochs):
            # Recompute log probs with gradient
            new_token_log_probs, new_all_log_probs = compute_log_probs_for_tokens(
                model, all_ids, gen_ids, prompt_len
            )

            # Ratio
            ratio = (new_token_log_probs - old_token_log_probs).exp()

            # Clipped surrogate objective
            surr1 = ratio * advantage_tokens
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantage_tokens
            pg_loss = -torch.min(surr1, surr2)
            pg_loss = (pg_loss * token_mask).sum() / token_mask.sum().clamp(min=1.0)

            # KL divergence from reference
            kl_per_token = new_token_log_probs - ref_token_log_probs
            kl = (kl_per_token * token_mask).sum() / token_mask.sum().clamp(min=1.0)

            # Entropy bonus
            probs = new_all_log_probs.exp()
            entropy_per_token = -(probs * new_all_log_probs).sum(dim=-1)  # [B, T]
            entropy = (entropy_per_token * token_mask).sum() / token_mask.sum().clamp(min=1.0)

            loss = pg_loss + args.kl_coeff * kl - args.entropy_coeff * entropy

            # Gradient accumulation across PPO epochs
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            accum_pg_loss += pg_loss.item()
            accum_kl += kl.item()
            accum_entropy += entropy.item()
            accum_loss += loss.item()

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # -------------------------------------------------------------------
        # 7. Logging
        # -------------------------------------------------------------------
        avg_pg_loss = accum_pg_loss / args.ppo_epochs
        avg_kl = accum_kl / args.ppo_epochs
        avg_entropy = accum_entropy / args.ppo_epochs
        avg_loss = accum_loss / args.ppo_epochs
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        accuracy = hit24.mean().item()

        metrics = {
            "step": step,
            "loss": avg_loss,
            "pg_loss": avg_pg_loss,
            "kl": avg_kl,
            "entropy": avg_entropy,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "accuracy": accuracy,
        }

        if use_wandb:
            wandb.log(metrics, step=step)

        if step % 50 == 0 or step < 5:
            print(
                f"Step {step:5d} | loss={avg_loss:.4f} pg={avg_pg_loss:.4f} "
                f"kl={avg_kl:.4f} ent={avg_entropy:.4f} | "
                f"reward={reward_mean:.3f}+/-{reward_std:.3f} acc={accuracy:.3f}"
            )
            for expr, r in zip(expressions[:3], rewards[:3].tolist()):
                print(f"  expr='{expr}' reward={r:.2f}")

    # -----------------------------------------------------------------------
    # Save final adapter
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nSaved LoRA adapter to {output_dir}")

    if use_wandb:
        wandb.finish()

    print("Training complete.")


if __name__ == "__main__":
    main()
