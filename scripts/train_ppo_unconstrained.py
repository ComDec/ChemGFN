#!/usr/bin/env python
"""Unconstrained PPO training for SMILES QED and Expr24 tasks.

No grammar constraints — uses soft vocab masking (large penalty on non-task
tokens) and validity-based reward. Matches standard RL paradigm in molecular
generation literature (REINVENT, PSV-PPO, etc.).

Usage:
    # SMILES QED
    python scripts/train_ppo_unconstrained.py --task smiles

    # Expr24
    python scripts/train_ppo_unconstrained.py --task expr24
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=["smiles", "expr24"])
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_coeff", type=float, default=0.1)
    p.add_argument("--entropy_coeff", type=float, default=0.05)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--min_new_tokens", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument(
        "--vocab_penalty",
        type=float,
        default=-50.0,
        help="Soft penalty on non-task tokens (like gfn-lm-tuning)",
    )
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--wandb_project", type=str, default="ChemGFN")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Soft vocab masking (following gfn-lm-tuning approach)
# ---------------------------------------------------------------------------
def build_vocab_mask(tokenizer, legal_tokens_path):
    """Build a soft mask: 0.0 for legal tokens, vocab_penalty for illegal ones."""
    vocab_size = tokenizer.vocab_size
    if hasattr(tokenizer, "get_vocab"):
        vocab_size = max(vocab_size, max(tokenizer.get_vocab().values()) + 1)
    # Start with all illegal
    mask = torch.ones(vocab_size, dtype=torch.float32)

    with open(legal_tokens_path) as f:
        legal_tokens = [line.strip() for line in f if line.strip()]

    legal_ids = set()
    for tok in legal_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) == 1:
            legal_ids.add(ids[0])
    legal_ids.add(tokenizer.eos_token_id)

    for tid in legal_ids:
        if tid < vocab_size:
            mask[tid] = 0.0

    n_legal = int((mask == 0).sum().item())
    print(f"Soft vocab mask: {n_legal} legal tokens, {vocab_size - n_legal} penalized")
    return mask  # multiply by vocab_penalty before adding to logits


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
def make_reward_fn(task, tokenizer):
    if task == "expr24":
        from chemgfn.models.validators import Expr24Validator

        validator = Expr24Validator(scorer="hit24", target_value=24)

        def compute_rewards(gen_ids):
            B = gen_ids.shape[0]
            rewards = torch.zeros(B, device=gen_ids.device)
            exprs = []
            for i in range(B):
                pieces = []
                for t in gen_ids[i]:
                    if t.item() == tokenizer.eos_token_id:
                        break
                    pieces.append(tokenizer.decode(t, skip_special_tokens=False))
                expr = "".join("".join(pieces).split())
                exprs.append(expr)
                if expr:
                    _, score, _ = validator._score_expression(expr)
                    rewards[i] = score
            return rewards, exprs

    else:  # smiles
        from rdkit import Chem
        from rdkit.Chem import QED

        def compute_rewards(gen_ids):
            B = gen_ids.shape[0]
            rewards = torch.zeros(B, device=gen_ids.device)
            frags = []
            for i in range(B):
                pieces = []
                for t in gen_ids[i]:
                    if t.item() == tokenizer.eos_token_id:
                        break
                    pieces.append(tokenizer.decode(t, skip_special_tokens=False))
                frag = "".join("".join(pieces).split())
                frags.append(frag)
                if frag:
                    try:
                        mol = Chem.MolFromSmiles(frag)
                        if mol:
                            rewards[i] = QED.qed(mol)
                    except Exception:
                        pass
            return rewards, frags

    return compute_rewards


# ---------------------------------------------------------------------------
# Generation + PPO utilities
# ---------------------------------------------------------------------------
def autoregressive_generate(
    model,
    prompt_ids,
    n_samples,
    max_new_tokens,
    min_new_tokens,
    temperature,
    eos_token_id,
    vocab_penalty_mask=None,
    vocab_penalty=-50.0,
):
    device = prompt_ids.device
    batch_prompt = prompt_ids.expand(n_samples, -1)
    current_ids = batch_prompt.clone()
    gen_token_ids, gen_token_lps = [], []
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)

    for t in range(max_new_tokens):
        with torch.no_grad():
            logits = model(current_ids).logits[:, -1, :]

        # Soft vocab masking (add penalty to non-task tokens)
        if vocab_penalty_mask is not None:
            penalty = vocab_penalty_mask.to(device) * vocab_penalty
            if penalty.shape[0] < logits.shape[1]:
                penalty = F.pad(penalty, (0, logits.shape[1] - penalty.shape[0]))
            elif penalty.shape[0] > logits.shape[1]:
                penalty = penalty[: logits.shape[1]]
            logits = logits + penalty.unsqueeze(0)

        if temperature != 1.0:
            logits = logits / temperature

        logits = logits.clamp(min=-1e4, max=1e4)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp().clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        next_tokens = torch.where(
            finished, torch.full_like(next_tokens, eos_token_id), next_tokens
        )

        gen_token_ids.append(next_tokens)
        gen_token_lps.append(log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1))

        finished = finished | (next_tokens == eos_token_id)
        current_ids = torch.cat([current_ids, next_tokens.unsqueeze(1)], dim=1)

        if finished.all() and (t + 1) >= min_new_tokens:
            break

    return current_ids, torch.stack(gen_token_ids, 1), torch.stack(gen_token_lps, 1)


def compute_log_probs(
    model,
    full_ids,
    generated,
    prompt_len,
    vocab_penalty_mask=None,
    vocab_penalty=-50.0,
):
    logits = model(full_ids).logits
    gen_len = generated.shape[1]
    relevant = logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]
    # Apply same soft vocab penalty as generation (critical for correct ratio)
    if vocab_penalty_mask is not None:
        penalty = (
            (vocab_penalty_mask.to(relevant.device) * vocab_penalty).unsqueeze(0).unsqueeze(0)
        )
        if penalty.shape[-1] < relevant.shape[-1]:
            penalty = F.pad(penalty, (0, relevant.shape[-1] - penalty.shape[-1]))
        elif penalty.shape[-1] > relevant.shape[-1]:
            penalty = penalty[..., : relevant.shape[-1]]
        relevant = relevant + penalty
    all_lp = F.log_softmax(relevant, dim=-1)
    tok_lp = all_lp.gather(2, generated.unsqueeze(-1)).squeeze(-1)
    return tok_lp, all_lp


def build_token_mask(gen_ids, eos_id):
    return (gen_ids.eq(eos_id).cumsum(dim=1) <= 1).float()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Defaults per task
    if args.task == "smiles":
        default_output = "./logs/rl_baselines/ppo_smiles_v2"
        default_max_new = 10
        prompt_path = PROJECT_ROOT / "data" / "SMILES" / "sidechain_prompts_qed.json"
        legal_tokens_path = (
            PROJECT_ROOT / "assets" / "token_list" / "SMILES" / "allowed_llama3.2_1B_allowed_token"
        )
        run_name = "ppo_smiles_unconstrained"
    else:
        default_output = "./logs/rl_baselines/ppo_expr24_v2"
        default_max_new = 11
        prompt_path = PROJECT_ROOT / "data" / "24_points" / "prompts.txt"
        legal_tokens_path = PROJECT_ROOT / "assets" / "token_list" / "24_points" / "general"
        run_name = "ppo_expr24_unconstrained"

    output_dir = args.output_dir or default_output
    max_new_tokens = args.max_new_tokens or default_max_new

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prompt
    if args.task == "smiles":
        with open(prompt_path) as f:
            prompt_data = json.load(f)
        prompt_text = prompt_data[0]["prompt"]
    else:
        prompt_text = prompt_path.read_text().strip()
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=True).to(
        device
    )
    prompt_len = prompt_ids.shape[1]
    print(f"Task: {args.task}, Prompt len: {prompt_len}")

    # Soft vocab mask
    vocab_mask = build_vocab_mask(tokenizer, legal_tokens_path)

    # Models
    model_kwargs = dict(dtype=torch.bfloat16)
    try:
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name, attn_implementation="flash_attention_2", **model_kwargs
        )
    except Exception:
        base = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

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
    model = get_peft_model(base, lora_config).to(device)
    model.print_trainable_parameters()

    try:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, attn_implementation="flash_attention_2", **model_kwargs
        )
    except Exception:
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    ref_model.to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    compute_rewards = make_reward_fn(args.task, tokenizer)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    print(
        f"PPO unconstrained: {args.max_steps} steps, kl={args.kl_coeff}, ent={args.entropy_coeff}"
    )
    print(f"Soft vocab penalty: {args.vocab_penalty} on {int(vocab_mask.sum())} tokens")

    model.train()
    for step in range(args.max_steps):
        model.eval()
        with torch.no_grad():
            all_ids, gen_ids, old_lps = autoregressive_generate(
                model,
                prompt_ids,
                args.n_samples,
                max_new_tokens,
                args.min_new_tokens,
                args.temperature,
                tokenizer.eos_token_id,
                vocab_mask,
                args.vocab_penalty,
            )
        model.train()

        gen_len = gen_ids.shape[1]
        if gen_len == 0:
            continue

        rewards, samples = compute_rewards(gen_ids)
        mask = build_token_mask(gen_ids, tokenizer.eos_token_id)

        with torch.no_grad():
            ref_lps, _ = compute_log_probs(
                ref_model, all_ids, gen_ids, prompt_len, vocab_mask, args.vocab_penalty
            )

        old_lps = old_lps.detach()
        with torch.no_grad():
            adv = (
                (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                if rewards.std() > 1e-8
                else rewards - rewards.mean()
            )
            adv_tok = adv.unsqueeze(1).expand(-1, gen_len) * mask

        optimizer.zero_grad()
        acc_loss = acc_kl = acc_ent = 0.0

        for _ in range(args.ppo_epochs):
            new_lps, new_all_lps = compute_log_probs(
                model, all_ids, gen_ids, prompt_len, vocab_mask, args.vocab_penalty
            )
            ratio = (new_lps - old_lps).exp()
            s1 = ratio * adv_tok
            s2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv_tok
            pg = -(torch.min(s1, s2) * mask).sum() / mask.sum().clamp(min=1)
            kl = ((new_lps - ref_lps) * mask).sum() / mask.sum().clamp(min=1)
            ent = (-(new_all_lps.exp() * new_all_lps).sum(-1) * mask).sum() / mask.sum().clamp(
                min=1
            )
            loss = pg + args.kl_coeff * kl - args.entropy_coeff * ent
            (loss / args.ppo_epochs).backward()
            acc_loss += loss.item()
            acc_kl += kl.item()
            acc_ent += ent.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        r_mean = rewards.mean().item()
        valid_rate = (rewards > 0).float().mean().item()
        metrics = {
            "step": step,
            "loss": acc_loss / args.ppo_epochs,
            "kl": acc_kl / args.ppo_epochs,
            "entropy": acc_ent / args.ppo_epochs,
            "reward_mean": r_mean,
            "valid_rate": valid_rate,
        }
        if use_wandb:
            import wandb

            wandb.log(metrics, step=step)

        if step % 50 == 0 or step < 5:
            print(
                f"Step {step:5d} | loss={metrics['loss']:.4f} kl={metrics['kl']:.4f} "
                f"ent={metrics['entropy']:.4f} | reward={r_mean:.3f} valid={valid_rate:.3f}"
            )
            for s, r in zip(samples[:3], rewards[:3].tolist()):
                print(f"  '{s}' → {r:.3f}")

    out = Path(output_dir) / "final"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    if use_wandb:
        import wandb

        wandb.finish()
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
