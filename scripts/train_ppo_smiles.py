#!/usr/bin/env python
"""PPO training script for the SMILES QED task.

Trains a LoRA-adapted Llama-3.2-1B to generate SMILES fragments with high QED,
using a manual PPO loop with grammar-constrained decoding.

Grammar constraint uses transformers_cfg (same as the GFlowNet experiments).
"""

import argparse
import json
import os
import sys
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for SMILES QED")
    parser.add_argument("--output_dir", type=str, default="./logs/rl_baselines/ppo_smiles")
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
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Grammar processor (identical to Expr24 version)
# ---------------------------------------------------------------------------
class WrappedGrammarProcessor:
    def __init__(self, processor):
        self.processor = processor

    def init_for_prompt(self, prompt_len: int):
        self.processor.reset()
        self.processor.set_prompt_length(prompt_len)

    def __call__(self, input_ids, scores):
        result = self.processor(input_ids, scores)
        if isinstance(result, dict):
            return result["masked_logits"]
        return result


def build_grammar_processor(tokenizer, grammar_path, legal_tokens_path):
    with open(grammar_path) as f:
        grammar_str = f.read()
    parsed_grammar = parse_ebnf(grammar_str)

    with open(legal_tokens_path) as f:
        legal_tokens = [line.strip() for line in f if line.strip()]
    legal_token_ids = []
    skipped = []
    for tok in legal_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) == 1:
            legal_token_ids.append(ids[0])
        else:
            skipped.append((tok, ids))
    if skipped:
        print(f"Warning: skipped {len(skipped)} multi-token entries")
    legal_token_ids.append(tokenizer.eos_token_id)

    raw_processor = GrammarIncrementalLogitsProcessorGeneral(
        parsed_grammar,
        tokenizer=tokenizer,
        nice_token_ids_list=legal_token_ids,
        execution_mode="limited",
    )
    return WrappedGrammarProcessor(raw_processor)


# ---------------------------------------------------------------------------
# Reward: QED score
# ---------------------------------------------------------------------------
def compute_rewards(generated_ids, tokenizer):
    """Compute QED reward for each sample."""
    from rdkit import Chem
    from rdkit.Chem import QED as QED_module

    batch_size = generated_ids.shape[0]
    rewards = torch.zeros(batch_size, device=generated_ids.device)
    qed_scores = torch.zeros(batch_size, device=generated_ids.device)
    fragments = []

    for i in range(batch_size):
        tokens = generated_ids[i]
        pieces = []
        for t in tokens:
            tid = t.item()
            if tid == tokenizer.eos_token_id:
                break
            pieces.append(tokenizer.decode(t, skip_special_tokens=False))
        frag = "".join("".join(pieces).split())
        fragments.append(frag)

        if not frag:
            continue
        try:
            mol = Chem.MolFromSmiles(frag)
            if mol is not None:
                qed = QED_module.qed(mol)
                rewards[i] = qed
                qed_scores[i] = qed
        except Exception:
            pass

    return rewards, qed_scores, fragments


def build_token_mask(generated, eos_token_id):
    is_eos = generated == eos_token_id
    cumsum_eos = is_eos.cumsum(dim=1)
    return (cumsum_eos <= 1).float()


def autoregressive_generate(
    model,
    prompt_ids,
    grammar_processor,
    n_samples,
    max_new_tokens,
    min_new_tokens,
    temperature,
    eos_token_id,
):
    device = prompt_ids.device
    batch_prompt = prompt_ids.expand(n_samples, -1)
    prompt_len = batch_prompt.shape[1]

    grammar_processor.init_for_prompt(prompt_len)

    current_ids = batch_prompt.clone()
    gen_token_ids = []
    gen_token_lps = []
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)

    for t in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(current_ids)
            logits = outputs.logits[:, -1, :]

        constrained_logits = grammar_processor(current_ids, logits.clone())

        if temperature != 1.0:
            constrained_logits = constrained_logits / temperature

        all_neg_inf = (constrained_logits == float("-inf")).all(dim=-1)
        if all_neg_inf.any():
            constrained_logits[all_neg_inf, eos_token_id] = 0.0

        constrained_logits = constrained_logits.clamp(min=-1e4, max=1e4)
        constrained_logits = torch.nan_to_num(constrained_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        log_probs = F.log_softmax(constrained_logits, dim=-1)
        probs = log_probs.exp().clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        next_tokens = torch.where(
            finished, torch.full_like(next_tokens, eos_token_id), next_tokens
        )

        token_lps = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
        gen_token_ids.append(next_tokens)
        gen_token_lps.append(token_lps)

        finished = finished | (next_tokens == eos_token_id)
        current_ids = torch.cat([current_ids, next_tokens.unsqueeze(1)], dim=1)

        if finished.all() and (t + 1) >= min_new_tokens:
            break

    gen_ids = torch.stack(gen_token_ids, dim=1)
    gen_log_probs = torch.stack(gen_token_lps, dim=1)
    return current_ids, gen_ids, gen_log_probs


def compute_log_probs_for_tokens(model, full_ids, generated, prompt_len):
    logits = model(full_ids).logits
    gen_len = generated.shape[1]
    relevant_logits = logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]
    all_log_probs = F.log_softmax(relevant_logits, dim=-1)
    token_log_probs = all_log_probs.gather(2, generated.unsqueeze(-1)).squeeze(-1)
    return token_log_probs, all_log_probs


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name="ppo_smiles", config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SMILES prompts
    prompt_path = PROJECT_ROOT / "data" / "SMILES" / "sidechain_prompts_qed.json"
    with open(prompt_path) as f:
        prompt_data = json.load(f)
    prompt_text = prompt_data[0]["prompt"]
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=True).to(
        device
    )
    prompt_len = prompt_ids.shape[1]
    print(f"Prompt: {prompt_text[:80]}...")
    print(f"Prompt length: {prompt_len} tokens")

    # Grammar
    grammar_path = PROJECT_ROOT / "assets" / "SMILES_grammars" / "generic.ebnf"
    legal_tokens_path = (
        PROJECT_ROOT / "assets" / "token_list" / "SMILES" / "allowed_llama3.2_1B_allowed_token"
    )
    grammar_processor = build_grammar_processor(tokenizer, grammar_path, legal_tokens_path)
    print(f"Grammar constraint enabled ({grammar_path.name})")

    # Models
    print(f"Loading model: {args.model_name}")
    model_kwargs = dict(dtype=torch.bfloat16)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, attn_implementation="flash_attention_2", **model_kwargs
        )
    except Exception:
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

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

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    print(f"\nStarting PPO-SMILES for {args.max_steps} steps, {args.n_samples} samples/step")

    model.train()
    for step in range(args.max_steps):
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
            continue

        rewards, qed_scores, fragments = compute_rewards(gen_ids, tokenizer)
        token_mask = build_token_mask(gen_ids, tokenizer.eos_token_id)

        with torch.no_grad():
            ref_token_log_probs, _ = compute_log_probs_for_tokens(
                ref_model, all_ids, gen_ids, prompt_len
            )

        old_token_log_probs = old_gen_log_probs.detach()

        with torch.no_grad():
            if rewards.std() > 1e-8:
                advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                advantage = rewards - rewards.mean()
            advantage_tokens = advantage.unsqueeze(1).expand(-1, gen_len) * token_mask

        accum_pg_loss = accum_kl = accum_entropy = accum_loss = 0.0
        optimizer.zero_grad()

        for ppo_epoch in range(args.ppo_epochs):
            new_token_log_probs, new_all_log_probs = compute_log_probs_for_tokens(
                model, all_ids, gen_ids, prompt_len
            )

            ratio = (new_token_log_probs - old_token_log_probs).exp()
            surr1 = ratio * advantage_tokens
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantage_tokens
            pg_loss = -torch.min(surr1, surr2)
            pg_loss = (pg_loss * token_mask).sum() / token_mask.sum().clamp(min=1.0)

            kl_per_token = new_token_log_probs - ref_token_log_probs
            kl = (kl_per_token * token_mask).sum() / token_mask.sum().clamp(min=1.0)

            probs = new_all_log_probs.exp()
            entropy_per_token = -(probs * new_all_log_probs).sum(dim=-1)
            entropy = (entropy_per_token * token_mask).sum() / token_mask.sum().clamp(min=1.0)

            loss = pg_loss + args.kl_coeff * kl - args.entropy_coeff * entropy
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            accum_pg_loss += pg_loss.item()
            accum_kl += kl.item()
            accum_entropy += entropy.item()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        reward_mean = rewards.mean().item()
        valid_rate = (rewards > 0).float().mean().item()

        metrics = {
            "step": step,
            "loss": accum_loss / args.ppo_epochs,
            "pg_loss": accum_pg_loss / args.ppo_epochs,
            "kl": accum_kl / args.ppo_epochs,
            "entropy": accum_entropy / args.ppo_epochs,
            "reward_mean": reward_mean,
            "qed_mean": qed_scores.mean().item(),
            "valid_rate": valid_rate,
        }

        if use_wandb:
            wandb.log(metrics, step=step)

        if step % 50 == 0 or step < 5:
            print(
                f"Step {step:5d} | loss={metrics['loss']:.4f} kl={metrics['kl']:.4f} "
                f"ent={metrics['entropy']:.4f} | "
                f"qed={metrics['qed_mean']:.3f} valid={valid_rate:.3f}"
            )
            for frag, r in zip(fragments[:3], rewards[:3].tolist()):
                print(f"  frag='{frag}' qed={r:.3f}")

    output_dir = Path(args.output_dir) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nSaved to {output_dir}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
