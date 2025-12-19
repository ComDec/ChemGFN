"""
Naive PPO training loop for the 24-point expression task.

- Policy: causal LM from 🤗 transformers (e.g., `meta-llama/Llama-3.2-1B` or any small local model).
- Reward: 1.0 if a generated expression has the format d op d op d op d (length 7) and evaluates to 24, else 0.0.
- Action space: restricted to digits 0-9 and operators + - * / plus EOS.

This script is intentionally lightweight and self-contained for experimentation.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -------------------------
# 24-point validator
# -------------------------
def is_expr24(tokens: list[str]) -> float:
    """Return 1.0 if tokens form a valid 24 expression, else 0.0."""
    if len(tokens) != 7:
        return 0.0
    # pattern d op d op d op d
    for i, tk in enumerate(tokens):
        if i % 2 == 0:
            if not tk.isdigit() or len(tk) != 1:
                return 0.0
        else:
            if tk not in {"+", "-", "*", "/"}:
                return 0.0

    # evaluate with operator precedence (* / before + -)
    nums = [float(tokens[i]) for i in (0, 2, 4, 6)]
    ops = [tokens[i] for i in (1, 3, 5)]
    try:
        # handle * and /
        v = nums[:]
        o = ops[:]
        i = 0
        while i < len(o):
            if o[i] in "*/":
                a, b = v[i], v[i + 1]
                if o[i] == "*":
                    res = a * b
                else:
                    if b == 0:
                        return 0.0
                    res = a / b
                v[i : i + 2] = [res]
                o.pop(i)
            else:
                i += 1
        acc = v[0]
        for op, b in zip(o, v[1:]):
            acc = acc + b if op == "+" else acc - b
        return 1.0 if abs(acc - 24) < 1e-6 else 0.0
    except Exception:
        return 0.0


# -------------------------
# Sampling and PPO buffers
# -------------------------
def sample_action(
    logits: torch.Tensor, legal_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one token id with a legal mask."""
    logits = logits + legal_mask  # mask already -inf for illegal
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    tok = dist.sample()
    logp = dist.log_prob(tok)
    return tok, logp


def decode_tokens(tokenizer, tokens: list[int]) -> list[str]:
    """Decode per-token (no joins) ensuring single-char tokens for digits/operators."""
    return [tokenizer.decode([t], skip_special_tokens=True) for t in tokens]


# -------------------------
# PPO core
# -------------------------
@dataclass
class PPOConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    lr: float = 5e-6
    batch_size: int = 32
    rollout_len: int = 7
    epochs: int = 1000
    ppo_clip: float = 0.2
    vf_coef: float = 0.0  # no value func here; keep 0 to simplify
    ent_coef: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    )


class PPOLearner:
    def __init__(self, cfg: PPOConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )
        if cfg.use_lora:
            lora_cfg = LoraConfig(
                target_modules=list(cfg.lora_target_modules),
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias=cfg.lora_bias,
                fan_in_fan_out=False,
            )
            base_model = get_peft_model(base_model, lora_cfg)

        self.model = base_model.to(cfg.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        # build legal mask
        allowed_tokens = list("0123456789+-*/")
        allowed_ids = {self.tokenizer.convert_tokens_to_ids(tok) for tok in allowed_tokens}
        allowed_ids.add(self.tokenizer.eos_token_id)
        vocab_size = len(self.tokenizer)
        illegal = torch.full((vocab_size,), float("-inf"))
        illegal[list(allowed_ids)] = 0.0
        self.legal_mask = illegal.to(cfg.device)

    def rollout(self) -> dict:
        """Generate one episode."""
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.cfg.device)
        logps = []
        actions = []

        for _ in range(self.cfg.rollout_len):
            logits = self.model(input_ids=input_ids).logits[:, -1, :]
            tok, logp = sample_action(logits, self.legal_mask)
            actions.append(tok)
            logps.append(logp)
            input_ids = torch.cat([input_ids, tok.unsqueeze(0)], dim=1)

        tokens = [int(t.item()) for t in actions]
        decoded = decode_tokens(self.tokenizer, tokens)
        reward = torch.tensor([is_expr24(decoded)], device=self.cfg.device, dtype=torch.float32)

        return {
            "actions": torch.tensor(tokens, device=self.cfg.device),
            "logps": torch.stack(logps).detach().to(self.cfg.device),
            "reward": reward,
            "decoded": decoded,
        }

    def ppo_step(self, batch: list[dict]) -> float:
        rewards = torch.stack([b["reward"] for b in batch]).squeeze()  # [B]
        # advantage: centered rewards
        adv = rewards - rewards.mean()

        # recompute logprobs under current policy
        actions = torch.stack([b["actions"] for b in batch])  # [B, T]
        B, T = actions.shape
        input_ids = torch.full(
            (B, 1), self.tokenizer.bos_token_id, device=self.cfg.device, dtype=torch.long
        )

        cur_logps_steps = []
        for t in range(T):
            logits = self.model(input_ids=input_ids).logits[:, -1, :]
            tok_logp = (
                F.log_softmax(logits + self.legal_mask, dim=-1)
                .gather(-1, actions[:, t].unsqueeze(-1))
                .squeeze(-1)
            )
            cur_logps_steps.append(tok_logp)
            input_ids = torch.cat([input_ids, actions[:, t].unsqueeze(-1)], dim=1)
        cur_logps = torch.stack(cur_logps_steps, dim=1)  # [B, T]

        old_logps = torch.stack([b["logps"] for b in batch], dim=0)  # [B, T]
        ratio = torch.exp(cur_logps - old_logps.squeeze(-1))

        adv = adv.unsqueeze(1).expand_as(ratio)  # same advantage per step
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # entropy bonus to keep exploration
        entropy = -(cur_logps.exp() * cur_logps).mean()
        loss = policy_loss - self.cfg.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())

    def train(self):
        eval_rows = []
        for step in range(1, self.cfg.epochs + 1):
            batch = [self.rollout() for _ in range(self.cfg.batch_size)]
            loss = self.ppo_step(batch)
            avg_r = torch.stack([b["reward"] for b in batch]).mean().item()
            if step % 1 == 0:
                sample_expr = " ".join(batch[0]["decoded"])
                print(f"[step {step}] loss={loss:.4f} avg_reward={avg_r:.4f} sample={sample_expr}")
            if step % 100 == 0:
                samples = [self.rollout() for _ in range(50)]
                rewards = torch.stack([s["reward"] for s in samples]).squeeze()
                acc = rewards.mean().item()
                decoded_samples = [" ".join(s["decoded"]) for s in samples[:100]]
                eval_rows.append(
                    {
                        "step": step,
                        "acc": acc,
                        "reward_mean": rewards.mean().item(),
                        "reward_std": rewards.std().item(),
                        "examples": decoded_samples,
                    }
                )
                print(
                    f"[eval step {step}] acc={acc:.4f} "
                    f"rewards_mean={rewards.mean().item():.4f} "
                    f"rewards_std={rewards.std().item():.4f} "
                    f"examples={decoded_samples}"
                )
        if eval_rows:
            df = pd.DataFrame(eval_rows)
            out_path = os.path.join(os.path.dirname(__file__), "ppo_expr24_eval.csv")
            df.to_csv(out_path, index=False)
            print(f"Saved eval metrics to {out_path}")


if __name__ == "__main__":
    cfg = PPOConfig()
    learner = PPOLearner(cfg)
    learner.train()
