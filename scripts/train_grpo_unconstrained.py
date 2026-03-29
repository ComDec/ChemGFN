#!/usr/bin/env python
"""Unconstrained GRPO training for SMILES QED and Expr24 tasks.

No grammar constraints — uses soft vocab masking (penalty on non-task tokens)
and validity-based reward (invalid → 0). This matches the standard RL paradigm
used in REINVENT, PSV-PPO, and other molecular generation RL papers.

Usage:
    # SMILES QED
    python scripts/train_grpo_unconstrained.py --task smiles

    # Expr24
    python scripts/train_grpo_unconstrained.py --task expr24
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from trl import GRPOConfig, GRPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SoftVocabPenaltyProcessor(LogitsProcessor):
    """Soft penalty on non-task tokens (following gfn-lm-tuning approach).
    Adds a large negative value to logits of tokens not in the legal set."""

    def __init__(self, legal_token_ids: set, penalty: float = -50.0, vocab_size: int = 128256):
        self.penalty = penalty
        self.mask = torch.ones(vocab_size)
        for tid in legal_token_ids:
            if tid < vocab_size:
                self.mask[tid] = 0.0

    def __call__(self, input_ids, scores):
        penalty = self.mask.to(scores.device) * self.penalty
        if penalty.shape[0] < scores.shape[1]:
            penalty = torch.nn.functional.pad(penalty, (0, scores.shape[1] - penalty.shape[0]))
        elif penalty.shape[0] > scores.shape[1]:
            penalty = penalty[: scores.shape[1]]
        return scores + penalty.unsqueeze(0)


def build_soft_vocab_processor(tokenizer, legal_tokens_path, penalty=-50.0):
    with open(legal_tokens_path) as f:
        legal_tokens = [line.strip() for line in f if line.strip()]
    legal_ids = set()
    for tok in legal_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) == 1:
            legal_ids.add(ids[0])
    legal_ids.add(tokenizer.eos_token_id)
    vocab_size = max(tokenizer.vocab_size, max(tokenizer.get_vocab().values()) + 1)
    print(f"Soft vocab processor: {len(legal_ids)} legal tokens, penalty={penalty}")
    return SoftVocabPenaltyProcessor(legal_ids, penalty, vocab_size)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=["smiles", "expr24"])
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_generations", type=int, default=32)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--max_completion_length", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--wandb_project", type=str, default="ChemGFN")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--vocab_penalty", type=float, default=-50.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Task-specific reward functions
# ---------------------------------------------------------------------------
def _make_expr24_reward():
    from chemgfn.models.validators import Expr24Validator

    validator = Expr24Validator(scorer="hit24", target_value=24)

    def reward_func(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for c in completions:
            expr = c.strip().replace(" ", "")
            if not expr:
                rewards.append(0.0)
                continue
            _, score, _ = validator._score_expression(expr)
            rewards.append(score)
        return rewards

    return reward_func


def _make_smiles_reward():
    from rdkit import Chem
    from rdkit.Chem import QED

    def reward_func(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for c in completions:
            frag = c.strip().replace(" ", "")
            if not frag:
                rewards.append(0.0)
                continue
            try:
                mol = Chem.MolFromSmiles(frag)
                rewards.append(float(QED.qed(mol)) if mol else 0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    return reward_func


def main():
    args = parse_args()

    # Task defaults
    if args.task == "smiles":
        default_output = "./logs/rl_baselines/grpo_smiles_v2"
        default_max_len = 10
        prompt_path = PROJECT_ROOT / "data" / "SMILES" / "sidechain_prompts_qed.json"
        reward_func = _make_smiles_reward()
        run_name = "grpo_smiles_unconstrained"
    else:
        default_output = "./logs/rl_baselines/grpo_expr24_v2"
        default_max_len = 11
        prompt_path = PROJECT_ROOT / "data" / "24_points" / "prompts.txt"
        reward_func = _make_expr24_reward()
        run_name = "grpo_expr24_unconstrained"

    output_dir = args.output_dir or default_output
    max_completion_length = args.max_completion_length or default_max_len

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompts
    if args.task == "smiles":
        with open(prompt_path) as f:
            prompt_data = json.load(f)
        prompts = [p["prompt"] for p in prompt_data]
    else:
        prompts = [prompt_path.read_text().strip()]

    prompt_text = prompts[0]
    print(f"Task: {args.task}")
    print(f"Prompt: {prompt_text[:80]}...")
    print(f"No grammar constraint (unconstrained generation)")

    # Dataset
    num_examples = (
        args.max_steps * args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    num_examples = int(num_examples * 1.1) + 100
    dataset_prompts = [prompts[i % len(prompts)] for i in range(num_examples)]
    dataset = Dataset.from_dict({"prompt": dataset_prompts})

    # LoRA (same as GFlowNet)
    peft_config = LoraConfig(
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

    report_to = "none" if args.no_wandb else "wandb"
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=args.max_steps,
        seed=args.seed,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        report_to=report_to,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        num_train_epochs=1,
        remove_unused_columns=False,
        run_name=run_name,
    )

    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # Build soft vocab processor for task tokens
    if args.task == "smiles":
        legal_path = (
            PROJECT_ROOT / "assets" / "token_list" / "SMILES" / "allowed_llama3.2_1B_allowed_token"
        )
    else:
        legal_path = PROJECT_ROOT / "assets" / "token_list" / "24_points" / "general"
    vocab_processor = build_soft_vocab_processor(tokenizer, legal_path, args.vocab_penalty)

    # Custom trainer to inject soft vocab penalty during generation
    class SoftVocabGRPOTrainer(GRPOTrainer):
        def __init__(self, *a, soft_processor=None, **kw):
            super().__init__(*a, **kw)
            self.soft_processor = soft_processor

        def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
            if self.soft_processor is None or self.use_vllm or self.use_transformers_paged:
                return super()._generate_single_turn(prompt_ids, images, multimodal_fields)
            from contextlib import nullcontext

            import numpy as np
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from trl.trainer.grpo_trainer import (
                pad,
                profiling_context,
                unwrap_model_for_generation,
            )

            device = self.accelerator.device
            prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
            padded_ids = pad(prompt_tensors, padding_value=self.pad_token_id, padding_side="left")
            attention_mask = pad(
                [torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left"
            )
            generate_inputs = {"input_ids": padded_ids, "attention_mask": attention_mask}
            for k, v in multimodal_fields.items():
                if isinstance(v, torch.Tensor):
                    generate_inputs[k] = v
                elif isinstance(v, list) and v and isinstance(v[0], list):
                    generate_inputs[k] = pad(
                        [torch.tensor(x) for x in v], padding_value=0, padding_side="left"
                    )
                else:
                    generate_inputs[k] = torch.tensor(np.array(v))
            generate_inputs = super(GRPOTrainer, self)._prepare_inputs(generate_inputs)

            lp = LogitsProcessorList([self.soft_processor])
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=self.generation_kwargs,
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext(),
            ):
                out = unwrapped_model.generate(
                    **generate_inputs,
                    generation_config=self.generation_config,
                    logits_processor=lp,
                    disable_compile=True,
                )
            prompt_length = generate_inputs["input_ids"].size(1)
            completion_ids = out[:, prompt_length:]
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full(
                (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
            )
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            seq_idx = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            cmask = (seq_idx <= eos_idx.unsqueeze(1)).int()
            completion_ids = [
                c[m].tolist() for c, m in zip(completion_ids, cmask.bool(), strict=True)
            ]
            return completion_ids, None, {}

    trainer = SoftVocabGRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        soft_processor=vocab_processor,
    )

    print(
        f"\nGRPO unconstrained: {args.max_steps} steps, beta={args.beta}, lr={args.learning_rate}"
    )
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()
