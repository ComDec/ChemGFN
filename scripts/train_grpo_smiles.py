#!/usr/bin/env python
"""GRPO training script for the SMILES QED task.

Trains a LoRA-adapted Llama-3.2-1B to generate SMILES fragments with high QED,
using TRL's GRPOTrainer with grammar-constrained decoding.

Grammar constraint uses transformers_cfg (same as the GFlowNet experiments).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList
from transformers_cfg.generation.logits_process import (
    GrammarIncrementalLogitsProcessorGeneral,
)
from transformers_cfg.parser import parse_ebnf
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chemgfn.models.validators import RDKitValidator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for SMILES QED")
    parser.add_argument("--output_dir", type=str, default="./logs/rl_baselines/grpo_smiles")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_generations", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="ChemGFN")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_completion_length", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_grammar", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Grammar processor (same as GFlowNet, same as Expr24 version)
# ---------------------------------------------------------------------------
class WrappedGrammarProcessor(torch.nn.Module):
    def __init__(self, processor):
        super().__init__()
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
        print(f"Warning: skipped {len(skipped)} multi-token entries in legal_tokens")
    legal_token_ids.append(tokenizer.eos_token_id)

    raw_processor = GrammarIncrementalLogitsProcessorGeneral(
        parsed_grammar,
        tokenizer=tokenizer,
        nice_token_ids_list=legal_token_ids,
        execution_mode="limited",
    )
    return WrappedGrammarProcessor(raw_processor)


# ---------------------------------------------------------------------------
# Reward function: QED score via RDKitValidator
# ---------------------------------------------------------------------------
_validator = RDKitValidator(scorer="qed", backend="pa")


def smiles_reward_func(completions: list[str], **kwargs) -> list[float]:
    """Reward = QED score of generated SMILES fragment.

    Uses RDKitValidator.score_smiles() which:
    1. Combines scaffold + fragment into full SMILES
    2. Validates with RDKit
    3. Returns QED score in [0, 1]

    For simplicity (no scaffold info in GRPO), we evaluate the fragment directly.
    """
    rewards = []
    for completion in completions:
        frag = completion.strip().replace(" ", "")
        if not frag:
            rewards.append(0.0)
            continue
        try:
            from rdkit import Chem
            from rdkit.Chem import QED as QED_module

            mol = Chem.MolFromSmiles(frag)
            if mol is not None:
                qed_score = QED_module.qed(mol)
                rewards.append(float(qed_score))
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Grammar-constrained GRPOTrainer (same as Expr24)
# ---------------------------------------------------------------------------
class GrammarGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, grammar_logits_processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grammar_logits_processor = grammar_logits_processor

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        if self.grammar_logits_processor is None:
            return super()._generate_single_turn(prompt_ids, images, multimodal_fields)
        if self.use_vllm or self.use_transformers_paged:
            return super()._generate_single_turn(prompt_ids, images, multimodal_fields)

        from contextlib import nullcontext

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from trl.trainer.grpo_trainer import (
            pad,
            profiling_context,
            unwrap_model_for_generation,
        )

        device = self.accelerator.device
        prompt_len = len(prompt_ids[0])
        self.grammar_logits_processor.init_for_prompt(prompt_len)

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

        logits_processor = LogitsProcessorList([self.grammar_logits_processor])

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
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs,
                generation_config=self.generation_config,
                logits_processor=logits_processor,
                disable_compile=True,
            )

        prompt_length = generate_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        completion_ids = [
            c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)
        ]
        return completion_ids, None, {}


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SMILES prompts
    prompt_path = PROJECT_ROOT / "data" / "SMILES" / "sidechain_prompts_qed.json"
    with open(prompt_path) as f:
        prompt_data = json.load(f)
    # Use first prompt (all prompts are similar scaffold-conditioned)
    prompt_text = prompt_data[0]["prompt"]
    prompt_ids_test = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_len = len(prompt_ids_test)
    print(f"Prompt: {prompt_text[:80]}...")
    print(f"Prompt length: {prompt_len} tokens")

    # Grammar constraint
    grammar_processor = None
    if not args.no_grammar:
        grammar_path = PROJECT_ROOT / "assets" / "SMILES_grammars" / "generic.ebnf"
        legal_tokens_path = (
            PROJECT_ROOT / "assets" / "token_list" / "SMILES" / "allowed_llama3.2_1B_allowed_token"
        )
        grammar_processor = build_grammar_processor(tokenizer, grammar_path, legal_tokens_path)
        print(f"Grammar constraint enabled ({grammar_path.name})")

    # Dataset
    num_examples = (
        args.max_steps * args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    num_examples = int(num_examples * 1.1) + 100
    # Alternate between prompts
    prompts = [p["prompt"] for p in prompt_data]
    dataset_prompts = [prompts[i % len(prompts)] for i in range(num_examples)]
    dataset = Dataset.from_dict({"prompt": dataset_prompts})
    print(f"Dataset size: {num_examples} ({len(prompts)} unique prompts)")

    # LoRA config (same as GFlowNet)
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

    # GRPO config
    report_to = "none" if args.no_wandb else "wandb"
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        seed=args.seed,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=args.max_completion_length,
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
        run_name="grpo_smiles",
    )

    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    trainer = GrammarGRPOTrainer(
        model=args.model_name,
        reward_funcs=smiles_reward_func,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        grammar_logits_processor=grammar_processor,
    )

    print(f"\nStarting GRPO-SMILES training for {args.max_steps} steps")
    print(f"  num_generations={args.num_generations}, beta={args.beta}")
    print(f"  max_completion_length={args.max_completion_length}")
    print(f"  grammar_constrained={not args.no_grammar}")

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved to {final_dir}")


if __name__ == "__main__":
    main()
