#!/usr/bin/env python
"""GRPO training script for the VarExpr24 task.

Trains a LoRA-adapted Llama-3.2-1B to generate arithmetic expressions that
evaluate to 24, using TRL's GRPOTrainer with grammar-constrained decoding.

Grammar constraint uses transformers_cfg (same as the GFlowNet experiments).
"""

import argparse
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
# Add project root to path so we can import from chemgfn
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chemgfn.models.validators import Expr24Validator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for VarExpr24")
    parser.add_argument("--output_dir", type=str, default="./logs/grpo_expr24")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_generations", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="ChemGFN")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_completion_length", type=int, default=9)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--no_grammar", action="store_true", help="Disable grammar constraints")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Per-device batch size for training.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Grammar processor setup using transformers_cfg (matching GFlowNet config)
# ---------------------------------------------------------------------------
class WrappedGrammarProcessor(torch.nn.Module):
    """Wraps transformers_cfg grammar processor to return a plain tensor.

    GrammarIncrementalLogitsProcessorGeneral returns a dict
    {'masked_logits': Tensor, 'acceptance': Tensor}. This wrapper extracts
    'masked_logits' so it works with LogitsProcessorList (expects tensor returns).

    IMPORTANT: Must call init_for_prompt(prompt_len) before each generation batch
    to tell the processor where the prompt ends and grammar-constrained generation begins.
    """

    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def init_for_prompt(self, prompt_len: int):
        """Reset state and set prompt length before each generation batch."""
        self.processor.reset()
        self.processor.set_prompt_length(prompt_len)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        result = self.processor(input_ids, scores)
        if isinstance(result, dict):
            return result["masked_logits"]
        return result


def build_grammar_processor(tokenizer, grammar_path, legal_tokens_path):
    """Build the same grammar processor used in GFlowNet VarExpr24 experiments.

    Uses transformers_cfg's GrammarIncrementalLogitsProcessorGeneral with
    execution_mode="limited", matching ChemGFNModule._build_pre_grammar_processor()
    with processor_type="prefix".
    """
    with open(grammar_path) as f:
        grammar_str = f.read()
    parsed_grammar = parse_ebnf(grammar_str)

    with open(legal_tokens_path) as f:
        legal_tokens = [line.strip() for line in f if line.strip()]
    legal_token_ids = []
    for tok in legal_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        assert (
            len(ids) == 1
        ), f"Legal token '{tok}' encodes to {ids} (length {len(ids)}), expected single token"
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
# Reward function compatible with TRL GRPOTrainer
# ---------------------------------------------------------------------------
_validator = Expr24Validator(scorer="hit24", target_value=24)


def expr24_reward_func(completions: list[str], **kwargs) -> list[float]:
    """Reward function for GRPOTrainer.

    Args:
        completions: list of decoded completion strings (without prompt).
        **kwargs: additional keyword arguments from GRPOTrainer.

    Returns:
        list of float rewards. 1.0 if expression evaluates to 24, else 0.0.
    """
    rewards = []
    for completion in completions:
        expr_str = completion.strip().replace(" ", "")
        if not expr_str:
            rewards.append(0.0)
            continue
        is_valid, score, value = _validator._score_expression(expr_str)
        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# Grammar-constrained GRPOTrainer subclass
# ---------------------------------------------------------------------------
class GrammarGRPOTrainer(GRPOTrainer):
    """GRPOTrainer subclass that injects grammar-constrained logits processing
    into the generation step."""

    def __init__(self, *args, grammar_logits_processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grammar_logits_processor = grammar_logits_processor

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        """Override to inject grammar constraint logits processor into the
        regular (non-vLLM) generation path."""
        if self.grammar_logits_processor is None:
            return super()._generate_single_turn(prompt_ids, images, multimodal_fields)

        if self.use_vllm or self.use_transformers_paged:
            return super()._generate_single_turn(prompt_ids, images, multimodal_fields)

        # --- Replicate the regular generation path with logits_processor injected ---
        from contextlib import nullcontext

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from trl.trainer.grpo_trainer import (
            pad,
            profiling_context,
            unwrap_model_for_generation,
        )

        device = self.accelerator.device

        # Initialize grammar processor for this prompt length
        prompt_len = len(prompt_ids[0])  # all prompts same length
        self.grammar_logits_processor.init_for_prompt(prompt_len)

        # Left-pad token IDs into tensors
        prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
        padded_ids = pad(prompt_tensors, padding_value=self.pad_token_id, padding_side="left")
        attention_mask = pad(
            [torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left"
        )
        generate_inputs = {"input_ids": padded_ids, "attention_mask": attention_mask}

        # Handle multimodal fields
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

        # Compute prompt length and extract completion ids
        prompt_length = generate_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        completion_ids = [
            c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)
        ]
        logprobs = None
        extra_fields = {}

        return completion_ids, logprobs, extra_fields


def main():
    args = parse_args()

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
    prompt_ids_test = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_len = len(prompt_ids_test)
    print(f"Prompt length: {prompt_len} tokens")

    # -----------------------------------------------------------------------
    # Grammar constraint logits processor (transformers_cfg, same as GFlowNet)
    # -----------------------------------------------------------------------
    grammar_processor = None
    if not args.no_grammar:
        grammar_path = PROJECT_ROOT / "assets" / "24_grammars" / "var_length.ebnf"
        legal_tokens_path = PROJECT_ROOT / "assets" / "token_list" / "24_points" / "general"
        grammar_processor = build_grammar_processor(tokenizer, grammar_path, legal_tokens_path)
        print(f"Grammar constraint enabled (transformers_cfg, {grammar_path.name})")

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    num_examples = (
        args.max_steps * args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    num_examples = int(num_examples * 1.1) + 100  # buffer
    dataset = Dataset.from_dict({"prompt": [prompt_text] * num_examples})
    print(f"Dataset size: {num_examples} examples (repeated single prompt)")

    # -----------------------------------------------------------------------
    # LoRA config (matching GFlowNet experiments)
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # GRPOConfig
    # -----------------------------------------------------------------------
    report_to = "none" if args.no_wandb else "wandb"

    # generation_batch_size must be divisible by num_generations
    generation_batch_size = args.num_generations

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        seed=args.seed,
        num_generations=args.num_generations,
        generation_batch_size=generation_batch_size,
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
        run_name="grpo_expr24",
    )

    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # -----------------------------------------------------------------------
    # Create trainer
    # -----------------------------------------------------------------------
    trainer = GrammarGRPOTrainer(
        model=args.model_name,
        reward_funcs=expr24_reward_func,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        grammar_logits_processor=grammar_processor,
    )

    print(f"\nStarting GRPO training for {args.max_steps} steps")
    print(f"  num_generations={args.num_generations}")
    print(f"  per_device_train_batch_size={args.per_device_train_batch_size}")
    print(f"  gradient_accumulation_steps={args.gradient_accumulation_steps}")
    print(f"  max_completion_length={args.max_completion_length}")
    print(f"  beta={args.beta}")
    print(f"  learning_rate={args.learning_rate}")
    print(f"  grammar_constrained={not args.no_grammar}")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    trainer.train()

    # -----------------------------------------------------------------------
    # Save final adapter
    # -----------------------------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved final LoRA adapter to {final_dir}")

    print("Training complete.")


if __name__ == "__main__":
    main()
