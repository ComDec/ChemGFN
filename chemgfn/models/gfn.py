"""GFlowNet fine-tuning of an autoregressive language model.

:class:`ChemGFNModule` ties together grammar-constrained sampling, reward evaluation against
a reference prior, and a GFlowNet objective (TB, SubTB, RapTB or AvgPrefixTB) in a single
Lightning module. It also produces the sample dumps and per-length / prefix-depth tables that
the reported metrics are computed from.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Sequence, cast

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from lightning import LightningModule
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from transformers_cfg.generation.logits_process import GrammarIncrementalLogitsProcessorGeneral
from transformers_cfg.parser import parse_ebnf

from chemgfn.models.losses import GFNLoss
from chemgfn.models.reward import compute_active_before
from chemgfn.utils.diversity import SequenceDiversity
from chemgfn.utils.gfn_utils import (
    calculate_diversity_by_length,
    generate_and_return_termination_logprob,
    get_termination_vals,
    prepare_token_mask,
)
from chemgfn.utils.prefix_metrics import (
    PrefixCollapseResult,
    prefix_collapse_by_k,
    prefix_collapse_by_position,
)
from chemgfn.utils.replay_buffer import ReplayBuffer

sys.setrecursionlimit(1500)


@dataclass
class LengthStatistics:
    """Length-conditioned accumulators for one stage (train, validation or test).

    Counts and score sums are keyed by pre-EOS sequence length. The ``_valid`` variants are
    restricted to samples the validator accepted, which is the population the paper's
    length and score statistics are reported on.
    """

    counts: dict[int, int] = field(default_factory=dict)
    score_sums: dict[int, float] = field(default_factory=dict)
    score_counts: dict[int, int] = field(default_factory=dict)
    counts_valid: dict[int, int] = field(default_factory=dict)
    score_sums_valid: dict[int, float] = field(default_factory=dict)
    score_counts_valid: dict[int, int] = field(default_factory=dict)
    log_pterm_sums: dict[int, float] = field(default_factory=dict)
    log_pterm_counts: dict[int, int] = field(default_factory=dict)

    def clear(self) -> None:
        """Drop every accumulated value."""
        for accumulator in vars(self).values():
            accumulator.clear()


@dataclass
class DiversityStatistics:
    """Running sums used to average the per-batch diversity metrics over an epoch."""

    batch_sum: float = 0.0
    batch_count: int = 0
    fp_internal_sum: float = 0.0
    fp_internal_count: int = 0
    fp_topk_sum: float = 0.0
    fp_topk_count: int = 0
    text_sum: float = 0.0
    text_count: int = 0

    def clear(self) -> None:
        """Reset every running sum and count to zero."""
        for name, value in list(vars(self).items()):
            setattr(self, name, type(value)())

    @staticmethod
    def mean(total: float, count: int) -> float | None:
        """Return ``total / count``, or ``None`` when nothing was accumulated."""
        return total / float(count) if count > 0 else None


class ChemGFNModule(LightningModule):
    """Lightning module training an LLM sampler with a GFlowNet objective.

    The policy is a LoRA adapter on a frozen base model; the same base model, unadapted,
    serves as the reference prior in the reward. Each step samples trajectories under the
    task grammar, scores them with the task validator, and minimises the configured
    GFlowNet loss. Samples are recycled through a replay buffer and an optional dataset
    buffer.
    """

    def __init__(
        self,
        net_config: dict[str, Any],
        lora_config: LoraConfig,
        tokenizer: PreTrainedTokenizer,
        reward: Any,
        loss_fn: GFNLoss,
        reward_buffer: ReplayBuffer,
        reward_config: dict[str, Any],
        training_mixed_config: dict[str, Any],
        constraint_config: dict[str, Any],
        optimizer: Callable[[Any], torch.optim.Optimizer],
        scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] | None,
        factor_schedulers: dict[str, Any],
        disable_peft: bool = False,
    ) -> None:
        """Build the policy, reference model, grammar processor and training state.

        Args:
            net_config: Base-model settings; must provide ``pretrained_model_name_or_path``.
            lora_config: LoRA adapter configuration applied to the policy.
            tokenizer: Tokenizer shared by the policy, reference model and validators.
            reward: Reward object exposing ``score`` and ``sentence_validator``.
            loss_fn: GFlowNet objective to minimise.
            reward_buffer: Replay buffer storing high-reward trajectories.
            reward_config: Reward-side settings such as the diversity metric.
            training_mixed_config: Sampling and buffer-mixing settings.
            constraint_config: Grammar, length bounds and legal-token settings.
            optimizer: Optimizer factory bound to the trainable parameters.
            scheduler: Optional learning-rate scheduler factory.
            factor_schedulers: Named step-indexed schedules (reward temperature, buffer
                probabilities, prefix-length bounds, ...).
            disable_peft: Train the full base model instead of a LoRA adapter.
        """
        super().__init__()

        self.save_hyperparameters(ignore=["net", "loss_fn"])
        self.net_config: Any = net_config
        self.reward_config: Any = reward_config
        self.constraint_config: Any = constraint_config
        self.training_mixed_config: Any = training_mixed_config

        model = AutoModelForCausalLM.from_pretrained(self.net_config.pretrained_model_name_or_path)
        model.train()

        model_frozen = AutoModelForCausalLM.from_pretrained(
            self.net_config.pretrained_model_name_or_path
        )
        model_frozen.eval()
        model_frozen.requires_grad_(False)

        self.net = model if disable_peft else get_peft_model(model, lora_config)
        self.net_frozen = model_frozen
        self.tokenizer = tokenizer

        self.end_of_sentence_token_id: int = int(self.tokenizer.eos_token_id)
        eos_override = getattr(self.constraint_config, "end_of_sentence_token", None) or getattr(
            self.constraint_config, "termination_token", None
        )
        if eos_override is not None:
            eos_text = str(eos_override)
            token_ids = self.tokenizer.encode(eos_text, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    "end_of_sentence_token/termination_token must map to exactly one token. "
                    f"Got {len(token_ids)} tokens for {eos_text!r}: {token_ids}"
                )
            self.end_of_sentence_token_id = int(token_ids[0])

        self.factor_schedulers = factor_schedulers
        self.get_reward_temp_at_step = self.factor_schedulers["reward_temp"]
        self.get_scaling_factor_at_step = self.factor_schedulers["scaling_factor"]
        self.get_reference_logits_scale_at_step = self.factor_schedulers["reference_logits_scale"]
        self.get_replay_buffer_at_step = self.factor_schedulers["replay_buffer"]
        self.get_dataset_buffer_at_step = self.factor_schedulers["dataset_buffer"]
        self.get_pf_temp_low_at_step = getattr(self.factor_schedulers, "pf_temp_low", None)
        self.get_pf_temp_high_at_step = getattr(self.factor_schedulers, "pf_temp_high", None)
        self.get_prefix_len_at_step = getattr(self.factor_schedulers, "k_max", None)
        self.get_k_min_at_step = getattr(self.factor_schedulers, "k_min", None)

        self.buffer_mixture_ratio = training_mixed_config["buffer_mixture_ratio"]

        self.reward = reward
        self.reward_buffer: ReplayBuffer = reward_buffer
        self.reward_buffer.set_termination_token_id(self.end_of_sentence_token_id)

        # Keep reward/validator termination consistent with our generation EOS.
        sentence_validator = getattr(self.reward, "sentence_validator", None)
        if sentence_validator is not None and hasattr(sentence_validator, "termination_token_id"):
            sentence_validator.termination_token_id = int(self.end_of_sentence_token_id)

        (
            self.legal_tokens_mask,
            self.illegal_tokens_mask,
            self.legal_token_ids_list,
        ) = self._load_token_masks()
        self.pre_grammar_processor = self._build_grammar_processor()

        self.illegal_vocab_penalty = float(getattr(reward, "illegal_vocab_penalty", 0))

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn: GFNLoss = loss_fn
        if hasattr(self.loss_fn, "set_weight_schedulers"):
            self.loss_fn.set_weight_schedulers(dict(self.factor_schedulers))

        self.val_samples_ids: list[list[int]] = []
        self.test_samples_ids: list[list[int]] = []

        self.val_samples_table: list[dict[str, Any]] = []
        self.test_samples_table: list[dict[str, Any]] = []

        self.val_log_rs: list[torch.Tensor] = []
        self.val_log_pfss: list[torch.Tensor] = []
        self.test_log_rs: list[torch.Tensor] = []
        self.test_log_pfss: list[torch.Tensor] = []

        self.val_samples_valid_flags: list[bool] = []
        self.test_samples_valid_flags: list[bool] = []

        self.val_length_stats = LengthStatistics()
        self.test_length_stats = LengthStatistics()
        self.val_diversity_stats = DiversityStatistics()
        self.test_diversity_stats = DiversityStatistics()

        self.val_decoded_seqs_and_scores: list[tuple[str, float]] = []
        self.test_decoded_seqs_and_scores: list[tuple[str, float]] = []

        div_metric = getattr(self.reward_config, "diversity_metric", None)
        div_model_name = getattr(
            self.reward_config,
            "diversity_model_name",
            "sentence-transformers/all-mpnet-base-v2",
        )
        self.sequence_diversity = (
            SequenceDiversity(div_metric, model_name=div_model_name) if div_metric else None
        )

        self.test_repeat_suffix: str = ""

    def forward(
        self,
        encoded_data: dict[str, Any],
        n_samples: int | None = None,
        pf_temperature: float = 1.0,
        reward_temperature: float = 1.0,
        scaling_factor: float = 1,
        reference_logits_scale: float = 0.5,
        action_seq: torch.Tensor | None = None,
        use_buffer_sample: bool = False,
        buffer_sample: torch.Tensor | None = None,
        buffer_mixture_ratio: float = 0.5,
    ) -> dict[str, Any]:
        """Sample trajectories from the policy and score them.

        Args:
            encoded_data: Batch holding ``encoded_prompt`` and optional task conditioning.
            n_samples: Number of trajectories to draw; defaults to the configured value.
            pf_temperature: Temperature applied to the forward policy.
            reward_temperature: Temperature applied to the reward.
            scaling_factor: Multiplier on the task score inside the reward.
            reference_logits_scale: Weight of the reference prior in the reward.
            action_seq: Pre-determined token sequence to evaluate instead of sampling.
            use_buffer_sample: Mix dataset-buffer sequences into the sampled batch.
            buffer_sample: Dataset-buffer sequences to mix in.
            buffer_mixture_ratio: Fraction of the batch replaced by buffer sequences.

        Returns:
            Dictionary with the generated ``state`` and the per-step ``log_pf``,
            ``log_pterm``, ``log_r`` tables plus their reference-model counterparts.
        """
        encoded_prompt = encoded_data["encoded_prompt"]
        encoded_prompt = encoded_prompt.squeeze(0) if encoded_prompt.ndim != 1 else encoded_prompt
        n_samples = n_samples or self.training_mixed_config.n_samples
        encoded_prompt = encoded_prompt.expand(n_samples, -1)
        encoded_data["encoded_prompt"] = encoded_prompt

        return generate_and_return_termination_logprob(
            model=self.net,
            encoded_data=encoded_data,
            grammar_processor=self.pre_grammar_processor,
            reward_fn=partial(
                self.reward.score,
                prompt_length=encoded_prompt.shape[1],
                model=self.net_frozen,
                tokenizer=self.tokenizer,
            ),
            termination_token_id=self.end_of_sentence_token_id,
            min_len=self.constraint_config.min_sentence_len,
            max_len=self.constraint_config.max_sentence_len,
            temperature=pf_temperature,
            reward_temperature=reward_temperature,
            scaling_factor=scaling_factor,
            reference_logits_scale=reference_logits_scale,
            action_seq=action_seq,
            vocab_nice_mask=self.legal_tokens_mask,
            vocab_invalid_mask=self.illegal_tokens_mask,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
            use_buffer_sample=use_buffer_sample,
            buffer_sample=buffer_sample,
            buffer_mixture_ratio=buffer_mixture_ratio,
            disable_grammar=getattr(self.constraint_config, "disable_grammar", False),
            grammar_disagree_penalty=getattr(
                self.training_mixed_config, "grammar_disagree_penalty", -80
            ),
        )

    # ------------------------------------------------------------------ #
    # Training / validation loops
    # ------------------------------------------------------------------ #

    def training_step(self, item: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Sample a batch, compute the GFlowNet loss and update the replay buffer."""
        encoded_prompt = item["encoded_prompt"]
        prompt_len = encoded_prompt.shape[-1]
        buffer_sample = item["buffer_encoded_sample"]
        use_dataset_buffer = False

        # Buffer 1: replay buffer.
        replay_buffer_result = None
        if random.random() <= self.get_replay_buffer_at_step(self.global_step):
            replay_buffer_result = self.generate_from_replay_buffer(item, encoded_prompt)
        use_replay_buffer = replay_buffer_result is not None

        if use_replay_buffer:
            _, result_dict = replay_buffer_result
            pf_temp = 1.0
        else:
            pf_temp = self._sample_pf_temperature()

            # Buffer 2: dataset buffer, under a step-dependent probability schedule.
            use_dataset_buffer = (
                buffer_sample is not None
                and random.random() < self.get_dataset_buffer_at_step(self.global_step)
            )
            result_dict = self.forward(
                item,
                pf_temperature=pf_temp,
                reward_temperature=self.reward.temperature,
                scaling_factor=self.get_scaling_factor_at_step(self.global_step),
                reference_logits_scale=self.get_reference_logits_scale_at_step(self.global_step),
                use_buffer_sample=use_dataset_buffer,
                buffer_sample=buffer_sample,
                buffer_mixture_ratio=self.buffer_mixture_ratio,
            )

        generated_text = result_dict["state"]
        log_pf = result_dict["log_pf"]
        log_pterm = result_dict["log_pterm"]
        log_pf_ref = result_dict["log_pf_ref"]
        log_pterm_ref = result_dict["log_pterm_ref"]
        model_log_r = result_dict["log_r"]
        log_r_unpenalized = result_dict["log_r_unpenalized"]

        if use_replay_buffer:
            log_r = model_log_r[:, : max(0, generated_text.shape[1] - prompt_len)]
        else:
            log_r = model_log_r
            self.reward_buffer.add_batch(
                prompt=encoded_prompt,
                sentences=generated_text[:, prompt_len:],
                logrewards=model_log_r * self.reward.temperature,
                tokenizer=self.tokenizer,
                result_dict=result_dict,
            )

        self._log_metrics(
            {
                f"scheduled/{key}": schedule(self.global_step)
                for key, schedule in self.factor_schedulers.items()
            },
            sync_dist=True,
            on_step=True,
        )

        # Grammar-agreement diagnostics.
        if not use_replay_buffer or not use_dataset_buffer:
            self._log_agreement_metrics(result_dict["agree_list"])

        if hasattr(self.loss_fn, "set_global_step"):
            self.loss_fn.set_global_step(self.global_step)

        loss_output = self._compute_loss(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_pf_ref=log_pf_ref,
            log_pterm_ref=log_pterm_ref,
            prompt_len=prompt_len,
        )
        loss = loss_output["loss"]
        self._log_loss_components("train", loss_output, on_step=True, prog_bar=True)

        _, last_log_r, last_log_r_unpenalized, sentence_len = get_termination_vals(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=prompt_len,
        )
        tokens = generated_text[:, prompt_len:]

        validator_metric_dict = self._validator_metrics(tokens, item.get("scaffold", None))
        self._log_validator_core_metrics(
            "train",
            validator_metric_dict,
            sync_dist=True,
            on_step=True,
        )

        log_ps = last_log_r * self.reward.temperature
        log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature

        if batch_idx % 5 == 0:
            self._log_metrics(
                {
                    f"train/replay_buffer_{key}": float(value)
                    for key, value in self.reward_buffer.stat().items()
                },
                on_step=True,
                sync_dist=True,
            )

        self._log_metrics(
            {
                "train/reward_var": log_r.var(dim=0).mean(),
                "train/loss": (loss, {"prog_bar": True}),
                "train/logR": last_log_r.mean(),
                "train/logP(s) (avg)": log_ps.mean(),
                "train/logP(s) (max)": log_ps.max(),
                "train/logP(s) unpenalized (avg)": log_ps_unpenalized.mean(),
                "train/logP(s) unpenalized (max)": log_ps_unpenalized.max(),
                "train/Mean(log_pterm - log_pterm_ref)": (log_pterm - log_pterm_ref).mean(),
                "train/sentence_len": sentence_len.float().mean(),
            },
            on_step=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Sample from the current policy and accumulate the validation metrics."""
        encoded_prompt = batch["encoded_prompt"]
        prompt_len = encoded_prompt.shape[-1]
        result_dict = self.forward(
            batch, reward_temperature=1.0, pf_temperature=self._eval_pf_temperature()
        )
        generated_text = result_dict["state"]
        log_pf = result_dict["log_pf"]
        log_pterm = result_dict["log_pterm"]
        log_r = result_dict["log_r"]
        log_pf_ref = result_dict["log_pf_ref"]
        log_pterm_ref = result_dict["log_pterm_ref"]
        log_r_unpenalized = result_dict["log_r_unpenalized"]
        validator_dict = result_dict.get("validator_dict")

        if hasattr(self.loss_fn, "set_global_step"):
            self.loss_fn.set_global_step(self.global_step)

        loss_output = self._compute_loss(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_pf_ref=log_pf_ref,
            log_pterm_ref=log_pterm_ref,
            prompt_len=prompt_len,
        )
        loss = loss_output["loss"]
        self._log_loss_components("val", loss_output, on_step=False)

        log_pfs, last_log_r, last_log_r_unpenalized, sentence_len = get_termination_vals(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=prompt_len,
        )
        tokens = generated_text[:, prompt_len:]
        mean_len_tok_ids = self._mean_token_length(tokens, device=generated_text.device)

        self._accumulate_log_pterm_by_length(tokens, log_pterm, self.val_length_stats)
        if log_pfs is not None:
            self.val_log_rs.append(last_log_r.detach().cpu())
            self.val_log_pfss.append(log_pfs.detach().cpu())

        valid_flags = self._get_valid_flags(validator_dict, generated_text, prompt_len)

        batch_sequences = self._strip_eos_from_batch(tokens)
        self.val_samples_ids.extend(batch_sequences)
        self.val_samples_valid_flags.extend(valid_flags)
        self.val_diversity_stats.batch_sum += self._calculate_diversity_ragged(batch_sequences)
        self.val_diversity_stats.batch_count += 1
        self._accumulate_text_diversity("val", tokens, self.val_diversity_stats)

        validator_metric_dict = self._validator_metrics(tokens, batch.get("scaffold", None))
        self._pop_fp_diversity(validator_metric_dict, self.val_diversity_stats)
        self._log_validator_core_metrics(
            "val",
            validator_metric_dict,
            sync_dist=True,
            on_epoch=True,
        )
        self._log_validator_full_metrics(
            "val/validator",
            validator_metric_dict,
            sync_dist=True,
            on_epoch=True,
        )
        self._accumulate_length_statistics(
            self.val_length_stats,
            tokens,
            validator_metric_dict,
            validator_dict,
            valid_flags,
        )

        log_ps = last_log_r * self.get_reward_temp_at_step(self.global_step)
        log_ps_unpenalized = last_log_r_unpenalized * self.get_reward_temp_at_step(
            self.global_step
        )
        self._log_metrics(
            {
                "val/loss": (loss, {"prog_bar": True}),
                "val/logR": last_log_r.mean(),
                "val/logP(s) (avg)": log_ps.mean(),
                "val/logP(s) (max)": log_ps.max(),
                "val/logP(s) unpenalized (avg)": log_ps_unpenalized.mean(),
                "val/logP(s) unpenalized (max)": log_ps_unpenalized.max(),
                "val/Mean(log_pterm - log_pterm_ref)": (log_pterm - log_pterm_ref).mean(),
                # Two length views: the EOS position reported by get_termination_vals, and the
                # length read straight off the token ids (robust when EOS is absent).
                "val/sentence_len": sentence_len.float().mean(),
                "val/len_tok_ids": mean_len_tok_ids,
            },
            sync_dist=True,
        )

        self._record_samples(
            self.val_samples_table,
            self.val_decoded_seqs_and_scores,
            result_dict=result_dict,
            generated_text=generated_text,
            prompt_len=prompt_len,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_pf_ref=log_pf_ref,
            log_pterm_ref=log_pterm_ref,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            valid_flags=valid_flags,
        )

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Sample from the trained policy and accumulate the reported test metrics."""
        encoded_prompt = batch["encoded_prompt"]
        prompt_len = encoded_prompt.shape[-1]
        result_dict = self.forward(
            batch, reward_temperature=1.0, pf_temperature=self._eval_pf_temperature()
        )
        generated_text = result_dict["state"]
        log_pf = result_dict["log_pf"]
        log_pterm = result_dict["log_pterm"]
        log_r = result_dict["log_r"]
        log_pf_ref = result_dict["log_pf_ref"]
        log_pterm_ref = result_dict["log_pterm_ref"]
        log_r_unpenalized = result_dict["log_r_unpenalized"]
        validator_dict = result_dict.get("validator_dict")

        batch_sequences = self._strip_eos_from_batch(generated_text[:, prompt_len:])
        self.test_samples_ids.extend(batch_sequences)
        self.test_diversity_stats.batch_sum += self._calculate_diversity_ragged(batch_sequences)
        self.test_diversity_stats.batch_count += 1

        if hasattr(self.loss_fn, "set_global_step"):
            self.loss_fn.set_global_step(self.global_step)

        loss_output = self._compute_loss(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_pf_ref=log_pf_ref,
            log_pterm_ref=log_pterm_ref,
            prompt_len=prompt_len,
        )
        loss = loss_output["loss"]
        self._log_loss_components("test", loss_output, on_step=False)

        log_pfs, last_log_r, last_log_r_unpenalized, _ = get_termination_vals(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=prompt_len,
        )
        tokens = generated_text[:, prompt_len:]
        self._accumulate_text_diversity("test", tokens, self.test_diversity_stats)
        self._accumulate_log_pterm_by_length(tokens, log_pterm, self.test_length_stats)
        if log_pfs is not None:
            self.test_log_rs.append(last_log_r.detach().cpu())
            self.test_log_pfss.append(log_pfs.detach().cpu())

        valid_flags = self._get_valid_flags(validator_dict, generated_text, prompt_len)
        self.test_samples_valid_flags.extend(valid_flags)

        validator_metric_dict = self._validator_metrics(tokens, batch.get("scaffold", None))
        self._pop_fp_diversity(validator_metric_dict, self.test_diversity_stats)
        self._log_validator_core_metrics(
            "test",
            validator_metric_dict,
            sync_dist=True,
            on_epoch=True,
        )
        self._log_validator_full_metrics(
            "test/validator",
            validator_metric_dict,
            sync_dist=True,
            on_epoch=True,
        )
        self._accumulate_length_statistics(
            self.test_length_stats,
            tokens,
            validator_metric_dict,
            validator_dict,
            valid_flags,
        )

        log_ps = last_log_r * self.get_reward_temp_at_step(self.global_step)
        log_ps_unpenalized = last_log_r_unpenalized * self.get_reward_temp_at_step(
            self.global_step
        )
        self._log_metrics(
            {
                "test/loss": (loss, {"prog_bar": True}),
                "test/logR": last_log_r.mean(),
                "test/logP(s) (avg)": log_ps.mean(),
                "test/logP(s) (max)": log_ps.max(),
                "test/logP(s) unpenalized (avg)": log_ps_unpenalized.mean(),
                "test/logP(s) unpenalized (max)": log_ps_unpenalized.max(),
                "test/Mean(log_pterm - log_pterm_ref)": (log_pterm - log_pterm_ref).mean(),
            },
            sync_dist=True,
        )

        self._record_samples(
            self.test_samples_table,
            self.test_decoded_seqs_and_scores,
            result_dict=result_dict,
            generated_text=generated_text,
            prompt_len=prompt_len,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_pf_ref=log_pf_ref,
            log_pterm_ref=log_pterm_ref,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            valid_flags=valid_flags,
        )

    def on_train_batch_start(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Push the scheduled reward temperature, scaling factor and learning rate."""
        reward_temp = self.get_reward_temp_at_step(self.global_step)
        scaling_factor = self.get_scaling_factor_at_step(self.global_step)
        lr_sched = cast(Any, self.lr_schedulers())
        lr = lr_sched.get_lr()[0] if lr_sched is not None else 0.0
        self.reward.temperature = reward_temp
        self.reward.scaling_factor = scaling_factor
        self._log_metrics({"train/reward_temp": reward_temp}, sync_dist=True, on_step=True)

        for pg in cast(Any, self.optimizers()).param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------ #
    # Epoch hooks
    # ------------------------------------------------------------------ #

    def on_train_start(self) -> None:
        """Dump samples from the untrained policy as a reference point."""
        val_dataset = self.trainer.datamodule.val_dataloader().dataset
        n_probe = min(10, int(len(val_dataset)))
        if n_probe <= 0:
            return
        probes = torch.utils.data.Subset(
            val_dataset, random.sample(range(len(val_dataset)), n_probe)
        )
        self.sample_probes(probes).to_csv(
            os.path.join(
                self.trainer.default_root_dir,
                f"samples_val_probes_wo_train_{self.trainer.global_step}.csv",
            ),
            index=False,
        )

    def on_train_epoch_start(self) -> None:
        """Log the scheduled reward temperature and learning rate for the new epoch."""
        self._log_metrics(
            {
                "scheduled/R_temperature": self.get_reward_temp_at_step(self.global_step),
                "scheduled/lr": (
                    cast(Any, self.lr_schedulers()).get_lr()[0] if self.lr_schedulers() else 0.0
                ),
            },
            sync_dist=True,
        )

    def on_train_epoch_end(self) -> None:
        """Persist the replay buffer contents for the finished epoch."""
        if hasattr(self, "global_rank") and self.global_rank != 0:
            return

        self.reward_buffer.save_csv(
            os.path.join(
                self.trainer.default_root_dir,
                "replay_buffer",
                f"replay_{self.trainer.current_epoch}.csv",
            ),
            self.tokenizer,
        )

    def on_validation_epoch_start(self) -> None:
        """Reset the per-epoch validation accumulators."""
        self.val_samples_ids.clear()
        self.val_samples_table.clear()
        self.val_log_rs.clear()
        self.val_log_pfss.clear()
        self.val_samples_valid_flags.clear()
        self.val_length_stats.clear()
        self.val_diversity_stats.clear()
        self.val_decoded_seqs_and_scores.clear()

    def on_validation_epoch_end(self) -> None:
        """Aggregate diversity, prefix-collapse and length metrics over the epoch."""
        diversity = self._calculate_diversity_ragged(self.val_samples_ids)
        self._log_metrics({"val/diversity": diversity}, sync_dist=True, on_epoch=True)
        valid_val_sequences = self._filter_valid_sequences(
            self.val_samples_ids, self.val_samples_valid_flags
        )
        diversity_val_valid = self._calculate_diversity_ragged(valid_val_sequences)
        self._log_metrics(
            {"val/diversity_valid": diversity_val_valid}, sync_dist=True, on_epoch=True
        )

        stats = self.val_diversity_stats
        batch_mean = stats.mean(stats.batch_sum, stats.batch_count)
        if batch_mean is not None:
            self._log_metrics(
                {"val/diversity_batch_mean": batch_mean}, sync_dist=True, on_epoch=True
            )
        text_mean = stats.mean(stats.text_sum, stats.text_count)
        if self.sequence_diversity is not None and text_mean is not None:
            self._log_metrics(
                {"val/diversity_text_batch_mean": text_mean}, sync_dist=True, on_epoch=True
            )
        mean_internal = stats.mean(stats.fp_internal_sum, stats.fp_internal_count)
        if mean_internal is not None:
            self._log_metrics(
                {
                    "val/fp_div_internal_valid_batch_mean": mean_internal,
                    "val/validator/fp_div_internal_valid_batch_mean": mean_internal,
                },
                sync_dist=True,
                on_epoch=True,
            )
        mean_topk = stats.mean(stats.fp_topk_sum, stats.fp_topk_count)
        if mean_topk is not None:
            self._log_metrics(
                {
                    "val/fp_div_topk_valid_batch_mean": mean_topk,
                    "val/validator/fp_div_topk_valid_batch_mean": mean_topk,
                },
                sync_dist=True,
                on_epoch=True,
            )

        diversity_by_len = calculate_diversity_by_length(
            self.val_samples_ids, self.end_of_sentence_token_id
        )
        diversity_by_len_valid = calculate_diversity_by_length(
            valid_val_sequences, self.end_of_sentence_token_id
        )
        self._log_text_diversity_epoch("val", self.val_samples_ids)

        fp_div = self._compute_global_fp_diversity(self.val_samples_ids)
        if fp_div:
            self._log_metrics(
                {
                    "val/fp_div_internal_valid": fp_div["fp_div_internal_valid"],
                    "val/fp_div_topk_valid": fp_div["fp_div_topk_valid"],
                    "val/validator/fp_div_internal_valid": fp_div["fp_div_internal_valid"],
                    "val/validator/fp_div_topk_valid": fp_div["fp_div_topk_valid"],
                },
                sync_dist=True,
                on_epoch=True,
            )

        pos, kmet = self._compute_prefix_collapse(
            self.val_samples_ids,
            self.val_samples_valid_flags,
            k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        self._log_metrics(
            {
                "prefix_pos/top1_auc_val": float(pos.top1_auc),
                "prefix_pos/top1_auc_correct_val": float(pos.top1_auc_correct),
            },
            on_epoch=True,
            sync_dist=True,
        )
        self._write_prefix_tables("prefix_pos_val", pos, kmet)
        self._write_length_tables(
            "length_metrics_val",
            self.val_length_stats.counts,
            self.val_length_stats.score_sums,
            self.val_length_stats.score_counts,
            diversity_by_len,
            self.val_length_stats.log_pterm_sums,
            self.val_length_stats.log_pterm_counts,
        )
        self._write_length_tables(
            "length_metrics_val_valid",
            self.val_length_stats.counts_valid,
            self.val_length_stats.score_sums_valid,
            self.val_length_stats.score_counts_valid,
            diversity_by_len_valid,
            {},
            {},
        )

        samples_dir = os.path.join(self.trainer.default_root_dir, "validation_samples")
        os.makedirs(samples_dir, exist_ok=True)
        self._samples_dataframe(self.val_samples_table).to_csv(
            os.path.join(samples_dir, f"samples_val_probes_{self.trainer.global_step}.csv"),
            index=False,
        )

        self._log_topk_sequence_metrics("val", self.val_decoded_seqs_and_scores)

        if self.val_log_rs and self.val_log_pfss:
            log_rs = torch.cat(self.val_log_rs)
            log_pfss = torch.cat(self.val_log_pfss)
            if log_rs.numel() > 0 and log_pfss.numel() > 0:
                self._log_metrics(
                    {"val/Var(logR - logPf(s))": (log_rs - log_pfss).var()},
                    sync_dist=True,
                )

    def on_test_epoch_start(self) -> None:
        """Reset the per-epoch test accumulators."""
        self.test_samples_ids.clear()
        self.test_samples_table.clear()
        self.test_log_rs.clear()
        self.test_log_pfss.clear()
        self.test_samples_valid_flags.clear()
        self.test_length_stats.clear()
        self.test_diversity_stats.clear()
        self.test_decoded_seqs_and_scores.clear()

    def on_test_epoch_end(self) -> None:
        """Aggregate the reported test metrics and write them to CSV and JSON."""
        diversity = self._calculate_diversity_ragged(self.test_samples_ids)
        self._log_metrics({"test/diversity": diversity}, sync_dist=True, on_epoch=True)
        valid_test_sequences = self._filter_valid_sequences(
            self.test_samples_ids, self.test_samples_valid_flags
        )
        diversity_test_valid = self._calculate_diversity_ragged(valid_test_sequences)
        self._log_metrics(
            {"test/diversity_valid": diversity_test_valid}, sync_dist=True, on_epoch=True
        )

        stats = self.test_diversity_stats
        batch_mean = stats.mean(stats.batch_sum, stats.batch_count)
        if batch_mean is not None:
            self._log_metrics(
                {"test/diversity_batch_mean": batch_mean}, sync_dist=True, on_epoch=True
            )
        text_mean = stats.mean(stats.text_sum, stats.text_count)
        if self.sequence_diversity is not None and text_mean is not None:
            self._log_metrics(
                {"test/diversity_text_batch_mean": text_mean}, sync_dist=True, on_epoch=True
            )
        mean_internal = stats.mean(stats.fp_internal_sum, stats.fp_internal_count)
        if mean_internal is not None:
            self._log_metrics(
                {"test/fp_div_internal_valid_batch_mean": mean_internal},
                sync_dist=True,
                on_epoch=True,
            )
        mean_topk = stats.mean(stats.fp_topk_sum, stats.fp_topk_count)
        if mean_topk is not None:
            self._log_metrics(
                {"test/fp_div_topk_valid_batch_mean": mean_topk},
                sync_dist=True,
                on_epoch=True,
            )

        diversity_by_len = calculate_diversity_by_length(
            self.test_samples_ids, self.end_of_sentence_token_id
        )
        diversity_by_len_valid = calculate_diversity_by_length(
            valid_test_sequences, self.end_of_sentence_token_id
        )
        self._log_text_diversity_epoch("test", self.test_samples_ids)

        fp_div = self._compute_global_fp_diversity(self.test_samples_ids)
        if fp_div:
            self._log_metrics(
                {
                    "test/fp_div_internal_valid": fp_div["fp_div_internal_valid"],
                    "test/fp_div_topk_valid": fp_div["fp_div_topk_valid"],
                },
                sync_dist=True,
                on_epoch=True,
            )

        self._write_length_tables(
            "length_metrics_test",
            self.test_length_stats.counts,
            self.test_length_stats.score_sums,
            self.test_length_stats.score_counts,
            diversity_by_len,
            self.test_length_stats.log_pterm_sums,
            self.test_length_stats.log_pterm_counts,
        )
        self._write_length_tables(
            "length_metrics_test_valid",
            self.test_length_stats.counts_valid,
            self.test_length_stats.score_sums_valid,
            self.test_length_stats.score_counts_valid,
            diversity_by_len_valid,
            {},
            {},
        )

        pos, kmet = self._compute_prefix_collapse(
            self.test_samples_ids,
            self.test_samples_valid_flags,
            k_list=list(
                range(
                    self.constraint_config.min_sentence_len,
                    self.constraint_config.max_sentence_len + 1,
                )
            ),
        )
        self._log_metrics(
            {
                "prefix_pos/top1_auc_test": float(pos.top1_auc),
                "prefix_pos/top1_auc_correct_test": float(pos.top1_auc_correct),
            },
            on_epoch=True,
            sync_dist=True,
        )
        self._write_prefix_tables("prefix_pos_test", pos, kmet)

        self._log_topk_sequence_metrics("test", self.test_decoded_seqs_and_scores)

        samples_dir = self._repeat_dir("test_samples")
        os.makedirs(samples_dir, exist_ok=True)
        self._samples_dataframe(self.test_samples_table).to_csv(
            os.path.join(
                samples_dir,
                f"samples_test_{self.trainer.global_step}{self._test_repeat_suffix()}.csv",
            ),
            index=False,
        )

        if self.test_log_rs and self.test_log_pfss:
            log_rs = torch.cat(self.test_log_rs)
            log_pfss = torch.cat(self.test_log_pfss)
            if log_rs.numel() > 0 and log_pfss.numel() > 0:
                self._log_metrics(
                    {"test/Var(logR - logPf(s))": (log_rs - log_pfss).var()},
                    sync_dist=True,
                )

        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return
        self._write_test_metrics_json(
            fp_div, diversity_by_len, diversity_test_valid, diversity_by_len_valid
        )

    # ------------------------------------------------------------------ #
    # Sampling utilities
    # ------------------------------------------------------------------ #

    def sample_probes(self, probes: Sequence[dict[str, Any]], n_samples: int = 4) -> pd.DataFrame:
        """Sample from a handful of prompts and return the decoded sequences.

        Args:
            probes: Prompt items to sample from.
            n_samples: Number of samples drawn per prompt.

        Returns:
            DataFrame with one row per sample and its per-step log-probability tables.
        """
        samples = []
        device = self.device
        for probe in probes:
            encoded_prompt = probe["encoded_prompt"]
            prompt_len = encoded_prompt.shape[-1]
            for key, value in probe.items():
                if isinstance(value, torch.Tensor):
                    probe[key] = value.to(device, non_blocking=True)
            with torch.no_grad():
                result_dict = self.forward(
                    probe,
                    n_samples=n_samples,
                    pf_temperature=1.0,
                    reward_temperature=1.0,
                    scaling_factor=self.get_scaling_factor_at_step(self.global_step),
                    reference_logits_scale=self.get_reference_logits_scale_at_step(
                        self.global_step
                    ),
                )
            generated_text = result_dict["state"]
            log_r = result_dict["log_r"]
            log_r_unpenalized = result_dict["log_r_unpenalized"]
            log_pf = result_dict["log_pf"]
            log_pterm = result_dict["log_pterm"]
            log_pf_ref = result_dict["log_pf_ref"]
            log_pterm_ref = result_dict["log_pterm_ref"]

            generated_sequences = self._decoded_sequences(result_dict, generated_text, prompt_len)

            for idx, sequence in enumerate(generated_sequences):
                samples.append(
                    {
                        "Sampled sentence": sequence,
                        "log_pf": log_pf[idx].tolist(),
                        "log_pterm": log_pterm[idx].tolist(),
                        "log_pf_ref": log_pf_ref[idx].tolist() if log_pf_ref is not None else [],
                        "log_pterm_ref": log_pterm_ref[idx].tolist()
                        if log_pterm_ref is not None
                        else [],
                        "log_r": log_r[idx].tolist(),
                        "log_r_unpenalized": log_r_unpenalized[idx].tolist(),
                    }
                )
        return pd.DataFrame(samples)

    def generate_from_replay_buffer(
        self, item: dict[str, Any], encoded_prompt: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]] | None:
        """Re-evaluate replay-buffer trajectories under the current policy.

        Args:
            item: Data item dictionary for the current prompt.
            encoded_prompt: Encoded prompt tensor.

        Returns:
            ``(action_seq, result_dict)`` when the buffer had samples, ``None`` otherwise.
        """
        prompt_tensor = encoded_prompt if encoded_prompt.ndim == 2 else encoded_prompt.unsqueeze(0)
        device = encoded_prompt.device

        buffer_sentences, _ = self.reward_buffer.sample(
            self.training_mixed_config.n_samples,
            prompt_tensor,
            self.tokenizer,
        )
        if buffer_sentences is None:
            return None

        buffer_sentences = buffer_sentences.to(device, non_blocking=True)
        prompt_expanded = prompt_tensor.expand(buffer_sentences.size(0), -1)
        action_seq = torch.cat([prompt_expanded[:, :-1], buffer_sentences], dim=1)
        result_dict = self.forward(
            item,
            action_seq=action_seq,
            pf_temperature=1.0,  # replayed trajectories are evaluated untempered
            reward_temperature=self.reward.temperature,
            scaling_factor=self.get_scaling_factor_at_step(self.global_step),
            reference_logits_scale=self.get_reference_logits_scale_at_step(self.global_step),
        )
        return action_seq, result_dict

    # ------------------------------------------------------------------ #
    # Lightning plumbing
    # ------------------------------------------------------------------ #

    def configure_optimizers(self) -> dict[str, Any]:
        """Build the optimizer and, when configured, the learning-rate scheduler."""
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    # --------------------------------------------------------------------- #
    # Setup helpers
    # --------------------------------------------------------------------- #

    def _load_token_masks(self) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Load the legal-token list and build the legal/illegal vocabulary masks."""
        tokens_path = getattr(self.constraint_config, "legal_tokens", None)
        if not tokens_path:
            raise ValueError(
                "constraint_config.legal_tokens is required: it must point to the file listing "
                "the tokens this task is allowed to emit."
            )
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(
                f"Legal token list not found at {tokens_path!r}. Set "
                "constraint_config.legal_tokens to an existing token list (see assets/token_list)."
            )
        return prepare_token_mask(self.tokenizer, tokens_path)

    def _build_grammar_processor(self) -> GrammarIncrementalLogitsProcessorGeneral:
        """Build the incremental grammar-constrained logits processor for this task."""
        processor_type = getattr(self.constraint_config, "processor_type", "prefix")
        if processor_type != "prefix":
            raise ValueError(
                f"Unsupported constraint_config.processor_type {processor_type!r}; "
                "the released tasks all use the incremental 'prefix' processor."
            )
        with open(self.constraint_config.grammar_path) as file:
            grammar_str = file.read()
        return GrammarIncrementalLogitsProcessorGeneral(
            parse_ebnf(grammar_str),
            tokenizer=self.tokenizer,
            nice_token_ids_list=self.legal_token_ids_list,
            execution_mode=self.constraint_config.parse_mode,
        )

    # --------------------------------------------------------------------- #
    # Loss helpers
    # --------------------------------------------------------------------- #

    def _compute_loss(
        self,
        *,
        generated_text: torch.Tensor,
        log_pf: torch.Tensor,
        log_pterm: torch.Tensor,
        log_r: torch.Tensor,
        log_pf_ref: torch.Tensor | None,
        log_pterm_ref: torch.Tensor | None,
        prompt_len: int,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the configured GFlowNet objective on one batch of trajectories."""
        return self.loss_fn(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=prompt_len,
            ref_log_pf=log_pf_ref,
            ref_log_pterm=log_pterm_ref,
            ref_scale=1.0,
            max_prefix_len=int(self.get_prefix_len_at_step(self.global_step))
            if self.get_prefix_len_at_step is not None
            else None,
            k_min=int(self.get_k_min_at_step(self.global_step))
            if self.get_k_min_at_step is not None
            else None,
        )

    def _log_loss_components(
        self,
        stage: str,
        loss_output: dict[str, torch.Tensor],
        *,
        on_step: bool,
        prog_bar: bool = False,
    ) -> None:
        """Log every auxiliary quantity the loss reported alongside its scalar value."""
        components = {
            f"{stage}/{key}": value for key, value in loss_output.items() if key != "loss"
        }
        if components:
            self._log_metrics(components, on_step=on_step, sync_dist=True, prog_bar=prog_bar)

    # --------------------------------------------------------------------- #
    # Sampling helpers
    # --------------------------------------------------------------------- #

    def _eval_pf_temperature(self) -> float:
        """Return the sampling temperature used for validation and test generation."""
        pf_temp_eval = getattr(self.training_mixed_config, "pf_temp_eval", None)
        if pf_temp_eval is None:
            pf_temp_eval = getattr(self.training_mixed_config, "pf_temp_high", 1.0)
        return float(pf_temp_eval)

    def _sample_pf_temperature(self) -> float:
        """Draw the forward-policy temperature for one training step."""
        if self.get_pf_temp_low_at_step is None or self.get_pf_temp_high_at_step is None:
            pf_low = self.training_mixed_config.pf_temp_low
            pf_high = self.training_mixed_config.pf_temp_high
        else:
            pf_low = self.get_pf_temp_low_at_step(self.global_step)
            pf_high = self.get_pf_temp_high_at_step(self.global_step)

        if random.random() >= self.training_mixed_config.pf_temp_prob:
            return 1.0
        if pf_high < pf_low:
            pf_low, pf_high = pf_high, pf_low
        if pf_high == pf_low:
            return pf_high
        return random.random() * (pf_high - pf_low) + pf_low

    def _decode_generated_tokens(
        self, tokens: torch.Tensor, skip_special_tokens: bool = True
    ) -> str:
        """Decode one generated token sequence to text."""
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _decoded_sequences(
        self, result_dict: dict[str, Any], generated_text: torch.Tensor, prompt_len: int
    ) -> list[str]:
        """Return the generated strings, preferring the task-specific decoding when present."""
        full_tokens = result_dict.get("full_tokens", None)
        if full_tokens is not None:
            return full_tokens
        return [
            self._decode_generated_tokens(text[prompt_len:], skip_special_tokens=False)
            for text in generated_text
        ]

    # --------------------------------------------------------------------- #
    # Metric accumulation helpers
    # --------------------------------------------------------------------- #

    def _validator_metrics(self, tokens: torch.Tensor, scaffold: Any) -> dict[str, Any]:
        """Score a batch of generated sequences with the task validator."""
        validator = self.reward.sentence_validator
        if validator is None:
            return {}
        return validator.accuracy(tokens, self.tokenizer, scaffold, return_hist=True)

    def _accumulate_length_statistics(
        self,
        stats: LengthStatistics,
        tokens: torch.Tensor,
        validator_metric_dict: dict[str, Any],
        validator_dict: dict[str, Any] | None,
        valid_flags: list[bool] | None,
    ) -> None:
        """Accumulate per-length counts and scores, over all samples and over valid ones.

        Length histograms reported by the validator take precedence over lengths recovered
        from the token ids, because the validator knows how to strip task-specific padding.
        """
        scores = validator_dict.get("global_score") if validator_dict is not None else None
        len_hist = (
            validator_metric_dict.pop("len_tok_hist", None) if validator_metric_dict else None
        )
        score_hist = (
            validator_metric_dict.pop("score_hist", None) if validator_metric_dict else None
        )
        if isinstance(len_hist, list):
            lengths = len_hist
            length_scores = score_hist if isinstance(score_hist, list) else scores
        else:
            lengths = self._lengths_from_tokens(tokens)
            length_scores = scores

        self._accumulate_length_stats(
            lengths, length_scores, stats.counts, stats.score_sums, stats.score_counts
        )
        if valid_flags is not None:
            self._accumulate_length_stats(
                lengths,
                length_scores,
                stats.counts_valid,
                stats.score_sums_valid,
                stats.score_counts_valid,
                valid_flags=valid_flags,
            )

    @staticmethod
    def _accumulate_length_stats(
        lengths: list[int],
        scores: list[float] | torch.Tensor | None,
        counts: dict[int, int],
        score_sums: dict[int, float],
        score_counts: dict[int, int],
        valid_flags: list[bool] | None = None,
    ) -> None:
        """Add one batch of (length, score) pairs to the given accumulators."""
        if not lengths:
            return
        if isinstance(scores, torch.Tensor):
            scores_list = scores.detach().cpu().tolist()
        elif isinstance(scores, list):
            scores_list = scores
        else:
            scores_list = None

        n = len(lengths)
        if scores_list is not None:
            n = min(n, len(scores_list))
        if valid_flags is not None:
            n = min(n, len(valid_flags))

        for idx in range(n):
            if valid_flags is not None and not valid_flags[idx]:
                continue
            length_int = int(lengths[idx])
            counts[length_int] = counts.get(length_int, 0) + 1
            if scores_list is not None:
                score_sums[length_int] = score_sums.get(length_int, 0.0) + float(scores_list[idx])
                score_counts[length_int] = score_counts.get(length_int, 0) + 1

    def _accumulate_log_pterm_by_length(
        self,
        tokens: torch.Tensor,
        log_pterm: torch.Tensor,
        stats: LengthStatistics,
    ) -> None:
        """Accumulate the termination log-probability at the sampled stop step, by length."""
        if log_pterm is None or tokens is None:
            return
        lengths = self._lengths_from_tokens(tokens)
        if not lengths:
            return
        log_pterm_cpu = log_pterm.detach().cpu()
        t_len = int(log_pterm_cpu.shape[1])
        if t_len <= 0:
            return
        n = min(len(lengths), log_pterm_cpu.shape[0])
        for idx in range(n):
            length = int(lengths[idx])
            t_idx = length if length < t_len else t_len - 1
            value = float(log_pterm_cpu[idx, t_idx].item())
            stats.log_pterm_sums[length] = stats.log_pterm_sums.get(length, 0.0) + value
            stats.log_pterm_counts[length] = stats.log_pterm_counts.get(length, 0) + 1

    @staticmethod
    def _pop_fp_diversity(metrics: dict[str, Any], stats: DiversityStatistics) -> None:
        """Move the validator's fingerprint-diversity entries into the epoch accumulators."""
        if not metrics:
            return
        internal = metrics.pop("fp_div_internal_valid", None)
        topk = metrics.pop("fp_div_topk_valid", None)
        if internal is not None:
            stats.fp_internal_sum += float(internal)
            stats.fp_internal_count += 1
        if topk is not None:
            stats.fp_topk_sum += float(topk)
            stats.fp_topk_count += 1

    def _accumulate_text_diversity(
        self, stage: str, tokens: torch.Tensor, stats: DiversityStatistics
    ) -> None:
        """Log and accumulate embedding-based text diversity for one batch."""
        if self.sequence_diversity is None:
            return
        decoded_batch = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        decoded_batch = [
            text.replace(self.tokenizer.eos_token, "").strip() for text in decoded_batch
        ]
        if len(decoded_batch) <= 1:
            return
        text_div = self.sequence_diversity(decoded_batch)
        if text_div is None:
            return
        stats.text_sum += float(text_div)
        stats.text_count += 1
        self._log_metrics(
            {f"{stage}/diversity_text": text_div},
            on_step=False,
            sync_dist=True,
        )

    def _log_text_diversity_epoch(self, stage: str, sequences: list[list[int]]) -> None:
        """Log embedding-based text diversity over all sequences seen this epoch."""
        if self.sequence_diversity is None:
            return
        decoded_all = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences]
        if len(decoded_all) <= 1:
            return
        text_div_epoch = self.sequence_diversity(decoded_all)
        if text_div_epoch is None:
            return
        self._log_metrics(
            {f"{stage}/diversity_text_epoch": text_div_epoch},
            sync_dist=True,
            on_epoch=True,
        )

    def _log_topk_sequence_metrics(
        self, stage: str, decoded_seqs_and_scores: list[tuple[str, float]]
    ) -> None:
        """Log top-k score, Levenshtein diversity and novelty for sequence tasks (AMP)."""
        if not decoded_seqs_and_scores:
            return
        try:
            from chemgfn.utils.sequence_metrics import (
                levenshtein_diversity,
                levenshtein_novelty,
                select_topk,
            )
        except ImportError:
            decoded_seqs_and_scores.clear()
            return

        sequences = [seq for seq, _ in decoded_seqs_and_scores]
        scores = [score for _, score in decoded_seqs_and_scores]
        validator = self.reward.sentence_validator
        topk_seqs, topk_scores = select_topk(sequences, scores, k=getattr(validator, "topk", 100))
        if topk_seqs:
            novelty = 0.0
            training_seqs = getattr(validator, "_training_sequences", None)
            if training_seqs:
                novelty = levenshtein_novelty(topk_seqs, training_seqs)
            self._log_metrics(
                {
                    f"{stage}/topk_performance": sum(topk_scores) / len(topk_scores),
                    f"{stage}/topk_diversity": levenshtein_diversity(topk_seqs)
                    if len(topk_seqs) >= 2
                    else 0.0,
                    f"{stage}/topk_novelty": novelty,
                },
                sync_dist=True,
                on_epoch=True,
            )
        decoded_seqs_and_scores.clear()

    def _record_samples(
        self,
        samples_table: list[dict[str, Any]],
        decoded_seqs_and_scores: list[tuple[str, float]],
        *,
        result_dict: dict[str, Any],
        generated_text: torch.Tensor,
        prompt_len: int,
        log_pf: torch.Tensor,
        log_pterm: torch.Tensor,
        log_pf_ref: torch.Tensor | None,
        log_pterm_ref: torch.Tensor | None,
        log_r: torch.Tensor,
        log_r_unpenalized: torch.Tensor,
        valid_flags: list[bool] | None,
    ) -> None:
        """Store decoded samples and their per-step log-probability tables for later export."""
        generated_sequences = self._decoded_sequences(result_dict, generated_text, prompt_len)

        validator_dict = result_dict.get("validator_dict")
        if validator_dict is not None and "global_score" in validator_dict:
            global_scores = validator_dict["global_score"]
            for idx, sequence in enumerate(generated_sequences):
                decoded_seqs_and_scores.append((sequence, float(global_scores[idx].item())))

        if valid_flags is None:
            valid_flags = [None] * len(generated_sequences)

        for idx, sequence in enumerate(generated_sequences):
            raw_ids = generated_text[idx, prompt_len:].detach().cpu().tolist()
            samples_table.append(
                {
                    "Sampled sentence": sequence,
                    "token_ids": self._strip_special_token_ids(raw_ids),
                    "is_valid": valid_flags[idx] if idx < len(valid_flags) else None,
                    "log_pf": log_pf[idx].detach().cpu().tolist(),
                    "log_pterm": log_pterm[idx].detach().cpu().tolist(),
                    "log_pf_ref": log_pf_ref[idx].detach().cpu().tolist()
                    if log_pf_ref is not None
                    else [],
                    "log_pterm_ref": log_pterm_ref[idx].detach().cpu().tolist()
                    if log_pterm_ref is not None
                    else [],
                    "log_r": log_r[idx].detach().cpu().tolist(),
                    "log_r_unpenalized": log_r_unpenalized[idx].detach().cpu().tolist(),
                }
            )

    # --------------------------------------------------------------------- #
    # Sequence helpers
    # --------------------------------------------------------------------- #

    def _strip_special_token_ids(self, token_ids: list[int]) -> list[int]:
        """Drop all tokenizer special tokens from a decoded id list."""
        special_ids = set(self.tokenizer.all_special_ids)
        return [tok for tok in token_ids if tok not in special_ids]

    def _strip_eos_token_ids(self, token_ids: list[int]) -> list[int]:
        """Truncate a token id list at the first termination token."""
        eos_id = int(self.end_of_sentence_token_id)
        try:
            eos_pos = token_ids.index(eos_id)
        except ValueError:
            return token_ids
        return token_ids[:eos_pos]

    def _strip_eos_from_batch(self, token_batch: torch.Tensor) -> list[list[int]]:
        """Convert a padded token batch into ragged pre-EOS id lists."""
        if token_batch is None:
            return []
        rows = token_batch.detach().cpu().tolist()
        return [self._strip_eos_token_ids([int(t) for t in row]) for row in rows]

    def _lengths_from_tokens(self, tokens: torch.Tensor) -> list[int]:
        """Return the pre-EOS length of every sequence in a padded token batch."""
        if tokens is None or tokens.numel() == 0:
            return []
        eos_id = int(self.end_of_sentence_token_id)
        tok_cpu = tokens.detach().cpu()
        eos_mask = tok_cpu.eq(eos_id)
        has_eos = eos_mask.any(dim=1)
        first_eos = eos_mask.float().argmax(dim=1)
        full_len = tok_cpu.shape[1]
        lengths = torch.where(has_eos, first_eos, torch.full_like(first_eos, full_len)).tolist()
        return [int(x) for x in lengths]

    def _mean_token_length(self, tokens: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        """Mean pre-EOS length of a batch, computed directly from token ids."""
        if tokens is None or tokens.numel() == 0:
            return torch.tensor(0.0, device=device)
        eos_mask = tokens.eq(int(self.end_of_sentence_token_id))
        has_eos = eos_mask.any(dim=1)
        first_eos = eos_mask.float().argmax(dim=1)
        full_len = int(tokens.shape[1])
        lengths_tok = torch.where(
            has_eos, first_eos, first_eos.new_full(first_eos.shape, full_len)
        ).to(dtype=torch.float32)
        return lengths_tok.mean()

    @staticmethod
    def _filter_valid_sequences(
        sequences: list[list[int]], valid_flags: list[bool] | torch.Tensor | None
    ) -> list[list[int]]:
        """Keep only the sequences the validator accepted."""
        if not sequences or valid_flags is None:
            return []
        if isinstance(valid_flags, torch.Tensor):
            valid_flags = valid_flags.detach().cpu().tolist()
        if not valid_flags:
            return []
        n = min(len(sequences), len(valid_flags))
        return [sequences[i] for i in range(n) if bool(valid_flags[i])]

    def _get_valid_flags(
        self,
        validator_dict: dict[str, Any] | None,
        generated_text: torch.Tensor,
        prompt_len: int,
    ) -> list[bool] | None:
        """Derive a per-sample validity flag from the validator output."""
        if not validator_dict:
            return None

        invalid = validator_dict.get("invalid")
        if isinstance(invalid, torch.Tensor):
            eos = self.end_of_sentence_token_id
            tokens = generated_text[:, prompt_len:].detach().cpu()
            invalid_cpu = invalid.detach().cpu()
            seq_len = tokens.shape[1]
            flags = []
            for i in range(tokens.shape[0]):
                row = tokens[i].tolist()
                try:
                    eos_pos = row.index(eos)
                except ValueError:
                    eos_pos = seq_len
                prefix_len = min(eos_pos, invalid_cpu.shape[1] - 1)
                flags.append(bool(invalid_cpu[i, prefix_len].item() == 0))
            return flags

        global_score = validator_dict.get("global_score")
        if isinstance(global_score, torch.Tensor):
            return (global_score > 0).detach().cpu().tolist()
        if isinstance(global_score, (list, tuple)):
            return [float(x) > 0 for x in global_score]

        return None

    # --------------------------------------------------------------------- #
    # Diversity helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _calculate_diversity_ragged(sequences: list[list[int]]) -> float:
        """Ragged token entropy: mean per-position entropy over surviving sequences."""
        if not sequences or len(sequences) <= 1:
            return 0.0
        max_len = max((len(seq) for seq in sequences), default=0)
        if max_len <= 0:
            return 0.0
        total_entropy = 0.0
        used = 0
        for pos in range(max_len):
            toks = [seq[pos] for seq in sequences if len(seq) > pos]
            n = len(toks)
            if n <= 1:
                continue
            _, counts = np.unique(np.array(toks, dtype=np.int64), return_counts=True)
            probs = counts.astype(np.float64) / float(n)
            total_entropy += float(-np.sum(probs * np.log(probs + 1e-10)))
            used += 1
        return total_entropy / float(used) if used > 0 else 0.0

    def _compute_global_fp_diversity(self, token_ids_list: list[list[int]]) -> dict[str, float]:
        """Fingerprint diversity over every sequence of an epoch (SMILES tasks only)."""
        validator = getattr(self.reward, "sentence_validator", None)
        if validator is None:
            return {}
        if not hasattr(validator, "_morgan_fp") or not hasattr(
            validator, "_mean_pairwise_tanimoto"
        ):
            return {}
        if not token_ids_list:
            return {}

        eos_id = int(self.end_of_sentence_token_id)
        seq_tensors = []
        for seq in token_ids_list:
            cleaned = self._strip_eos_token_ids(self._strip_special_token_ids(seq))
            if not cleaned:
                cleaned = [eos_id]
            seq_tensors.append(torch.tensor(cleaned, dtype=torch.long))

        if not seq_tensors:
            return {}

        tokens = torch.nn.utils.rnn.pad_sequence(
            seq_tensors,
            batch_first=True,
            padding_value=eos_id,
        )
        try:
            metrics = validator.accuracy(tokens, self.tokenizer, return_hist=False)
        except Exception:
            return {}

        return {
            "fp_div_internal_valid": float(metrics.get("fp_div_internal_valid", 0.0)),
            "fp_div_topk_valid": float(metrics.get("fp_div_topk_valid", 0.0)),
        }

    def _compute_prefix_collapse(
        self,
        sequences: list[list[int]],
        valid_flags: list[bool],
        k_list: list[int],
    ) -> tuple[PrefixCollapseResult, dict[int, dict[str, float]]]:
        """Compute the prefix-collapse diagnostics (Surv / PefEnt / Top1) for an epoch."""
        eos = self.end_of_sentence_token_id
        seqs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in sequences],
            batch_first=True,
            padding_value=eos,
        )
        mask_noeos = compute_active_before(seqs, eos=eos) & (seqs != eos)
        seqs_list = seqs.detach().cpu().tolist()
        mask_list = mask_noeos.detach().cpu().tolist()
        invalid_flags = (~torch.tensor(valid_flags)).tolist()

        pos = prefix_collapse_by_position(
            seqs_list, mask_list, collapse_thr=0.95, invalid=invalid_flags
        )
        kmet = prefix_collapse_by_k(seqs_list, mask_list, k_list=k_list, invalid=invalid_flags)
        return pos, kmet

    # --------------------------------------------------------------------- #
    # Output helpers
    # --------------------------------------------------------------------- #

    def _test_repeat_suffix(self) -> str:
        """Suffix identifying the current evaluation repeat, empty for a single run."""
        return getattr(self, "test_repeat_suffix", "") or ""

    def _repeat_dir(self, base_name: str) -> str:
        """Repeat-scoped output directory under ``trainer.default_root_dir``."""
        suffix = self._test_repeat_suffix()
        name = f"{base_name}{suffix}" if suffix else base_name
        return os.path.join(self.trainer.default_root_dir, name)

    @staticmethod
    def _samples_dataframe(samples_table: list[dict[str, Any]]) -> pd.DataFrame:
        """Build the sample-dump DataFrame, with a stable schema when no sample was kept."""
        if samples_table:
            return pd.DataFrame(samples_table)
        return pd.DataFrame(
            columns=[
                "Sampled sentence",
                "token_ids",
                "is_valid",
                "log_pf",
                "log_pterm",
                "log_pf_ref",
                "log_pterm_ref",
                "log_r",
                "log_r_unpenalized",
            ]
        )

    def _write_prefix_tables(
        self,
        tag: str,
        pos: PrefixCollapseResult,
        kmet: dict[int, dict[str, float]],
    ) -> None:
        """Write the prefix-collapse tables (by position and by depth k) as CSV.

        These files back the Surv / PefEnt / Top1 versus depth curves; both the all-sample
        and the correct-only variants are written when available.
        """
        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return

        epoch = int(getattr(self.trainer, "current_epoch", 0))
        root_dir = self._repeat_dir("prefix_tables")
        os.makedirs(root_dir, exist_ok=True)
        local_tag = tag.replace("/", "_")

        def write_position_table(
            suffix: str,
            top1_mass: list[float],
            entropy: list[float],
            eff_support: list[float],
            unique: list[int] | None,
            support: list[int] | None,
            correct_frac: list[float] | None,
        ) -> None:
            n_positions = min(len(top1_mass), len(entropy), len(eff_support))
            if n_positions <= 0:
                return
            columns = ["epoch", "t", "top1_mass", "entropy", "eff_support"]
            if unique is not None:
                columns.append("unique")
            if support is not None:
                columns.append("support")
            if correct_frac is not None:
                columns.append("correct_frac")

            rows = []
            for t in range(n_positions):
                row: list[Any] = [
                    epoch,
                    int(t),
                    float(top1_mass[t]),
                    float(entropy[t]),
                    float(eff_support[t]),
                ]
                if unique is not None:
                    row.append(int(unique[t]) if t < len(unique) else 0)
                if support is not None:
                    row.append(int(support[t]) if t < len(support) else 0)
                if correct_frac is not None:
                    row.append(float(correct_frac[t]) if t < len(correct_frac) else 0.0)
                rows.append(row)

            pd.DataFrame(rows, columns=columns).to_csv(
                os.path.join(root_dir, f"{local_tag}_pos{suffix}_{epoch}.csv"), index=False
            )

        def write_k_table(suffix: str) -> None:
            ks = sorted(kmet.keys())
            if not ks:
                return
            if suffix == "":
                fields = ("top1", "top5", "entropy", "eff", "n", "unique")
            else:
                fields = (
                    "top1_correct",
                    "top5_correct",
                    "entropy_correct",
                    "eff_correct",
                    "n_correct",
                    "unique_correct",
                )
                probe_fields = (fields[4], fields[0], fields[1])  # n, top1, top5
                if not any(
                    field_name in (kmet[k] or {}) for k in ks for field_name in probe_fields
                ):
                    return

            rows = []
            for k in ks:
                values = kmet[k] or {}
                rows.append([epoch, int(k)] + [float(values.get(f, 0.0)) for f in fields])
            pd.DataFrame(
                rows, columns=["epoch", "k", "top1", "top5", "entropy", "eff", "n", "unique"]
            ).to_csv(os.path.join(root_dir, f"{local_tag}_k{suffix}_{epoch}.csv"), index=False)

        if getattr(pos, "top1_mass", None):
            write_position_table(
                "",
                [float(v) for v in pos.top1_mass],
                [float(v) for v in getattr(pos, "entropy", [])],
                [float(v) for v in getattr(pos, "eff_support", [])],
                getattr(pos, "unique", None),
                getattr(pos, "support", None),
                None,
            )
        if getattr(pos, "top1_mass_correct", None):
            write_position_table(
                "_correct",
                [float(v) for v in pos.top1_mass_correct],
                [float(v) for v in getattr(pos, "entropy_correct", [])],
                [float(v) for v in getattr(pos, "eff_support_correct", [])],
                getattr(pos, "unique_correct", None),
                getattr(pos, "support_correct", None),
                getattr(pos, "correct_frac", None),
            )
        if kmet:
            write_k_table("")
            write_k_table("_correct")

    def _write_length_tables(
        self,
        tag: str,
        length_counts: dict[int, int],
        score_sums: dict[int, float],
        score_counts: dict[int, int],
        diversity_by_len: dict[int, float],
        log_pterm_sums: dict[int, float],
        log_pterm_counts: dict[int, int],
    ) -> None:
        """Write the length histogram and the per-length score, diversity and termination stats."""
        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return
        lengths = sorted(
            set(length_counts) | set(diversity_by_len) | set(log_pterm_sums) | set(score_sums)
        )
        if not lengths:
            return

        scorer_name = "score"
        validator = getattr(self.reward, "sentence_validator", None)
        if validator is not None:
            scorer_name = getattr(validator, "scorer_name", scorer_name)

        epoch = int(getattr(self.trainer, "current_epoch", 0))
        total = float(sum(length_counts.values()))
        rows = []
        for length in lengths:
            count = int(length_counts.get(length, 0))
            score_count = int(score_counts.get(length, 0))
            pterm_count = int(log_pterm_counts.get(length, 0))
            log_pterm = (
                float(log_pterm_sums.get(length, 0.0)) / pterm_count if pterm_count > 0 else 0.0
            )
            rows.append(
                [
                    epoch,
                    int(length),
                    count,
                    (count / total * 100.0) if total > 0 else 0.0,
                    float(score_sums.get(length, 0.0)) / score_count if score_count > 0 else 0.0,
                    float(diversity_by_len.get(length, 0.0)),
                    log_pterm,
                    float(np.exp(log_pterm)),
                ]
            )

        root_dir = self._repeat_dir("length_metrics")
        os.makedirs(root_dir, exist_ok=True)
        pd.DataFrame(
            rows,
            columns=[
                "epoch",
                "length",
                "count",
                "count_percent",
                scorer_name,
                "diversity",
                "log_pterm",
                "pterm",
            ],
        ).to_csv(os.path.join(root_dir, f"{tag.replace('/', '_')}_{epoch}.csv"), index=False)

    def _write_test_metrics_json(
        self,
        fp_div: dict[str, float],
        diversity_by_len: dict[int, float],
        diversity_valid: float,
        diversity_by_len_valid: dict[int, float],
    ) -> None:
        """Write every reported terminal metric of the test epoch to a JSON file.

        The file carries the logged ``test/*`` scalars (Acc, Score, FPDiv, ...) together with
        the length-conditioned counts, scores, entropy and termination probabilities that the
        reported Len and log p_term statistics are derived from.
        """
        callback_metrics = getattr(self.trainer, "callback_metrics", {})
        if not callback_metrics:
            return

        metrics: dict[str, Any] = {}
        for key, value in callback_metrics.items():
            if not isinstance(key, str) or not key.startswith("test/"):
                continue
            if isinstance(value, torch.Tensor):
                metrics[key] = (
                    float(value.detach().cpu().item())
                    if value.ndim == 0
                    else value.detach().cpu().tolist()
                )
            elif isinstance(value, (float, int)):
                metrics[key] = float(value)
            elif isinstance(value, bool):
                metrics[key] = bool(value)
        if not metrics:
            return

        epoch = int(getattr(self.trainer, "current_epoch", 0))
        global_step = int(getattr(self.trainer, "global_step", 0))
        metrics["epoch"] = epoch
        metrics["global_step"] = global_step
        if fp_div:
            metrics["test/fp_div_internal_valid"] = fp_div["fp_div_internal_valid"]
            metrics["test/fp_div_topk_valid"] = fp_div["fp_div_topk_valid"]
            metrics["test/validator/fp_div_internal_valid"] = fp_div["fp_div_internal_valid"]
            metrics["test/validator/fp_div_topk_valid"] = fp_div["fp_div_topk_valid"]

        stats = self.test_length_stats
        metrics["len_counts"] = stats.counts
        metrics["score_sum_by_len"] = stats.score_sums
        metrics["score_count_by_len"] = stats.score_counts
        metrics["score_mean_by_len"] = self._mean_by_length(stats.score_sums, stats.score_counts)
        metrics["len_counts_valid"] = stats.counts_valid
        metrics["score_sum_by_len_valid"] = stats.score_sums_valid
        metrics["score_count_by_len_valid"] = stats.score_counts_valid
        metrics["score_mean_by_len_valid"] = self._mean_by_length(
            stats.score_sums_valid, stats.score_counts_valid
        )
        metrics["diversity_by_len"] = diversity_by_len
        metrics["diversity_valid"] = diversity_valid
        metrics["diversity_by_len_valid"] = diversity_by_len_valid
        metrics["log_pterm_sum"] = stats.log_pterm_sums
        metrics["log_pterm_count"] = stats.log_pterm_counts
        log_pterm_by_len = self._mean_by_length(stats.log_pterm_sums, stats.log_pterm_counts)
        metrics["log_pterm_by_len"] = log_pterm_by_len
        metrics["pterm_by_len"] = {
            length: float(np.exp(value)) for length, value in log_pterm_by_len.items()
        }

        exp_name = None
        hparams = getattr(self, "hparams", None)
        if hparams is not None:
            exp_name = (
                hparams.get("exp_name")
                if isinstance(hparams, dict)
                else getattr(hparams, "exp_name", None)
            )
        if not exp_name:
            experiment = getattr(getattr(self, "logger", None), "experiment", None)
            exp_name = getattr(experiment, "name", None)
        exp_name = str(exp_name or "exp").replace(" ", "_")
        metrics["exp_name"] = exp_name

        root_dir = os.path.join(self.trainer.default_root_dir, "json")
        os.makedirs(root_dir, exist_ok=True)
        out_path = os.path.join(
            root_dir,
            f"test_metrics_{exp_name}_epoch_{epoch}_step_{global_step}"
            f"{self._test_repeat_suffix()}.json",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

    @staticmethod
    def _mean_by_length(sums: dict[int, float], counts: dict[int, int]) -> dict[int, float]:
        """Average a per-length sum by its per-length count, reporting 0 for empty buckets."""
        means = {}
        for length, total in sums.items():
            count = counts.get(length, 0)
            means[int(length)] = float(total) / float(count) if count else 0.0
        return means

    # --------------------------------------------------------------------- #
    # Logging helpers
    # --------------------------------------------------------------------- #

    def _prepare_metric(self, value: Any, sync_dist: bool) -> Any:
        """Move a metric onto the module device so it can be reduced across ranks."""
        if not sync_dist:
            return value
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, (float, int)):
            return torch.tensor(value, device=self.device)
        return value

    def _log_metrics(self, metrics: dict[str, Any], **common_kwargs: Any) -> None:
        """Log a batch of metrics; a ``(value, overrides)`` tuple overrides the shared kwargs."""
        for name, value in metrics.items():
            if isinstance(value, tuple):
                metric_value, overrides = value
                kwargs = {**common_kwargs, **overrides}
            else:
                metric_value = value
                kwargs = common_kwargs
            metric_value = self._prepare_metric(metric_value, bool(kwargs.get("sync_dist", False)))
            self.log(name, metric_value, **kwargs)

    def _log_validator_core_metrics(
        self,
        prefix: str,
        validator_metric_dict: dict[str, Any] | None,
        *,
        sync_dist: bool,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
    ) -> None:
        """Log accuracy, task score and fingerprint diversity from the validator output."""
        if not validator_metric_dict:
            return
        scorer_name = None
        validator = getattr(self.reward, "sentence_validator", None) if self.reward else None
        if validator is not None:
            scorer_name = getattr(validator, "scorer_name", None)

        metrics = {}
        for key in ("acc", "fp_div_internal_valid", "fp_div_topk_valid"):
            if key in validator_metric_dict:
                metrics[f"{prefix}/{key}"] = validator_metric_dict[key]
        if scorer_name:
            if scorer_name in validator_metric_dict:
                metrics[f"{prefix}/{scorer_name}"] = validator_metric_dict[scorer_name]
            filter_key = f"{scorer_name}_filter"
            if filter_key in validator_metric_dict:
                metrics[f"{prefix}/{filter_key}"] = validator_metric_dict[filter_key]

        if metrics:
            self._log_metrics(
                metrics,
                sync_dist=sync_dist,
                on_step=on_step,
                on_epoch=on_epoch,
            )

    def _log_validator_full_metrics(
        self,
        prefix: str,
        validator_metric_dict: dict[str, Any] | None,
        *,
        sync_dist: bool,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
    ) -> None:
        """Log every scalar entry of the validator output under ``prefix``."""
        if not validator_metric_dict:
            return
        skip_keys = {
            "len_tok_hist",
            "len_tok_valid_hist",
            "len_char_hist",
            "len_char_valid_hist",
            "score_hist",
        }
        metrics = {}
        for key, value in validator_metric_dict.items():
            if key in skip_keys:
                continue
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    metrics[f"{prefix}/{key}"] = value
                continue
            if isinstance(value, (float, int, bool)):
                metrics[f"{prefix}/{key}"] = value
        if metrics:
            self._log_metrics(
                metrics,
                sync_dist=sync_dist,
                on_step=on_step,
                on_epoch=on_epoch,
            )

    def _log_agreement_metrics(self, agree_list: Any) -> None:
        """Log how often sampled tokens agree with the grammar-constrained proposal."""
        if agree_list is None:
            return
        if isinstance(agree_list, torch.Tensor):
            tensors = [agree_list]
        elif isinstance(agree_list, Sequence):
            if len(agree_list) == 0:
                return
            tensors = list(agree_list)
        else:
            return

        if len(tensors) == 0:
            return

        agree_means = [torch.mean(x.sum(-1).float()) for x in tensors]
        mid_index = len(tensors) // 2

        self._log_metrics(
            {
                "train/agree_mean": torch.stack(agree_means).mean(),
                "train/agree_start": agree_means[0],
                "train/agree_midd": agree_means[mid_index],
                "train/agree_end": agree_means[-1],
            },
            on_step=True,
            sync_dist=True,
        )
