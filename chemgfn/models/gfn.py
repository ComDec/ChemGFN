# type: ignore
# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import json
import logging
import os
import random
import sys
from collections import deque
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
from lightning import LightningModule
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from transformers_cfg.generation.logits_process import (
    GrammarConstrainedLogitsProcessor,
    GrammarIncrementalLogitsProcessorGeneral,
    GrammarIncrementalLogitsProcessorSampleEnhanced,
)
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer

from chemgfn.models.losses import GFNLoss
from chemgfn.utils.cond_var_metrics import (
    compute_raptb_targets,
    compute_subtb_targets_delta,
    compute_tb_targets,
    grouped_weighted_var,
)
from chemgfn.utils.diversity import SequenceDiversity
from chemgfn.utils.gfn_utils import (
    base_to_lora,
    calculate_diversity_by_length,
    generate_and_return_termination_logprob,
    get_termination_vals,
    lora_to_base,
    prepare_token_mask,
)
from chemgfn.utils.phi_utils import compute_active_before
from chemgfn.utils.prefix_metrics import (
    prefix_collapse_by_k,
    prefix_collapse_by_position,
)
from chemgfn.utils.replay_buffer import ReplayBuffer
from chemgfn.utils.schedulers import Scheduler

log = logging.getLogger(__name__)

# Re-export Scheduler for backward compatibility with config files
__all__ = ["ChemGFNModule", "Scheduler"]

sys.setrecursionlimit(1500)


class ChemGFNModule(LightningModule):
    """Main Lightning module wrapping model, reward and optimisation logic.

    This module implements a GFlowNet training loop for chemical molecule generation.
    It handles autoregressive generation with grammar constraints, reward computation,
    and SubTB loss optimization.
    """

    def __init__(
        self,
        net_config: dict[str, Any],
        lora_config: LoraConfig,
        tokenizer: PreTrainedTokenizer,
        reward,
        loss_fn: GFNLoss,
        reward_buffer,
        reward_config: dict[str, Any],
        training_mixed_config: dict[str, Any],
        ema_config: dict[str, Any],
        constraint_config: dict[str, Any],
        optimizer: Callable[[Any], torch.optim.Optimizer],
        scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] | None,
        factor_schedulers: dict[str, Any],
        coverage_config: dict[str, Any] | None = None,
        compile_model: bool | None = None,
        compile: bool | None = None,
        disable_peft: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["net", "loss_fn"])
        self.net_config: Any = net_config
        self.reward_config: Any = reward_config
        self.constraint_config: Any = constraint_config
        self.training_mixed_config: Any = training_mixed_config
        self.ema_config: Any = ema_config

        model = AutoModelForCausalLM.from_pretrained(self.net_config.pretrained_model_name_or_path)
        model.train()

        model_frozen = AutoModelForCausalLM.from_pretrained(
            self.net_config.pretrained_model_name_or_path
        )
        model_frozen.eval()
        model_frozen.requires_grad_(False)

        if not disable_peft:
            self.net = get_peft_model(model, lora_config)
        else:
            self.net = model

        self.net_frozen = model_frozen
        self._net_frozen_raw = model_frozen
        self.tokenizer = tokenizer
        self.disable_peft = disable_peft
        self.lora_config = lora_config

        self.reward_config = reward_config
        self.constraint_config = constraint_config
        self.training_mixed_config = training_mixed_config
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

        # Initialize all schedulers from config
        # Can use either hydra instantiate or direct parameter specification
        # Initialize all factor schedulers
        self.factor_schedulers = factor_schedulers
        self.get_reward_temp_at_step = self.factor_schedulers["reward_temp"]
        self.get_scaling_factor_at_step = self.factor_schedulers["scaling_factor"]
        self.get_reference_logits_scale_at_step = self.factor_schedulers["reference_logits_scale"]
        self.get_alpha_reference_at_step = self.factor_schedulers["alpha_reference"]
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
        self._setup_grammar_processors()

        # Read illegal_vocab_penalty from reward object
        if hasattr(reward, "illegal_vocab_penalty"):
            self.illegal_vocab_penalty = float(reward.illegal_vocab_penalty)
        else:
            self.illegal_vocab_penalty = 0

        self.optimizer = optimizer
        self.scheduler = scheduler
        # Backward compatibility: prefer explicit compile_model, fall back to legacy compile flag.
        chosen_compile = compile_model if compile_model is not None else compile
        self.use_compile = bool(chosen_compile)

        self.train_sentence_length: list = []
        self.train_samples: list = []

        self.train_samples_ids: list = []
        self.val_samples_ids: list = []
        self.test_samples_ids: list = []

        self.val_samples_table: list = []
        self.val_log_rs: list = []
        self.val_log_pfss: list = []

        self.test_samples_table: list = []
        self.test_log_rs: list = []
        self.test_log_pfss: list = []

        self.train_samples_valid_flags: list = []
        self.val_samples_valid_flags: list = []
        self.test_samples_valid_flags: list = []
        self.train_len_counts: dict[int, int] = {}
        self.train_score_sum_by_len: dict[int, float] = {}
        self.train_score_count_by_len: dict[int, int] = {}
        self.train_len_counts_valid: dict[int, int] = {}
        self.train_score_sum_by_len_valid: dict[int, float] = {}
        self.train_score_count_by_len_valid: dict[int, int] = {}
        self.val_len_counts: dict[int, int] = {}
        self.val_score_sum_by_len: dict[int, float] = {}
        self.val_score_count_by_len: dict[int, int] = {}
        self.val_len_counts_valid: dict[int, int] = {}
        self.val_score_sum_by_len_valid: dict[int, float] = {}
        self.val_score_count_by_len_valid: dict[int, int] = {}
        self.test_len_counts: dict[int, int] = {}
        self.test_score_sum_by_len: dict[int, float] = {}
        self.test_score_count_by_len: dict[int, int] = {}
        self.test_len_counts_valid: dict[int, int] = {}
        self.test_score_sum_by_len_valid: dict[int, float] = {}
        self.test_score_count_by_len_valid: dict[int, int] = {}
        self.train_log_pterm_sum: dict[int, float] = {}
        self.train_log_pterm_count: dict[int, int] = {}
        self.val_log_pterm_sum: dict[int, float] = {}
        self.val_log_pterm_count: dict[int, int] = {}
        self.test_log_pterm_sum: dict[int, float] = {}
        self.test_log_pterm_count: dict[int, int] = {}
        self.val_batch_diversity_sum = 0.0
        self.val_batch_diversity_count = 0
        self.val_batch_fp_div_internal_sum = 0.0
        self.val_batch_fp_div_internal_count = 0
        self.val_batch_fp_div_topk_sum = 0.0
        self.val_batch_fp_div_topk_count = 0
        self.val_text_diversity_sum = 0.0
        self.val_text_diversity_count = 0
        self.val_text_diversity_sum = 0.0
        self.val_text_diversity_count = 0
        self.test_batch_diversity_sum = 0.0
        self.test_batch_diversity_count = 0
        self.test_batch_fp_div_internal_sum = 0.0
        self.test_batch_fp_div_internal_count = 0
        self.test_batch_fp_div_topk_sum = 0.0
        self.test_batch_fp_div_topk_count = 0
        self.test_text_diversity_sum = 0.0
        self.test_text_diversity_count = 0
        self.test_condvar_tokens: list[torch.Tensor] = []
        self.test_condvar_log_pf_steps: list[torch.Tensor] = []
        self.test_condvar_log_pterm: list[torch.Tensor] = []
        self.test_condvar_log_r: list[torch.Tensor] = []
        self.test_condvar_tau: list[int] = []

        div_metric = getattr(self.reward_config, "diversity_metric", None)
        div_model_name = getattr(
            self.reward_config,
            "diversity_model_name",
            "sentence-transformers/all-mpnet-base-v2",
        )
        self.sequence_diversity = (
            SequenceDiversity(div_metric, model_name=div_model_name) if div_metric else None
        )
        self.test_condvar_ref_log_pf_steps: list[torch.Tensor] = []
        self.test_condvar_ref_log_pterm: list[torch.Tensor] = []
        self.test_repeat_suffix: str = ""
        self.test_repeat_suffix: str = ""

        self.coverage_config = coverage_config or {}
        self._coverage_enabled = bool(self.coverage_config.get("enabled", False))
        self._coverage_reference_set: set[tuple[int, ...]] = set()
        self._coverage_seen_set: set[tuple[int, ...]] = set()
        self._coverage_total = 0
        self._coverage_steps_to_full: int | None = None
        self._coverage_total_samples = 0
        self._coverage_initialized = False

        self.skip_baseline_sampling = self.training_mixed_config.skip_baseline_sampling
        # If True, skip heavy epoch-end logging and only persist replay buffer.
        self.minimal_epoch_end_logging = getattr(
            self.training_mixed_config, "minimal_epoch_end_logging", True
        )

        self.ema_config = ema_config
        self._ema_enabled = bool(self._ema_cfg("enabled", False))
        self._ema_decay = float(self._ema_cfg("decay", 0.999))
        self._ema_start_epoch = int(self._ema_cfg("ema_start_epoch", 0))
        self._ema_reference_start_epoch = int(
            self._ema_cfg("reference_start_epoch", self._ema_start_epoch)
        )
        self._ema_reference_interval = int(self._ema_cfg("reference_update_interval", 1))
        self._ema_reference_delay_epochs = int(self._ema_cfg("reference_delay_epochs", 0))
        self._ema_log_param_delta = bool(self._ema_cfg("log_param_delta", False))
        self._ema_log_reference_delta = bool(self._ema_cfg("log_reference_delta", False))
        self._ema_state: dict[str, torch.Tensor] = {}
        self._ema_reference_update_count = 0
        self._ema_reference_prev_state: dict[str, torch.Tensor] = {}
        self._ema_state_history = (
            deque(maxlen=self._ema_reference_delay_epochs + 1)
            if self._ema_reference_delay_epochs > 0
            else None
        )

        # Initialize loss function from config
        self.loss_fn: GFNLoss = loss_fn

        # Optional: loss weight schedulers
        self.loss_weight_schedulers = dict(self.factor_schedulers)
        if hasattr(self.loss_fn, "set_weight_schedulers"):
            self.loss_fn.set_weight_schedulers(self.loss_weight_schedulers)

        if hasattr(self.loss_fn, "set_alpha_reference"):
            self.loss_fn.set_alpha_reference(self.get_alpha_reference_at_step(self.global_step))

        try:
            if self.use_compile:
                self.net = torch.compile(self.net, mode="max-autotune", fullgraph=False)
                self.net_frozen = torch.compile(
                    self.net_frozen, mode="max-autotune", fullgraph=False
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            log.warning("torch.compile failed, continuing without compilation: %s", exc)

        # phi cache
        self._pv_probe_cache = None
        self._pv_report_epoch = -1

    def forward(
        self,
        encoded_data,
        n_samples=None,
        pf_temperature=1.0,
        reward_temperature=1.0,
        scaling_factor=1,
        reference_logits_scale=0.5,
        action_seq=None,
        use_buffer_sample: bool = False,
        buffer_sample: torch.Tensor | None = None,
        buffer_mixture_ratio: float = 0.5,
    ):
        """Forward pass for generation and reward computation.

        Args:
            encoded_data: Dictionary containing encoded prompt and optional molecule.
            n_samples: Number of samples to generate (default: from config).
            pf_temperature: Temperature for forward policy sampling (default: 1.0).
            reward_temperature: Temperature for reward computation (default: 1.0).
            scaling_factor: Reward scaling factor (default: 50).
            action_seq: Optional pre-computed action sequence.
            use_buffer_sample: Whether to use buffer samples (default: False).
            buffer_sample: Optional buffer samples tensor.
            buffer_mixture_ratio: Ratio of samples to replace with buffer (default: 0.5).

        Returns:
            Dictionary with generation results (state, log_pf, log_pterm, log_r, etc.).
        """
        encoded_prompt = encoded_data["encoded_prompt"]
        encoded_prompt = encoded_prompt.squeeze(0) if encoded_prompt.ndim != 1 else encoded_prompt
        n_samples = n_samples or self.training_mixed_config.n_samples
        encoded_prompt = encoded_prompt.expand(n_samples, -1)
        encoded_data["encoded_prompt"] = encoded_prompt
        prompt_len = encoded_prompt.shape[1]

        generation_config = {
            "model": self.net,
            "encoded_data": encoded_data,
            "grammar_processor": self.pre_grammar_processor,
            "reward_fn": partial(
                self.reward.score,
                prompt_length=encoded_prompt.shape[1],
                model=self.net_frozen,
                tokenizer=self.tokenizer,
            ),
            "termination_token_id": self.end_of_sentence_token_id,
            "min_len": self.constraint_config.min_sentence_len,
            "max_len": self.constraint_config.max_sentence_len,
            "temperature": pf_temperature,
            "reward_temperature": reward_temperature,
            "scaling_factor": scaling_factor,
            "reference_logits_scale": reference_logits_scale,
            "skip_rewards": False,
            "action_seq": action_seq,
            "vocab_nice_mask": self.legal_tokens_mask,
            "vocab_invalid_mask": self.illegal_tokens_mask,
            "illegal_vocab_penalty": self.illegal_vocab_penalty,
            "use_buffer_sample": use_buffer_sample,
            "buffer_sample": buffer_sample,
            "buffer_mixture_ratio": buffer_mixture_ratio,
            "disable_grammar": getattr(self.constraint_config, "disable_grammar", False),
            "grammar_disagree_penalty": getattr(
                self.training_mixed_config, "grammar_disagree_penalty", -80
            ),
        }

        result = generate_and_return_termination_logprob(**generation_config)

        return result

    # ------------------------------------------------------------------ #
    # Training / validation loops
    # ------------------------------------------------------------------ #

    def training_step(self, item, batch_idx) -> torch.Tensor:
        encoded_prompt = item["encoded_prompt"]
        prompt_len = encoded_prompt.shape[-1]
        buffer_sample = item["buffer_encoded_sample"]
        use_dataset_buffer = False

        # Set prefix value memory if available
        pv_mem = getattr(self.reward, "pv_mem", None)
        if pv_mem is not None:
            if hasattr(self.reward, "set_step"):
                self.reward.set_step(self.global_step)
            self.reward.pv_mem.set_step(self.global_step)
            nums_pv_keys = len(self.reward.pv_mem.stats.keys())
            self._log_metrics(
                {"train/nums_pv_keys": nums_pv_keys},
                sync_dist=True,
                on_step=True,
            )

        # Buffer 1: Try to use replay buffer
        if random.random() <= self.get_replay_buffer_at_step(self.global_step):
            replay_buffer_result = self.generate_from_replay_buffer(item, encoded_prompt)
            if replay_buffer_result is not None:
                use_replay_buffer = True
            else:
                use_replay_buffer = False
        else:
            use_replay_buffer = False

        if use_replay_buffer:
            # Using replay buffer, no need for dataset buffer
            _, result_dict = replay_buffer_result
            pf_temp = 1.0
        else:
            # Not using replay buffer, decide whether to use dataset buffer
            pf_temp = self._sample_pf_temperature()

            # Buffer 2: Decide whether to use dataset buffer (dynamic probability schedule)
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
        agree_list = result_dict["agree_list"]

        # Extra phi information
        phi_state = result_dict.get("phi_state", None)
        phi_tok = result_dict.get("phi_tok", None)
        pv = result_dict.get("pv", None)
        phi_weight = result_dict.get("phi_weight", 0)

        self._log_metrics(
            {
                "train/phi_weight": phi_weight if phi_weight is not None else 0,
            },
            sync_dist=True,
            on_step=True,
        )

        if phi_state is None:
            phi_state = torch.zeros_like(log_pterm)
        if phi_tok is None:
            phi_tok = torch.zeros_like(log_pf)
        if pv is None:
            pv = torch.zeros_like(log_pf)

        if self._pv_probe_cache is None:
            if (not use_replay_buffer) and (not use_dataset_buffer):
                eos = self.end_of_sentence_token_id
                B = generated_text.shape[0]
                B_probe = min(B, 128)

                T_tok = log_pf_ref.shape[1]

                probe_tokens = (
                    generated_text[:B_probe, prompt_len : prompt_len + T_tok].detach().cpu()
                )
                probe_active_before = compute_active_before(probe_tokens, eos=eos).detach().cpu()
                probe_ref_log_pf = log_pf_ref[:B_probe, :T_tok].detach().cpu()

                self._pv_probe_cache = {
                    "tokens": probe_tokens,
                    "active_before": probe_active_before,
                    "ref_log_pf": probe_ref_log_pf,
                }

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

        # automatically log all classes inside factor_schedulers
        # we need read from the class name and log the value, group all those values together
        scheduled_values = {}
        for key, value in self.factor_schedulers.items():
            scheduled_values[key] = value(self.global_step)

        # Log each scheduled value separately (Lightning doesn't support logging dicts directly)
        self._log_metrics(
            {f"scheduled/{key}": val for key, val in scheduled_values.items()},
            sync_dist=True,
            on_step=True,
        )

        # CFG agreement metrics
        if not use_replay_buffer or not use_dataset_buffer:
            self._log_agreement_metrics(agree_list)

        if hasattr(self.loss_fn, "set_global_step"):
            self.loss_fn.set_global_step(self.global_step)

        loss_output = self.loss_fn(
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

        # Handle dict or scalar output for backward compatibility
        if isinstance(loss_output, dict):
            loss = loss_output["loss"]
            # Log all component losses automatically
            component_losses = {f"train/{k}": v for k, v in loss_output.items() if k != "loss"}
            if component_losses:
                self._log_metrics(component_losses, on_step=True, sync_dist=True, prog_bar=True)
        else:
            loss = loss_output

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
        self._update_coverage_from_tokens(tokens)
        self._accumulate_log_pterm_by_length(
            tokens,
            log_pterm,
            self.train_log_pterm_sum,
            self.train_log_pterm_count,
        )

        validator_dict = result_dict.get("validator_dict")
        valid_flags = self._get_valid_flags(validator_dict, generated_text, prompt_len)
        if self.reward.sentence_validator is None:
            validator_metric_dict = {}
        else:
            # compatibility for SMILES and expr24 validators
            try:
                validator_metric_dict = self.reward.sentence_validator.accuracy(
                    tokens,
                    self.tokenizer,
                    item.get("scaffold", None),
                    return_hist=True,
                )
            except TypeError:
                try:
                    validator_metric_dict = self.reward.sentence_validator.accuracy(
                        tokens,
                        self.tokenizer,
                        item.get("scaffold", None),
                    )
                except TypeError:
                    validator_metric_dict = self.reward.sentence_validator.accuracy(
                        tokens,
                        self.tokenizer,
                    )
        self._log_validator_core_metrics(
            "train",
            validator_metric_dict,
            sync_dist=True,
            on_step=True,
        )
        if validator_metric_dict:
            len_hist = validator_metric_dict.pop("len_tok_hist", None)
            score_hist = validator_metric_dict.pop("score_hist", None)
            scores = validator_dict.get("global_score") if validator_dict is not None else None
            if isinstance(len_hist, list):
                scores_src = score_hist if isinstance(score_hist, list) else scores
                self._accumulate_length_stats(
                    len_hist,
                    scores_src,
                    self.train_len_counts,
                    self.train_score_sum_by_len,
                    self.train_score_count_by_len,
                )
                if valid_flags is not None:
                    self._accumulate_length_stats(
                        len_hist,
                        scores_src,
                        self.train_len_counts_valid,
                        self.train_score_sum_by_len_valid,
                        self.train_score_count_by_len_valid,
                        valid_flags=valid_flags,
                    )
            else:
                lengths = self._lengths_from_tokens(tokens)
                self._accumulate_length_stats(
                    lengths,
                    scores,
                    self.train_len_counts,
                    self.train_score_sum_by_len,
                    self.train_score_count_by_len,
                )
                if valid_flags is not None:
                    self._accumulate_length_stats(
                        lengths,
                        scores,
                        self.train_len_counts_valid,
                        self.train_score_sum_by_len_valid,
                        self.train_score_count_by_len_valid,
                        valid_flags=valid_flags,
                    )
        else:
            scores = validator_dict.get("global_score") if validator_dict is not None else None
            lengths = self._lengths_from_tokens(tokens)
            self._accumulate_length_stats(
                lengths,
                scores,
                self.train_len_counts,
                self.train_score_sum_by_len,
                self.train_score_count_by_len,
            )
            if valid_flags is not None:
                self._accumulate_length_stats(
                    lengths,
                    scores,
                    self.train_len_counts_valid,
                    self.train_score_sum_by_len_valid,
                    self.train_score_count_by_len_valid,
                    valid_flags=valid_flags,
                )
        self.train_sentence_length.append(sentence_len.detach().cpu())

        log_ps = last_log_r * self.reward.temperature
        log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature

        if batch_idx % 5 == 0:
            decoded = self._decode_generated_tokens(generated_text[0, prompt_len:])
            if valid_flags is not None and len(valid_flags) > 0:
                acc_val = float(valid_flags[0])
            else:
                acc_val = 0.0

            self.train_samples_ids.extend(generated_text[:, prompt_len:].detach().cpu().tolist())
            self.train_samples_valid_flags.extend(valid_flags)

            self.train_samples.append(
                {
                    "decoded": decoded,
                    "pf_temp": float(pf_temp),
                    "reward_temp": float(self.reward.temperature),
                    "logP": float(log_ps[0].item()),
                    "logR": float(last_log_r[0].item()),
                    "log_pf_list": [float(x) for x in log_pf[0].detach().cpu().tolist()],
                    "log_pf_ref_list": [float(x) for x in log_pf_ref[0].detach().cpu().tolist()],
                    "log_r_list": [float(x) for x in log_r[0].detach().cpu().tolist()],
                    "log_r_unpenalized_list": [
                        float(x) for x in log_r_unpenalized[0].detach().cpu().tolist()
                    ],
                    "log_pterm_list": [float(x) for x in log_pterm[0].detach().cpu().tolist()],
                    "log_pterm_ref_list": [
                        float(x) for x in log_pterm_ref[0].detach().cpu().tolist()
                    ],
                    "phi_state_list": [
                        round(float(x), 3) for x in phi_state[0].detach().cpu().tolist()
                    ],
                    "phi_tok_list": [
                        round(float(x), 3) for x in phi_tok[0].detach().cpu().tolist()
                    ],
                    "pv_list": [round(float(x), 3) for x in pv[0].detach().cpu().tolist()],
                    "valid": float(acc_val),
                    "buffer": bool(use_replay_buffer or use_dataset_buffer),
                    "source": "replay"
                    if use_replay_buffer
                    else ("dataset" if use_dataset_buffer else "online"),
                }
            )

            replay_buffer_stats = self.reward_buffer.stat()
            self._log_metrics(
                {
                    f"train/replay_buffer_{key}": float(value)
                    for key, value in replay_buffer_stats.items()
                },
                on_step=True,
                sync_dist=True,
            )

        self._log_metrics(
            {
                "train/reward_var": log_r.var(dim=0).mean(),
                "train/loss": (loss, {"prog_bar": True}),
            },
            on_step=True,
            sync_dist=True,
        )

        prefix_value_diag = result_dict.get("prefix_value_diag", None)
        if prefix_value_diag is not None:
            metrics = {}
            for key, value in prefix_value_diag.items():
                if isinstance(value, torch.Tensor):
                    if value.ndim == 0:
                        metrics[f"prefix_value_diag/{key}"] = value
                    elif value.ndim == 1:
                        for idx, v in enumerate(value):
                            metrics[f"prefix_value_diag/{key}/t{idx}"] = v
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    for idx, v in enumerate(value):
                        metrics[f"prefix_value_diag/{key}/t{idx}"] = v
                else:
                    metrics[f"prefix_value_diag/{key}"] = value
            if metrics:
                self._log_metrics(metrics, on_step=True, sync_dist=True)

        self._log_metrics(
            {
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

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        encoded_prompt = batch["encoded_prompt"]
        prompt_len = encoded_prompt.shape[-1]
        pf_temp_eval = getattr(self.training_mixed_config, "pf_temp_eval", None)
        pf_temp_eval = (
            pf_temp_eval
            if pf_temp_eval is not None
            else getattr(self.training_mixed_config, "pf_temp_high", 1.0)
        )
        result_dict = self.forward(
            batch, reward_temperature=1.0, pf_temperature=float(pf_temp_eval)
        )
        generated_text = result_dict["state"]
        log_pf = result_dict["log_pf"]
        log_pterm = result_dict["log_pterm"]
        log_r = result_dict["log_r"]
        log_pf_ref = result_dict["log_pf_ref"]
        log_pterm_ref = result_dict["log_pterm_ref"]
        validator_dict = result_dict.get("validator_dict")

        log_r_unpenalized = result_dict["log_r_unpenalized"]

        if hasattr(self.loss_fn, "set_global_step"):
            self.loss_fn.set_global_step(self.global_step)

        loss_output = self.loss_fn(
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

        # Handle dict or scalar output for backward compatibility
        if isinstance(loss_output, dict):
            loss = loss_output["loss"]
            # Log all component losses automatically
            component_losses = {f"val/{k}": v for k, v in loss_output.items() if k != "loss"}
            if component_losses:
                self._log_metrics(component_losses, on_step=False, sync_dist=True)
        else:
            loss = loss_output

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

        # Mean generated length (token ids) for validation logging.
        # This is computed directly from token ids and is robust when EOS is absent.
        eos_id = int(self.end_of_sentence_token_id)
        if tokens is not None and tokens.numel() > 0:
            eos_mask = tokens.eq(eos_id)
            has_eos = eos_mask.any(dim=1)
            first_eos = eos_mask.float().argmax(dim=1)
            full_len = int(tokens.shape[1])
            lengths_tok = torch.where(
                has_eos, first_eos, first_eos.new_full(first_eos.shape, full_len)
            ).to(dtype=torch.float32)
            mean_len_tok_ids = lengths_tok.mean()
        else:
            mean_len_tok_ids = torch.tensor(0.0, device=generated_text.device)
        self._update_coverage_from_tokens(tokens)
        self._accumulate_log_pterm_by_length(
            tokens,
            log_pterm,
            self.val_log_pterm_sum,
            self.val_log_pterm_count,
        )
        if log_pfs is not None:
            self.val_log_rs.append(last_log_r.detach().cpu())
            self.val_log_pfss.append(log_pfs.detach().cpu())

        validator_dict = result_dict.get("validator_dict")
        valid_flags = self._get_valid_flags(validator_dict, generated_text, prompt_len)

        batch_sequences = self._strip_eos_from_batch(generated_text[:, prompt_len:])
        self.val_samples_ids.extend(batch_sequences)
        self.val_samples_valid_flags.extend(valid_flags)
        batch_diversity = self._calculate_diversity_ragged(batch_sequences)
        self.val_batch_diversity_sum += float(batch_diversity)
        self.val_batch_diversity_count += 1
        if self.sequence_diversity is not None:
            decoded_batch = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
            decoded_batch = [
                text.replace(self.tokenizer.eos_token, "").strip() for text in decoded_batch
            ]
            if len(decoded_batch) > 1:
                text_div = self.sequence_diversity(decoded_batch)
                if text_div is not None:
                    self.val_text_diversity_sum += float(text_div)
                    self.val_text_diversity_count += 1
                    self._log_metrics(
                        {"val/diversity_text": text_div},
                        on_step=False,
                        sync_dist=True,
                    )

        if self.reward.sentence_validator is None:
            validator_metric_dict = {}
        else:
            try:
                validator_metric_dict = self.reward.sentence_validator.accuracy(
                    tokens,
                    self.tokenizer,
                    batch.get("scaffold", None),
                    return_hist=True,
                )
            except TypeError:
                try:
                    validator_metric_dict = self.reward.sentence_validator.accuracy(
                        tokens,
                        self.tokenizer,
                        batch.get("scaffold", None),
                    )
                except TypeError:
                    validator_metric_dict = self.reward.sentence_validator.accuracy(
                        tokens,
                        self.tokenizer,
                    )
        fp_div_internal = None
        fp_div_topk = None
        if validator_metric_dict:
            fp_div_internal = validator_metric_dict.get("fp_div_internal_valid")
            fp_div_topk = validator_metric_dict.get("fp_div_topk_valid")
        if validator_metric_dict:
            validator_metric_dict.pop("fp_div_internal_valid", None)
            validator_metric_dict.pop("fp_div_topk_valid", None)
        if fp_div_internal is not None:
            self.val_batch_fp_div_internal_sum += float(fp_div_internal)
            self.val_batch_fp_div_internal_count += 1
        if fp_div_topk is not None:
            self.val_batch_fp_div_topk_sum += float(fp_div_topk)
            self.val_batch_fp_div_topk_count += 1
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
        if validator_metric_dict:
            len_hist = validator_metric_dict.pop("len_tok_hist", None)
            score_hist = validator_metric_dict.pop("score_hist", None)
            scores = validator_dict.get("global_score") if validator_dict is not None else None
            if isinstance(len_hist, list):
                scores_src = score_hist if isinstance(score_hist, list) else scores
                self._accumulate_length_stats(
                    len_hist,
                    scores_src,
                    self.val_len_counts,
                    self.val_score_sum_by_len,
                    self.val_score_count_by_len,
                )
                if valid_flags is not None:
                    self._accumulate_length_stats(
                        len_hist,
                        scores_src,
                        self.val_len_counts_valid,
                        self.val_score_sum_by_len_valid,
                        self.val_score_count_by_len_valid,
                        valid_flags=valid_flags,
                    )
            else:
                lengths = self._lengths_from_tokens(tokens)
                self._accumulate_length_stats(
                    lengths,
                    scores,
                    self.val_len_counts,
                    self.val_score_sum_by_len,
                    self.val_score_count_by_len,
                )
                if valid_flags is not None:
                    self._accumulate_length_stats(
                        lengths,
                        scores,
                        self.val_len_counts_valid,
                        self.val_score_sum_by_len_valid,
                        self.val_score_count_by_len_valid,
                        valid_flags=valid_flags,
                    )
        else:
            scores = validator_dict.get("global_score") if validator_dict is not None else None
            lengths = self._lengths_from_tokens(tokens)
            self._accumulate_length_stats(
                lengths,
                scores,
                self.val_len_counts,
                self.val_score_sum_by_len,
                self.val_score_count_by_len,
            )
            if valid_flags is not None:
                self._accumulate_length_stats(
                    lengths,
                    scores,
                    self.val_len_counts_valid,
                    self.val_score_sum_by_len_valid,
                    self.val_score_count_by_len_valid,
                    valid_flags=valid_flags,
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
                # Two length views for monitoring.
                # - sentence_len: length from get_termination_vals (EOS position)
                # - len_tok_ids: length computed directly from token ids
                "val/sentence_len": sentence_len.float().mean(),
                "val/len_tok_ids": mean_len_tok_ids,
            },
            sync_dist=True,
        )

        generated_sequences = [
            self._decode_generated_tokens(text[prompt_len:], skip_special_tokens=False)
            for text in generated_text
        ]
        if result_dict.get("full_tokens", None) is not None:
            generated_sequences = result_dict["full_tokens"]

        if valid_flags is None:
            valid_flags = [None] * len(generated_sequences)

        for idx, sequence in enumerate(generated_sequences):
            raw_ids = generated_text[idx, prompt_len:].detach().cpu().tolist()
            raw_ids = self._strip_special_token_ids(raw_ids)
            self.val_samples_table.append(
                {
                    "Sampled sentence": sequence,
                    "token_ids": raw_ids,
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

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        encoded_prompt = batch["encoded_prompt"]
        prompt_len = encoded_prompt.shape[-1]
        pf_temp_eval = getattr(self.training_mixed_config, "pf_temp_eval", None)
        pf_temp_eval = (
            pf_temp_eval
            if pf_temp_eval is not None
            else getattr(self.training_mixed_config, "pf_temp_high", 1.0)
        )
        result_dict = self.forward(
            batch, reward_temperature=1.0, pf_temperature=float(pf_temp_eval)
        )
        generated_text = result_dict["state"]
        log_pf = result_dict["log_pf"]
        log_pterm = result_dict["log_pterm"]
        log_r = result_dict["log_r"]
        log_pf_ref = result_dict["log_pf_ref"]
        log_pterm_ref = result_dict["log_pterm_ref"]
        validator_dict = result_dict.get("validator_dict")

        log_r_unpenalized = result_dict["log_r_unpenalized"]

        batch_sequences = self._strip_eos_from_batch(generated_text[:, prompt_len:])
        self.test_samples_ids.extend(batch_sequences)
        batch_diversity = self._calculate_diversity_ragged(batch_sequences)
        self.test_batch_diversity_sum += float(batch_diversity)
        self.test_batch_diversity_count += 1

        if hasattr(self.loss_fn, "set_global_step"):
            self.loss_fn.set_global_step(self.global_step)

        loss_output = self.loss_fn(
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

        if isinstance(loss_output, dict):
            loss = loss_output["loss"]
            component_losses = {f"test/{k}": v for k, v in loss_output.items() if k != "loss"}
            if component_losses:
                self._log_metrics(component_losses, on_step=False, sync_dist=True)
        else:
            loss = loss_output

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
        if self.sequence_diversity is not None:
            decoded_batch = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
            decoded_batch = [
                text.replace(self.tokenizer.eos_token, "").strip() for text in decoded_batch
            ]
            if len(decoded_batch) > 1:
                text_div = self.sequence_diversity(decoded_batch)
                if text_div is not None:
                    self.test_text_diversity_sum += float(text_div)
                    self.test_text_diversity_count += 1
                    self._log_metrics(
                        {"test/diversity_text": text_div},
                        on_step=False,
                        sync_dist=True,
                    )
        tau_tensor = self._compute_tau_from_tokens(tokens)
        self._accumulate_log_pterm_by_length(
            tokens,
            log_pterm,
            self.test_log_pterm_sum,
            self.test_log_pterm_count,
        )
        if log_pfs is not None:
            self.test_log_rs.append(last_log_r.detach().cpu())
            self.test_log_pfss.append(log_pfs.detach().cpu())
        log_pf_steps = log_pf[:, :-1] if log_pf is not None else None
        if (
            log_pf_steps is not None
            and log_pterm is not None
            and log_r is not None
            and tau_tensor is not None
        ):
            self._cache_test_conditional_variance_inputs(
                tokens=tokens,
                log_pf_steps=log_pf_steps,
                log_pterm=log_pterm,
                log_r=log_r,
                tau=tau_tensor,
                ref_log_pf_steps=log_pf_ref[:, :-1] if log_pf_ref is not None else None,
                ref_log_pterm=log_pterm_ref if log_pterm_ref is not None else None,
            )

        validator_dict = result_dict.get("validator_dict")
        valid_flags = self._get_valid_flags(validator_dict, generated_text, prompt_len)
        self.test_samples_valid_flags.extend(valid_flags)
        if self.reward.sentence_validator is None:
            validator_metric_dict = {}
        else:
            try:
                validator_metric_dict = self.reward.sentence_validator.accuracy(
                    tokens,
                    self.tokenizer,
                    batch.get("scaffold", None),
                    return_hist=True,
                )
            except TypeError:
                try:
                    validator_metric_dict = self.reward.sentence_validator.accuracy(
                        tokens,
                        self.tokenizer,
                        batch.get("scaffold", None),
                    )
                except TypeError:
                    validator_metric_dict = self.reward.sentence_validator.accuracy(
                        tokens,
                        self.tokenizer,
                    )
        fp_div_internal = None
        fp_div_topk = None
        if validator_metric_dict:
            fp_div_internal = validator_metric_dict.get("fp_div_internal_valid")
            fp_div_topk = validator_metric_dict.get("fp_div_topk_valid")
        if validator_metric_dict:
            validator_metric_dict.pop("fp_div_internal_valid", None)
            validator_metric_dict.pop("fp_div_topk_valid", None)
        if fp_div_internal is not None:
            self.test_batch_fp_div_internal_sum += float(fp_div_internal)
            self.test_batch_fp_div_internal_count += 1
        if fp_div_topk is not None:
            self.test_batch_fp_div_topk_sum += float(fp_div_topk)
            self.test_batch_fp_div_topk_count += 1

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
        if validator_metric_dict:
            len_hist = validator_metric_dict.pop("len_tok_hist", None)
            score_hist = validator_metric_dict.pop("score_hist", None)

            scores = validator_dict.get("global_score") if validator_dict is not None else None
            if isinstance(len_hist, list):
                scores_src = score_hist if isinstance(score_hist, list) else scores
                self._accumulate_length_stats(
                    len_hist,
                    scores_src,
                    self.test_len_counts,
                    self.test_score_sum_by_len,
                    self.test_score_count_by_len,
                )
                if valid_flags is not None:
                    self._accumulate_length_stats(
                        len_hist,
                        scores_src,
                        self.test_len_counts_valid,
                        self.test_score_sum_by_len_valid,
                        self.test_score_count_by_len_valid,
                        valid_flags=valid_flags,
                    )
            else:
                lengths = self._lengths_from_tokens(tokens)
                self._accumulate_length_stats(
                    lengths,
                    scores,
                    self.test_len_counts,
                    self.test_score_sum_by_len,
                    self.test_score_count_by_len,
                )
                if valid_flags is not None:
                    self._accumulate_length_stats(
                        lengths,
                        scores,
                        self.test_len_counts_valid,
                        self.test_score_sum_by_len_valid,
                        self.test_score_count_by_len_valid,
                        valid_flags=valid_flags,
                    )
        else:
            scores = validator_dict.get("global_score") if validator_dict is not None else None
            lengths = self._lengths_from_tokens(tokens)
            self._accumulate_length_stats(
                lengths,
                scores,
                self.test_len_counts,
                self.test_score_sum_by_len,
                self.test_score_count_by_len,
            )
            if valid_flags is not None:
                self._accumulate_length_stats(
                    lengths,
                    scores,
                    self.test_len_counts_valid,
                    self.test_score_sum_by_len_valid,
                    self.test_score_count_by_len_valid,
                    valid_flags=valid_flags,
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

        generated_sequences = [
            self._decode_generated_tokens(text[prompt_len:], skip_special_tokens=False)
            for text in generated_text
        ]
        if result_dict.get("full_tokens", None) is not None:
            generated_sequences = result_dict["full_tokens"]

        if valid_flags is None:
            valid_flags = [None] * len(generated_sequences)

        for idx, sequence in enumerate(generated_sequences):
            raw_ids = generated_text[idx, prompt_len:].detach().cpu().tolist()
            raw_ids = self._strip_special_token_ids(raw_ids)
            self.test_samples_table.append(
                {
                    "Sampled sentence": sequence,
                    "token_ids": raw_ids,
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

    def on_train_batch_start(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
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

    def on_train_epoch_start(self):
        self._log_metrics(
            {"scheduled/R_temperature": self.get_reward_temp_at_step(self.global_step)},
            sync_dist=True,
        )
        self._log_metrics(
            {
                "scheduled/lr": (
                    cast(Any, self.lr_schedulers()).get_lr()[0] if self.lr_schedulers() else 0.0
                )
            },
            sync_dist=True,
        )
        self.train_len_counts.clear()
        self.train_score_sum_by_len.clear()
        self.train_score_count_by_len.clear()
        self.train_len_counts_valid.clear()
        self.train_score_sum_by_len_valid.clear()
        self.train_score_count_by_len_valid.clear()
        self.train_log_pterm_sum.clear()
        self.train_log_pterm_count.clear()
        self.train_samples_valid_flags.clear()

    def on_train_epoch_end(self):
        self._maybe_update_ema()

        if hasattr(self, "global_rank") and self.global_rank != 0:
            return

        if self.minimal_epoch_end_logging:
            self.reward_buffer.save_csv(
                os.path.join(
                    self.trainer.default_root_dir,
                    "replay_buffer",
                    f"replay_{self.trainer.current_epoch}.csv",
                ),
                self.tokenizer,
            )
            self.train_samples.clear()
            self.train_samples_ids.clear()
            self.train_sentence_length.clear()
            return

        # log training samples
        if self.train_samples:
            df = pd.DataFrame(self.train_samples)
            df = df.sort_values(by="buffer", ascending=False).reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=["decoded", "pf_temp", "logP", "valid", "buffer", "source"])

        _root_dir = os.path.join(self.trainer.default_root_dir, "train_samples")
        os.makedirs(_root_dir, exist_ok=True)
        df.to_csv(
            os.path.join(_root_dir, f"samples_train_probes_{self.trainer.current_epoch}.csv"),
            index=False,
        )

        # log replay buffer
        self.reward_buffer.save_csv(
            os.path.join(
                self.trainer.default_root_dir,
                "replay_buffer",
                f"replay_{self.trainer.current_epoch}.csv",
            ),
            self.tokenizer,
        )

        # log prefix collapse metrics
        eos = self.end_of_sentence_token_id
        seqs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in self.train_samples_ids],
            batch_first=True,
            padding_value=eos,
        )
        valid_flags = torch.tensor(self.train_samples_valid_flags)
        active_before = compute_active_before(seqs, eos=eos)
        non_eos = seqs != eos
        mask_noeos = active_before & non_eos
        seqs_list = seqs.detach().cpu().tolist()
        mask_list = mask_noeos.detach().cpu().tolist()
        invalid_flags = (~valid_flags).tolist()

        pos = prefix_collapse_by_position(
            seqs_list, mask_list, collapse_thr=0.95, invalid=invalid_flags
        )
        kmet = prefix_collapse_by_k(
            seqs_list,
            mask_list,
            k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            invalid=invalid_flags,
        )

        log = {
            "prefix_pos/top1_auc_train": float(pos.top1_auc),
            "prefix_pos/top1_auc_correct_train": float(pos.top1_auc_correct),
        }

        self._log_metrics(log, on_epoch=True, sync_dist=True)
        self._log_wandb_prefix_tables("prefix_pos_train", pos, kmet)

        # log training sentence length dist.
        plt.hist(self.train_sentence_length, bins=50)
        _root_dir = os.path.join(self.trainer.default_root_dir, "train_sentence_length")
        os.makedirs(_root_dir, exist_ok=True)
        plt.savefig(
            os.path.join(_root_dir, f"train_sentence_length_{self.trainer.current_epoch}.png")
        )
        plt.close()

        # if self.logger is not None:
        #     self.logger.log_table("train/samples_latest", dataframe=df)

        # log prefix collapse metrics
        if (self._pv_probe_cache is None) or (getattr(self.reward, "pv_mem", None) is None):
            return

        pv_mem = self.reward.pv_mem
        if (self.trainer.current_epoch + 1) % 10 == 0:
            root_dir = os.path.join(self.trainer.default_root_dir, "phi_report")
            os.makedirs(root_dir, exist_ok=True)
            csv_kgram_path = os.path.join(root_dir, f"kgram_{self.trainer.current_epoch}.csv")
            csv_prefix_path = os.path.join(root_dir, f"prefix_{self.trainer.current_epoch}.csv")

            rep = pv_mem.report(
                max_keys_sample=20000,
                csv_kgram_path=csv_kgram_path,
                csv_prefix_path=csv_prefix_path,
                probe_tokens=self._pv_probe_cache["tokens"],
                probe_active_before=self._pv_probe_cache["active_before"],
                probe_ref_log_pf=self._pv_probe_cache["ref_log_pf"],
                phi_eta=getattr(self.reward, "phi_eta", 1.0),
                phi_clamp=getattr(self.reward, "phi_clamp", 2.0),
                tau_conf=getattr(pv_mem, "tau_conf", 20.0),
                short_split=getattr(self.reward, "pv_split", 2),
            )
        else:
            rep = pv_mem.report(
                max_keys_sample=20000,
                csv_kgram_path=None,
                csv_prefix_path=None,
                probe_tokens=self._pv_probe_cache["tokens"],
                probe_active_before=self._pv_probe_cache["active_before"],
                probe_ref_log_pf=self._pv_probe_cache["ref_log_pf"],
                phi_eta=getattr(self.reward, "phi_eta", 1.0),
                phi_clamp=getattr(self.reward, "phi_clamp", 2.0),
                tau_conf=getattr(pv_mem, "tau_conf", 20.0),
                short_split=getattr(self.reward, "pv_split", 2),
            )

        self._log_metrics(
            {f"pv_report/{k}": v for k, v in rep.items()},
            on_epoch=True,
            sync_dist=True,
        )

        self.train_samples.clear()
        self.train_samples_ids.clear()
        self.train_sentence_length.clear()

    def on_validation_epoch_start(self):
        """Prepare validation probes and reset cached samples."""
        val_dataset = self.trainer.datamodule.val_dataloader().dataset
        self.val_probes = torch.utils.data.Subset(
            val_dataset, random.sample(range(len(val_dataset)), 10)
        )

        self.val_samples_ids.clear()
        self.val_samples_table.clear()
        self.val_log_rs.clear()
        self.val_log_pfss.clear()
        self.val_samples_valid_flags.clear()
        self.val_len_counts.clear()
        self.val_score_sum_by_len.clear()
        self.val_score_count_by_len.clear()
        self.val_len_counts_valid.clear()
        self.val_score_sum_by_len_valid.clear()
        self.val_score_count_by_len_valid.clear()
        self.val_log_pterm_sum.clear()
        self.val_log_pterm_count.clear()
        self.val_batch_diversity_sum = 0.0
        self.val_batch_diversity_count = 0
        self.val_batch_fp_div_internal_sum = 0.0
        self.val_batch_fp_div_internal_count = 0
        self.val_batch_fp_div_topk_sum = 0.0
        self.val_batch_fp_div_topk_count = 0

    def on_validation_epoch_end(self):
        diversity = self._calculate_diversity_ragged(self.val_samples_ids)
        self._log_metrics({"val/diversity": diversity}, sync_dist=True, on_epoch=True)
        valid_val_sequences = self._filter_valid_sequences(
            self.val_samples_ids, self.val_samples_valid_flags
        )
        diversity_val_valid = self._calculate_diversity_ragged(valid_val_sequences)
        self._log_metrics(
            {"val/diversity_valid": diversity_val_valid}, sync_dist=True, on_epoch=True
        )
        if self.val_batch_diversity_count > 0:
            self._log_metrics(
                {
                    "val/diversity_batch_mean": self.val_batch_diversity_sum
                    / float(self.val_batch_diversity_count)
                },
                sync_dist=True,
                on_epoch=True,
            )
        if self.sequence_diversity is not None and self.val_text_diversity_count > 0:
            self._log_metrics(
                {
                    "val/diversity_text_batch_mean": self.val_text_diversity_sum
                    / float(self.val_text_diversity_count)
                },
                sync_dist=True,
                on_epoch=True,
            )
        if self.val_batch_fp_div_internal_count > 0:
            mean_internal = self.val_batch_fp_div_internal_sum / float(
                self.val_batch_fp_div_internal_count
            )
            self._log_metrics(
                {
                    "val/fp_div_internal_valid_batch_mean": mean_internal,
                    "val/validator/fp_div_internal_valid_batch_mean": mean_internal,
                },
                sync_dist=True,
                on_epoch=True,
            )
        if self.val_batch_fp_div_topk_count > 0:
            mean_topk = self.val_batch_fp_div_topk_sum / float(self.val_batch_fp_div_topk_count)
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
        if self.sequence_diversity is not None:
            decoded_all = [
                self.tokenizer.decode(seq, skip_special_tokens=True)
                for seq in self.val_samples_ids
            ]
            if len(decoded_all) > 1:
                text_div_epoch = self.sequence_diversity(decoded_all)
                if text_div_epoch is not None:
                    self._log_metrics(
                        {"val/diversity_text_epoch": text_div_epoch},
                        sync_dist=True,
                        on_epoch=True,
                    )
        fp_div = self._compute_global_fp_diversity(self.val_samples_ids)
        if fp_div:
            self._log_metrics(
                {
                    "val/fp_div_internal_valid": fp_div.get("fp_div_internal_valid", 0.0),
                    "val/fp_div_topk_valid": fp_div.get("fp_div_topk_valid", 0.0),
                    "val/validator/fp_div_internal_valid": fp_div.get(
                        "fp_div_internal_valid", 0.0
                    ),
                    "val/validator/fp_div_topk_valid": fp_div.get("fp_div_topk_valid", 0.0),
                },
                sync_dist=True,
                on_epoch=True,
            )

        # log prefix collapse metrics
        eos = self.end_of_sentence_token_id
        valid_flags = torch.tensor(self.val_samples_valid_flags)
        seqs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in self.val_samples_ids],
            batch_first=True,
            padding_value=eos,
        )
        active_before = compute_active_before(seqs, eos=eos)
        non_eos = seqs != eos
        mask_noeos = active_before & non_eos
        seqs_list = seqs.detach().cpu().tolist()
        mask_list = mask_noeos.detach().cpu().tolist()
        invalid_flags = (~valid_flags).tolist()
        pos = prefix_collapse_by_position(
            seqs_list, mask_list, collapse_thr=0.95, invalid=invalid_flags
        )
        kmet = prefix_collapse_by_k(
            seqs_list,
            mask_list,
            k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            invalid=invalid_flags,
        )

        log = {
            "prefix_pos/top1_auc_val": float(pos.top1_auc),
            "prefix_pos/top1_auc_correct_val": float(pos.top1_auc_correct),
        }
        self._log_metrics(log, on_epoch=True, sync_dist=True)
        self._log_wandb_prefix_tables("prefix_pos_val", pos, kmet)
        self._log_wandb_length_metrics(
            "length_metrics_val",
            self.val_len_counts,
            self.val_score_sum_by_len,
            self.val_score_count_by_len,
            diversity_by_len,
            self.val_log_pterm_sum,
            self.val_log_pterm_count,
        )
        self._log_wandb_length_metrics(
            "length_metrics_val_valid",
            self.val_len_counts_valid,
            self.val_score_sum_by_len_valid,
            self.val_score_count_by_len_valid,
            diversity_by_len_valid,
            {},
            {},
        )

        if self.val_samples_table:
            samples_table = pd.DataFrame(self.val_samples_table)
        else:
            samples_table = pd.DataFrame(
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
        _root_dir = os.path.join(self.trainer.default_root_dir, "validation_samples")
        os.makedirs(_root_dir, exist_ok=True)
        samples_table.to_csv(
            os.path.join(
                _root_dir,
                f"samples_val_probes_{self.trainer.global_step}.csv",
            ),
            index=False,
        )

        if self.val_log_rs and self.val_log_pfss:
            log_rs = torch.cat(self.val_log_rs)
            log_pfss = torch.cat(self.val_log_pfss)
            if log_rs.numel() > 0 and log_pfss.numel() > 0:
                self._log_metrics(
                    {"val/Var(logR - logPf(s))": (log_rs - log_pfss).var()},
                    sync_dist=True,
                )

    def on_test_epoch_start(self):
        self.test_samples_ids.clear()
        self.test_samples_table.clear()
        self.test_log_rs.clear()
        self.test_log_pfss.clear()
        self.test_samples_valid_flags.clear()
        self.test_len_counts.clear()
        self.test_score_sum_by_len.clear()
        self.test_score_count_by_len.clear()
        self.test_len_counts_valid.clear()
        self.test_score_sum_by_len_valid.clear()
        self.test_score_count_by_len_valid.clear()
        self.test_log_pterm_sum.clear()
        self.test_log_pterm_count.clear()
        self.test_batch_diversity_sum = 0.0
        self.test_batch_diversity_count = 0
        self.test_batch_fp_div_internal_sum = 0.0
        self.test_batch_fp_div_internal_count = 0
        self.test_batch_fp_div_topk_sum = 0.0
        self.test_batch_fp_div_topk_count = 0
        self.test_text_diversity_sum = 0.0
        self.test_text_diversity_count = 0
        self.test_condvar_tokens.clear()
        self.test_condvar_log_pf_steps.clear()
        self.test_condvar_log_pterm.clear()
        self.test_condvar_log_r.clear()
        self.test_condvar_tau.clear()
        self.test_condvar_ref_log_pf_steps.clear()
        self.test_condvar_ref_log_pterm.clear()

    def on_test_epoch_end(self):
        cond_var_stats: dict[int, dict[str, float]] = {}
        diversity = self._calculate_diversity_ragged(self.test_samples_ids)
        self._log_metrics({"test/diversity": diversity}, sync_dist=True, on_epoch=True)
        valid_test_sequences = self._filter_valid_sequences(
            self.test_samples_ids, self.test_samples_valid_flags
        )
        diversity_test_valid = self._calculate_diversity_ragged(valid_test_sequences)
        self._log_metrics(
            {"test/diversity_valid": diversity_test_valid}, sync_dist=True, on_epoch=True
        )
        if self.test_batch_diversity_count > 0:
            self._log_metrics(
                {
                    "test/diversity_batch_mean": self.test_batch_diversity_sum
                    / float(self.test_batch_diversity_count)
                },
                sync_dist=True,
                on_epoch=True,
            )
        if self.sequence_diversity is not None and self.test_text_diversity_count > 0:
            self._log_metrics(
                {
                    "test/diversity_text_batch_mean": self.test_text_diversity_sum
                    / float(self.test_text_diversity_count)
                },
                sync_dist=True,
                on_epoch=True,
            )
        if self.test_batch_fp_div_internal_count > 0:
            mean_internal = self.test_batch_fp_div_internal_sum / float(
                self.test_batch_fp_div_internal_count
            )
            self._log_metrics(
                {"test/fp_div_internal_valid_batch_mean": mean_internal},
                sync_dist=True,
                on_epoch=True,
            )
        if self.test_batch_fp_div_topk_count > 0:
            mean_topk = self.test_batch_fp_div_topk_sum / float(self.test_batch_fp_div_topk_count)
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
        if self.sequence_diversity is not None:
            decoded_all = [
                self.tokenizer.decode(seq, skip_special_tokens=True)
                for seq in self.test_samples_ids
            ]
            if len(decoded_all) > 1:
                text_div_epoch = self.sequence_diversity(decoded_all)
                if text_div_epoch is not None:
                    self._log_metrics(
                        {"test/diversity_text_epoch": text_div_epoch},
                        sync_dist=True,
                        on_epoch=True,
                    )
        fp_div = self._compute_global_fp_diversity(self.test_samples_ids)
        if fp_div:
            self._log_metrics(
                {
                    "test/fp_div_internal_valid": fp_div.get("fp_div_internal_valid", 0.0),
                    "test/fp_div_topk_valid": fp_div.get("fp_div_topk_valid", 0.0),
                },
                sync_dist=True,
                on_epoch=True,
            )
        self._log_wandb_length_metrics(
            "length_metrics_test",
            self.test_len_counts,
            self.test_score_sum_by_len,
            self.test_score_count_by_len,
            diversity_by_len,
            self.test_log_pterm_sum,
            self.test_log_pterm_count,
        )
        self._log_wandb_length_metrics(
            "length_metrics_test_valid",
            self.test_len_counts_valid,
            self.test_score_sum_by_len_valid,
            self.test_score_count_by_len_valid,
            diversity_by_len_valid,
            {},
            {},
        )

        eos = self.end_of_sentence_token_id
        valid_flags = torch.tensor(self.test_samples_valid_flags)
        seqs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in self.test_samples_ids],
            batch_first=True,
            padding_value=eos,
        )
        active_before = compute_active_before(seqs, eos=eos)
        non_eos = seqs != eos
        mask_noeos = active_before & non_eos
        seqs_list = seqs.detach().cpu().tolist()
        mask_list = mask_noeos.detach().cpu().tolist()
        invalid_flags = (~valid_flags).tolist()
        pos = prefix_collapse_by_position(
            seqs_list, mask_list, collapse_thr=0.95, invalid=invalid_flags
        )
        kmet = prefix_collapse_by_k(
            seqs_list,
            mask_list,
            k_list=list(
                range(
                    self.constraint_config.min_sentence_len,
                    self.constraint_config.max_sentence_len + 1,
                )
            ),
            invalid=invalid_flags,
        )

        log = {
            "prefix_pos/top1_auc_test": float(pos.top1_auc),
            "prefix_pos/top1_auc_correct_test": float(pos.top1_auc_correct),
        }
        self._log_metrics(log, on_epoch=True, sync_dist=True)
        self._log_wandb_prefix_tables("prefix_pos_test", pos, kmet)

        cond_var_log, cond_var_json = self._compute_test_conditional_variance_metrics()
        if cond_var_log:
            self._log_metrics(cond_var_log, sync_dist=True, on_epoch=True)

        if self.test_samples_table:
            samples_table = pd.DataFrame(self.test_samples_table)
        else:
            samples_table = pd.DataFrame(
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
        _root_dir = self._repeat_dir("test_samples")
        os.makedirs(_root_dir, exist_ok=True)
        samples_table.to_csv(
            os.path.join(
                _root_dir,
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

        # save as json file
        callback_metrics = getattr(self.trainer, "callback_metrics", {})
        if callback_metrics:
            metrics = {}
            for key, value in callback_metrics.items():
                if not isinstance(key, str) or not key.startswith("test/"):
                    continue
                if isinstance(value, torch.Tensor):
                    if value.ndim == 0:
                        metrics[key] = float(value.detach().cpu().item())
                    else:
                        metrics[key] = value.detach().cpu().tolist()
                elif isinstance(value, (float, int)):
                    metrics[key] = float(value)
                elif isinstance(value, bool):
                    metrics[key] = bool(value)
            if metrics:
                epoch = int(getattr(self.trainer, "current_epoch", 0))
                global_step = int(getattr(self.trainer, "global_step", 0))
                metrics["epoch"] = epoch
                metrics["global_step"] = global_step
                if fp_div:
                    metrics["test/fp_div_internal_valid"] = fp_div.get(
                        "fp_div_internal_valid", 0.0
                    )
                    metrics["test/fp_div_topk_valid"] = fp_div.get("fp_div_topk_valid", 0.0)
                    metrics["test/validator/fp_div_internal_valid"] = fp_div.get(
                        "fp_div_internal_valid", 0.0
                    )
                    metrics["test/validator/fp_div_topk_valid"] = fp_div.get(
                        "fp_div_topk_valid", 0.0
                    )
                metrics["len_counts"] = self.test_len_counts
                metrics["score_sum_by_len"] = self.test_score_sum_by_len
                metrics["score_count_by_len"] = self.test_score_count_by_len
                score_mean_by_len = {}
                for length, total in self.test_score_sum_by_len.items():
                    count = self.test_score_count_by_len.get(length, 0)
                    score_mean_by_len[int(length)] = float(total) / float(count) if count else 0.0
                metrics["score_mean_by_len"] = score_mean_by_len
                metrics["len_counts_valid"] = self.test_len_counts_valid
                metrics["score_sum_by_len_valid"] = self.test_score_sum_by_len_valid
                metrics["score_count_by_len_valid"] = self.test_score_count_by_len_valid
                score_mean_by_len_valid = {}
                for length, total in self.test_score_sum_by_len_valid.items():
                    count = self.test_score_count_by_len_valid.get(length, 0)
                    score_mean_by_len_valid[int(length)] = (
                        float(total) / float(count) if count else 0.0
                    )
                metrics["score_mean_by_len_valid"] = score_mean_by_len_valid
                metrics["diversity_by_len"] = diversity_by_len
                metrics["diversity_valid"] = diversity_test_valid
                metrics["diversity_by_len_valid"] = diversity_by_len_valid
                metrics["log_pterm_sum"] = self.test_log_pterm_sum
                metrics["log_pterm_count"] = self.test_log_pterm_count
                log_pterm_by_len = {}
                for length, total in self.test_log_pterm_sum.items():
                    count = self.test_log_pterm_count.get(length, 0)
                    log_pterm_by_len[int(length)] = float(total) / float(count) if count else 0.0
                metrics["log_pterm_by_len"] = log_pterm_by_len
                metrics["pterm_by_len"] = {
                    length: float(np.exp(val)) for length, val in log_pterm_by_len.items()
                }
                if cond_var_json:
                    metrics["prefix_conditional_variance"] = cond_var_json

                exp_name = None
                hparams = getattr(self, "hparams", None)
                if hparams is not None:
                    if isinstance(hparams, dict):
                        exp_name = hparams.get("exp_name")
                    else:
                        exp_name = getattr(hparams, "exp_name", None)
                if not exp_name:
                    exp = getattr(getattr(self, "logger", None), "experiment", None)
                    exp_name = getattr(exp, "name", None)
                if not exp_name:
                    exp_name = "exp"
                exp_name = str(exp_name).replace(" ", "_")
                metrics["exp_name"] = exp_name

                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                _root_dir = self._repeat_eval_dir()
                os.makedirs(_root_dir, exist_ok=True)
                out_path = os.path.join(
                    _root_dir,
                    f"test_metrics_{exp_name}_epoch_{epoch}_step_{global_step}{self._test_repeat_suffix()}.json",
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, sort_keys=True)

    def on_train_start(self):
        self._init_coverage_reference()
        val_dataset = self.trainer.datamodule.val_dataloader().dataset
        n_probe = min(10, int(len(val_dataset)))
        probe_idx = random.sample(range(len(val_dataset)), n_probe) if n_probe > 0 else []
        val_probes = torch.utils.data.Subset(val_dataset, probe_idx)
        if val_probes is not None:
            self.val_probes = val_probes
            samples_table = self.sample_probes(val_probes)
            samples_table.to_csv(
                os.path.join(
                    self.trainer.default_root_dir,
                    f"samples_val_probes_wo_train_{self.trainer.global_step}.csv",
                ),
                index=False,
            )

        if self.skip_baseline_sampling:
            return

        val_data = self.trainer.datamodule.val_dataloader().dataset
        samples = {}
        for idx, prompt in enumerate(val_data):
            prompt_tensor = prompt["encoded_prompt"]
            samples_ = self.sample_baselines(prompt_tensor.to(self.device), n_samples=8)

            for method, data in samples_.items():
                if method in samples:
                    samples[method]["sample"].extend(data["sample"])
                else:
                    samples[method] = data.copy()

            if idx == 10:
                break

        pd.DataFrame(samples).T.to_csv(
            os.path.join(
                self.trainer.default_root_dir,
                f"samples_baselines_{self.trainer.current_epoch}.csv",
            ),
            index=True,
        )

    # ------------------------------------------------------------------ #
    # Sampling utilities
    # ------------------------------------------------------------------ #

    def sample_probes(self, probes, n_samples=4, return_metrics: bool = False):
        """Sample and decode probe sequences for logging.

        Args:
            probes: List of probe data items.
            n_samples: Number of samples per probe (default: 4).
            return_metrics: Whether to return logR/logPf(s) tensors for diagnostics.

        Returns:
            DataFrame with sampled sequences and their metrics.
            If return_metrics=True, also returns (log_rs, log_pfss).
        """
        samples = []
        log_rs = []
        log_pfss = []
        device = self.device
        for probe in probes:
            encoded_prompt = probe["encoded_prompt"]
            prompt_len = encoded_prompt.shape[-1]
            # Move all tensors to device efficiently
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

            if return_metrics:
                log_pfs, log_r_val, _, _ = get_termination_vals(
                    generated_text=generated_text,
                    log_pf=log_pf,
                    log_pterm=log_pterm,
                    log_r=log_r,
                    log_r_unpenalized=log_r_unpenalized,
                    termination_token_id=self.end_of_sentence_token_id,
                    prompt_len=prompt_len,
                )
                if log_pfs is not None:
                    log_rs.append(log_r_val)
                    log_pfss.append(log_pfs)

            generated_sequences = [
                self._decode_generated_tokens(text[prompt_len:], skip_special_tokens=False)
                for text in generated_text
            ]

            if result_dict.get("full_tokens", None) is not None:
                generated_sequences = result_dict["full_tokens"]

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
        df = pd.DataFrame(samples)
        if return_metrics:
            if log_rs:
                log_rs_out = torch.cat(log_rs)
                log_pfss_out = torch.cat(log_pfss)
            else:
                log_rs_out = torch.tensor([], device=self.device)
                log_pfss_out = torch.tensor([], device=self.device)
            return df, log_rs_out, log_pfss_out
        return df

    def sample_baselines(self, prompt, n_samples=4):
        assert prompt.ndim == 2

        def generate(prompt_tensor, **kwargs):
            with torch.no_grad():
                lora_to_base(self.net)

                try:
                    self.pre_grammar_processor.set_return_dict(False)
                except AttributeError:
                    if self.pre_grammar_processor is not None:
                        self.pre_grammar_processor.return_dict = False

                generated_text = self.net.generate(
                    prompt_tensor,
                    min_new_tokens=self.constraint_config.min_sentence_len,
                    max_new_tokens=self.constraint_config.max_sentence_len + 1,
                    eos_token_id=self.end_of_sentence_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    forced_eos_token_id=self.end_of_sentence_token_id,
                    suppress_tokens=None,
                    logits_processor=[self.pre_grammar_processor]
                    if self.pre_grammar_processor is not None
                    else None,
                    **kwargs,
                )
                base_to_lora(self.net)

                try:
                    self.pre_grammar_processor.set_return_dict(True)
                except AttributeError:
                    if self.pre_grammar_processor is not None:
                        self.pre_grammar_processor.return_dict = True

                generated_text = generated_text[:, prompt_tensor.shape[1] :]
                generated_text = torch.where(
                    generated_text == self.tokenizer.eos_token_id,
                    self.end_of_sentence_token_id,
                    generated_text,
                )
                generated_text = self.tokenizer.batch_decode(generated_text)
                eos_piece = self.tokenizer.decode(
                    [int(self.end_of_sentence_token_id)], skip_special_tokens=False
                )
                eos_piece = eos_piece.strip()
                if eos_piece:
                    stripped: list[str] = []
                    for text in generated_text:
                        t = text.rstrip()
                        if t.endswith(eos_piece):
                            t = t[: -len(eos_piece)]
                        stripped.append(t)
                    generated_text = stripped
                return {"sample": generated_text}

        samples = {}

        samples["beam"] = generate(
            prompt=prompt,
            do_sample=False,
            num_beams=n_samples * 5,
            length_penalty=0.0,
        )
        samples["beam [fair]"] = generate(
            prompt=prompt,
            do_sample=False,
            num_beams=n_samples,
            length_penalty=0.0,
        )
        samples["diverse beam"] = generate(
            prompt=prompt,
            do_sample=False,
            num_beams=n_samples * 5,
            num_beam_groups=n_samples,
            num_return_sequences=n_samples,
            diversity_penalty=1.0,
            length_penalty=0.0,
        )
        samples["diverse beam [fair]"] = generate(
            prompt=prompt,
            do_sample=False,
            num_beams=n_samples,
            num_beam_groups=n_samples,
            num_return_sequences=n_samples,
            diversity_penalty=1.0,
            length_penalty=0.0,
        )
        samples["LM"] = generate(
            prompt=prompt,
            do_sample=False,
            num_return_sequences=1,
            top_k=0,
        )
        samples["LM tempered"] = generate(
            prompt=prompt,
            do_sample=False,
            num_return_sequences=1,
            top_k=0,
            temperature=2.0,
        )
        samples["greedy"] = generate(
            prompt=prompt,
            do_sample=False,
        )
        samples["nucleus"] = generate(
            prompt=prompt,
            do_sample=False,
            num_return_sequences=1,
            top_k=0,
            top_p=0.95,
        )

        return samples

    # ------------------------------------------------------------------ #
    # Lightning plumbing
    # ------------------------------------------------------------------ #

    def setup(self, stage: str) -> None:
        if self.use_compile and stage == "fit":
            self.net = torch.compile(self.net)
            self.net_frozen = torch.compile(self.net_frozen)

    def configure_optimizers(self) -> dict[str, Any]:
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
    # Helpers
    # --------------------------------------------------------------------- #

    def _load_token_masks(self):
        tokens_path = getattr(self.constraint_config, "legal_tokens", None)
        if tokens_path and os.path.exists(tokens_path):
            return prepare_token_mask(self.tokenizer, tokens_path)

        if tokens_path:
            log.warning("Legal tokens file not found: %s", tokens_path)

        # Fallback: support explicit illegal token strings (common for text tasks).
        illegal_tokens = getattr(self.constraint_config, "illegal_tokens", None)
        if isinstance(illegal_tokens, (list, tuple)) and len(illegal_tokens) > 0:
            vocab_size = len(self.tokenizer)
            legal_mask = torch.ones(vocab_size, dtype=torch.bool)
            illegal_mask = torch.zeros(vocab_size, dtype=torch.bool)
            bad_ids: list[int] = []
            for tok in illegal_tokens:
                try:
                    ids = self.tokenizer.encode(str(tok), add_special_tokens=False)
                except Exception:
                    continue
                if len(ids) == 1:
                    bad_ids.append(int(ids[0]))
            if bad_ids:
                illegal_mask[bad_ids] = True
                # No explicit legal id list in this mode.
                return legal_mask, illegal_mask, None

        return None, None, None

    def _build_pre_grammar_processor(self, parsed_grammar):
        processor_type = getattr(self.constraint_config, "processor_type", "none")
        if processor_type == "none":
            return None
        if processor_type == "general":
            if self.grammar is None:
                log.warning("Grammar parsing failed with current tokenizer, disabling general processor")
                return None
            return GrammarConstrainedLogitsProcessor(self.grammar)

        processor_map = {
            "prefix": GrammarIncrementalLogitsProcessorGeneral,
            "prefix_enhanced": GrammarIncrementalLogitsProcessorSampleEnhanced,
        }

        processor_cls = processor_map.get(processor_type)
        if processor_cls is None:
            raise ValueError(f"Unsupported processor type: {processor_type}")

        return processor_cls(
            parsed_grammar,
            tokenizer=self.tokenizer,
            nice_token_ids_list=self.legal_token_ids_list,
            execution_mode=self.constraint_config.parse_mode,
        )

    def _setup_grammar_processors(self):
        with open(self.constraint_config.grammar_path) as file:
            grammar_str = file.read()

        parsed_grammar = parse_ebnf(grammar_str)
        self.string_grammar = StringRecognizer(
            parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"]
        )
        try:
            self.grammar = IncrementalGrammarConstraint(grammar_str, "root", self.tokenizer)
        except Exception:
            self.grammar = None
            print("Grammar parsing failed with current tokenizer, disable general processor")

        self.pre_grammar_processor = self._build_pre_grammar_processor(parsed_grammar)

    def _prepare_metric(self, value: Any, sync_dist: bool) -> Any:
        if not sync_dist:
            return value
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, (float, int)):
            return torch.tensor(value, device=self.device)
        return value

    def _normalize_sequence_for_coverage(
        self, sequence: list[int] | torch.Tensor
    ) -> tuple[int, ...] | None:
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.detach().cpu().tolist()
        if not sequence:
            return None
        token_ids = [int(x) for x in sequence]
        token_ids = self._strip_eos_token_ids(token_ids)
        token_ids = self._strip_special_token_ids(token_ids)
        if not token_ids:
            return None
        return tuple(token_ids)

    def _iter_coverage_reference_sequences(self, reference: Any) -> list[list[int]]:
        sequences: list[list[int]] = []
        if isinstance(reference, torch.Tensor):
            if reference.dim() == 2:
                sequences = reference.detach().cpu().tolist()
            elif reference.dim() == 1:
                sequences = [reference.detach().cpu().tolist()]
        elif isinstance(reference, (list, tuple)):
            for entry in reference:
                if isinstance(entry, torch.Tensor):
                    sequences.append(entry.detach().cpu().tolist())
                elif isinstance(entry, (list, tuple)):
                    sequences.append([int(x) for x in entry])
        return sequences

    def _init_coverage_reference(self) -> None:
        if not self._coverage_enabled or self._coverage_initialized:
            return

        reference = None
        source = self.coverage_config.get("reference_source", "dataset_buffer")
        if source == "dataset_buffer":
            datamodule = getattr(self.trainer, "datamodule", None)
            if datamodule is not None:
                reference = getattr(datamodule, "buffer_sample", None)
                if reference is None:
                    reference = getattr(datamodule, "dataset_buffer", None)

        if reference is None:
            reference_path = self.coverage_config.get("reference_path")
            if reference_path and os.path.exists(reference_path):
                reference = torch.load(reference_path, map_location="cpu")

        reference_set: set[tuple[int, ...]] = set()
        for seq in self._iter_coverage_reference_sequences(reference):
            norm = self._normalize_sequence_for_coverage(seq)
            if norm is not None:
                reference_set.add(norm)

        self._coverage_reference_set = reference_set
        self._coverage_total = len(reference_set)
        self._coverage_initialized = True

    def _update_coverage_from_tokens(self, tokens: torch.Tensor) -> None:
        if not self._coverage_enabled:
            return
        if not self._coverage_initialized:
            self._init_coverage_reference()
        if not self._coverage_reference_set:
            return

        sequences = self._strip_eos_from_batch(tokens)
        self._coverage_total_samples += len(sequences)
        for seq in sequences:
            norm = self._normalize_sequence_for_coverage(seq)
            if norm is None:
                continue
            if norm in self._coverage_reference_set:
                self._coverage_seen_set.add(norm)

        coverage_num = len(self._coverage_seen_set)
        coverage_total = self._coverage_total
        coverage_rate = coverage_num / float(coverage_total) if coverage_total else 0.0
        unique_correct_rate = (
            coverage_num / float(self._coverage_total_samples)
            if self._coverage_total_samples
            else 0.0
        )
        if (
            self._coverage_steps_to_full is None
            and coverage_total
            and coverage_num >= coverage_total
        ):
            self._coverage_steps_to_full = int(self.global_step)
        steps_to_full = (
            self._coverage_steps_to_full if self._coverage_steps_to_full is not None else -1
        )

        self._log_metrics(
            {
                "coverage/overall_rate": coverage_rate,
                "coverage/overall_num": coverage_num,
                "coverage/unique_correct_rate": unique_correct_rate,
                "coverage/steps_to_full": steps_to_full,
            },
            on_step=True,
            sync_dist=True,
        )

    def _strip_special_token_ids(self, token_ids: list[int]) -> list[int]:
        special_ids = set(self.tokenizer.all_special_ids)
        return [tok for tok in token_ids if tok not in special_ids]

    def _strip_eos_token_ids(self, token_ids: list[int]) -> list[int]:
        eos_id = int(self.end_of_sentence_token_id)
        try:
            eos_pos = token_ids.index(eos_id)
        except ValueError:
            return token_ids
        return token_ids[:eos_pos]

    def _strip_eos_from_batch(self, token_batch: torch.Tensor) -> list[list[int]]:
        if token_batch is None:
            return []
        rows = token_batch.detach().cpu().tolist()
        return [self._strip_eos_token_ids([int(t) for t in row]) for row in rows]

    def _filter_valid_sequences(
        self, sequences: list[list[int]], valid_flags: list[bool] | torch.Tensor | None
    ) -> list[list[int]]:
        if not sequences or valid_flags is None:
            return []
        if isinstance(valid_flags, torch.Tensor):
            valid_flags = valid_flags.detach().cpu().tolist()
        if not valid_flags:
            return []
        n = min(len(sequences), len(valid_flags))
        return [sequences[i] for i in range(n) if bool(valid_flags[i])]

    def _calculate_diversity_ragged(self, sequences: list[list[int]]) -> float:
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

    def _compute_global_fp_diversity(
        self,
        token_ids_list: list[list[int]],
    ) -> dict[str, float]:
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
            cleaned = self._strip_special_token_ids(seq)
            cleaned = self._strip_eos_token_ids(cleaned)
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
            metrics = validator.smiles_accuracy(tokens, self.tokenizer, return_hist=False)
        except Exception:
            try:
                metrics = validator.accuracy(tokens, self.tokenizer)
            except Exception:
                return {}

        return {
            "fp_div_internal_valid": float(metrics.get("fp_div_internal_valid", 0.0)),
            "fp_div_topk_valid": float(metrics.get("fp_div_topk_valid", 0.0)),
        }

    def _lengths_from_tokens(self, tokens: torch.Tensor) -> list[int]:
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

    @staticmethod
    def _accumulate_length_stats(
        lengths: list[int],
        scores: list[float] | torch.Tensor | None,
        counts: dict[int, int],
        score_sums: dict[int, float],
        score_counts: dict[int, int],
        valid_flags: list[bool] | None = None,
    ) -> None:
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
        sums: dict[int, float],
        counts: dict[int, int],
    ) -> None:
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
            val = float(log_pterm_cpu[idx, t_idx].item())
            sums[length] = sums.get(length, 0.0) + val
            counts[length] = counts.get(length, 0) + 1

    def _compute_tau_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute tau as the first EOS position (or last index if none)."""
        if tokens is None:
            return torch.tensor([], dtype=torch.long, device=self.device)
        if tokens.numel() == 0:
            return torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
        max_len = tokens.shape[1]
        eos_mask = tokens == self.end_of_sentence_token_id
        idxs = torch.arange(max_len, device=tokens.device).unsqueeze(0).expand_as(tokens)
        first_eos = torch.where(eos_mask, idxs, torch.full_like(idxs, max_len))
        tau = first_eos.min(dim=1).values
        tau = torch.clamp(tau, max=max_len - 1)
        return tau.to(dtype=torch.long)

    def _cache_test_conditional_variance_inputs(
        self,
        *,
        tokens: torch.Tensor,
        log_pf_steps: torch.Tensor,
        log_pterm: torch.Tensor,
        log_r: torch.Tensor,
        tau: torch.Tensor,
        ref_log_pf_steps: torch.Tensor | None = None,
        ref_log_pterm: torch.Tensor | None = None,
    ) -> None:
        """Cache per-sample tensors required for prefix-conditional variance."""
        if (
            tokens is None
            or log_pf_steps is None
            or log_pterm is None
            or log_r is None
            or tau is None
        ):
            return

        tokens_cpu = tokens.detach().cpu()
        log_pf_cpu = log_pf_steps.detach().cpu()
        log_pterm_cpu = log_pterm.detach().cpu()
        log_r_cpu = log_r.detach().cpu()
        tau_cpu = tau.detach().cpu()
        ref_log_pf_cpu = ref_log_pf_steps.detach().cpu() if ref_log_pf_steps is not None else None
        ref_log_pterm_cpu = ref_log_pterm.detach().cpu() if ref_log_pterm is not None else None

        n = min(
            tokens_cpu.shape[0],
            log_pf_cpu.shape[0],
            log_pterm_cpu.shape[0],
            log_r_cpu.shape[0],
            tau_cpu.shape[0],
            ref_log_pf_cpu.shape[0] if ref_log_pf_cpu is not None else tokens_cpu.shape[0],
            ref_log_pterm_cpu.shape[0] if ref_log_pterm_cpu is not None else tokens_cpu.shape[0],
        )
        if n <= 0:
            return
        tokens_cpu = tokens_cpu[:n]
        log_pf_cpu = log_pf_cpu[:n]
        log_pterm_cpu = log_pterm_cpu[:n]
        log_r_cpu = log_r_cpu[:n]
        tau_cpu = tau_cpu[:n]
        if ref_log_pf_cpu is not None:
            ref_log_pf_cpu = ref_log_pf_cpu[:n]
        if ref_log_pterm_cpu is not None:
            ref_log_pterm_cpu = ref_log_pterm_cpu[:n]

        max_len = min(
            tokens_cpu.shape[1],
            log_pterm_cpu.shape[1],
            log_r_cpu.shape[1],
            log_pf_cpu.shape[1] + 1,
            ref_log_pterm_cpu.shape[1]
            if ref_log_pterm_cpu is not None
            else log_pterm_cpu.shape[1],
            (ref_log_pf_cpu.shape[1] + 1)
            if ref_log_pf_cpu is not None
            else log_pf_cpu.shape[1] + 1,
        )
        if max_len <= 1:
            return

        tokens_cpu = tokens_cpu[:, :max_len]
        log_pterm_cpu = log_pterm_cpu[:, :max_len]
        log_r_cpu = log_r_cpu[:, :max_len]
        log_pf_cpu = log_pf_cpu[:, : max_len - 1]
        tau_cpu = torch.clamp(tau_cpu, max=max_len - 1)
        if ref_log_pf_cpu is not None:
            ref_log_pf_cpu = ref_log_pf_cpu[:, : max_len - 1]
        if ref_log_pterm_cpu is not None:
            ref_log_pterm_cpu = ref_log_pterm_cpu[:, :max_len]

        self.test_condvar_tokens.extend([row.clone() for row in tokens_cpu])
        self.test_condvar_log_pf_steps.extend([row.clone() for row in log_pf_cpu])
        self.test_condvar_log_pterm.extend([row.clone() for row in log_pterm_cpu])
        self.test_condvar_log_r.extend([row.clone() for row in log_r_cpu])
        self.test_condvar_tau.extend([int(x) for x in tau_cpu.tolist()])
        if ref_log_pf_cpu is not None:
            self.test_condvar_ref_log_pf_steps.extend([row.clone() for row in ref_log_pf_cpu])
        if ref_log_pterm_cpu is not None:
            self.test_condvar_ref_log_pterm.extend([row.clone() for row in ref_log_pterm_cpu])

    def _default_condvar_m_values(self) -> list[int]:
        """Default prefix cut positions for conditional variance diagnostics."""
        return [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]

    def _compute_test_conditional_variance_metrics(self) -> tuple[dict[str, float], dict]:
        """Compute method-aligned conditional variance metrics based on exp_name."""
        if not self.test_condvar_tokens:
            return {}, {}
        try:
            tokens = torch.stack(self.test_condvar_tokens, dim=0)
            log_pf_steps = torch.stack(self.test_condvar_log_pf_steps, dim=0)
            log_pterm = torch.stack(self.test_condvar_log_pterm, dim=0)
            log_r = torch.stack(self.test_condvar_log_r, dim=0)
            tau = torch.tensor(self.test_condvar_tau, dtype=torch.long)
            ref_log_pf_steps = (
                torch.stack(self.test_condvar_ref_log_pf_steps, dim=0)
                if self.test_condvar_ref_log_pf_steps
                else None
            )
            ref_log_pterm = (
                torch.stack(self.test_condvar_ref_log_pterm, dim=0)
                if self.test_condvar_ref_log_pterm
                else None
            )
        except Exception:
            return {}, {}

        max_len = min(
            tokens.shape[1],
            log_pterm.shape[1],
            log_r.shape[1],
            log_pf_steps.shape[1] + 1,
            ref_log_pterm.shape[1] if ref_log_pterm is not None else log_pterm.shape[1],
            (ref_log_pf_steps.shape[1] + 1)
            if ref_log_pf_steps is not None
            else log_pf_steps.shape[1] + 1,
        )
        if max_len <= 1:
            return {}, {}

        tokens = tokens[:, :max_len]
        log_pterm = log_pterm[:, :max_len]
        log_r = log_r[:, :max_len]
        log_pf_steps = log_pf_steps[:, : max_len - 1]
        tau = torch.clamp(tau, max=max_len - 1)
        if ref_log_pf_steps is not None:
            ref_log_pf_steps = ref_log_pf_steps[:, : max_len - 1]
        if ref_log_pterm is not None:
            ref_log_pterm = ref_log_pterm[:, :max_len]

        m_values = [m for m in self._default_condvar_m_values() if 0 <= m < max_len]
        if not m_values:
            return {}, {}

        # method detection
        exp_name = None
        hparams = getattr(self, "hparams", None)
        if hparams is not None:
            if isinstance(hparams, dict):
                exp_name = hparams.get("exp_name")
            else:
                exp_name = getattr(hparams, "exp_name", None)
        if exp_name is None:
            exp = getattr(getattr(self, "logger", None), "experiment", None)
            exp_name = getattr(exp, "name", "")
        exp_name = str(exp_name or "").lower()
        want_raptb = "raptb" in exp_name
        want_subtb = "subtb" in exp_name

        log_payload: dict[str, float] = {}
        json_payload: dict = {}

        # TB baseline
        tb_targets = compute_tb_targets(
            tokens=tokens,
            log_pf_steps=log_pf_steps,
            log_pterm=log_pterm,
            log_r=log_r,
            tau=tau,
            m_values=m_values,
        )
        tb_json = {}
        for m, data in tb_targets.items():
            stats = grouped_weighted_var(data["keys"], data["targets"])
            if not stats:
                continue
            prefix = f"test/cond_var_Ym_m{m}"
            log_payload[prefix] = stats["cond_var"]
            log_payload[f"{prefix}_singleton_mass"] = stats["singleton_mass"]
            log_payload[f"{prefix}_max_group_mass"] = stats["max_group_mass"]
            log_payload[f"{prefix}_num_groups"] = stats["num_groups"]
            log_payload[f"{prefix}_finite_rate"] = stats["finite_rate"]
            tb_json[m] = stats
        if tb_json:
            json_payload["TB"] = tb_json

        # RapTB effective/hybrid
        if want_raptb:
            # compute valid_end
            eos = int(self.end_of_sentence_token_id)
            eos_or_after = (tokens[:, :-1] == eos).cumsum(dim=1) >= 1
            valid_end = ~eos_or_after
            log_pf_full = torch.cat([log_pf_steps, torch.zeros_like(log_pf_steps[:, :1])], dim=1)
            rap_targets = compute_raptb_targets(
                tokens=tokens,
                log_pf=log_pf_full,
                log_pterm=log_pterm,
                log_r=log_r,
                tau=tau,
                valid_end=valid_end,
                m_values=m_values,
                gamma=float(getattr(self.loss_fn, "gamma", 1.0)),
                k_min=int(getattr(self.loss_fn, "k_min", 1)),
                extra_absorb_eps=float(getattr(self.loss_fn, "extra_absorb_eps", 0.0)),
                soft_beta=float(getattr(self.loss_fn, "soft_beta", 1.0)),
                soft_rho=float(getattr(self.loss_fn, "soft_rho", 0.0)),
                target_mode=str(getattr(self.loss_fn, "target_mode", "future_max")),
                mix_weight=float(getattr(self.loss_fn, "mix_weight", 1.0)),
                ref_log_pf=ref_log_pf_steps,
                ref_log_pterm=ref_log_pterm,
                ref_scale=float(getattr(self.loss_fn, "ref_scale", 1.0)),
                max_prefix_len=int(getattr(self.loss_fn, "k_max", max_len - 1))
                if getattr(self.loss_fn, "k_max", None) is not None
                else None,
            )
            rap_json = {}
            for m, data in rap_targets.items():
                if "targets_eff" in data:
                    stats_eff = grouped_weighted_var(data["keys_eff"], data["targets_eff"])
                    if stats_eff:
                        prefix = f"test/cond_var_Yeff_m{m}"
                        log_payload[prefix] = stats_eff["cond_var"]
                        log_payload[f"{prefix}_singleton_mass"] = stats_eff["singleton_mass"]
                        log_payload[f"{prefix}_max_group_mass"] = stats_eff["max_group_mass"]
                        log_payload[f"{prefix}_num_groups"] = stats_eff["num_groups"]
                        log_payload[f"{prefix}_finite_rate"] = stats_eff["finite_rate"]
                        log_payload[f"{prefix}_apply_absorb_rate"] = data.get("apply_rate", 0.0)
                        rap_json.setdefault(m, {})["eff"] = stats_eff
                        rap_json[m]["apply_rate"] = data.get("apply_rate", 0.0)
                if "targets_hyb" in data:
                    stats_hyb = grouped_weighted_var(data["keys_hyb"], data["targets_hyb"])
                    if stats_hyb:
                        prefix = f"test/cond_var_Yhyb_m{m}"
                        log_payload[prefix] = stats_hyb["cond_var"]
                        log_payload[f"{prefix}_singleton_mass"] = stats_hyb["singleton_mass"]
                        log_payload[f"{prefix}_max_group_mass"] = stats_hyb["max_group_mass"]
                        log_payload[f"{prefix}_num_groups"] = stats_hyb["num_groups"]
                        log_payload[f"{prefix}_finite_rate"] = stats_hyb["finite_rate"]
                        log_payload[f"{prefix}_apply_absorb_rate"] = data.get("apply_rate", 0.0)
                        rap_json.setdefault(m, {})["hyb"] = stats_hyb
                        rap_json[m]["apply_rate"] = data.get("apply_rate", 0.0)
            if rap_json:
                json_payload["RapTB"] = rap_json

        # SubTB approximate (delta-based)
        if want_subtb:
            sub_targets = compute_subtb_targets_delta(
                tokens=tokens,
                log_pf=torch.cat([log_pf_steps, torch.zeros_like(log_pf_steps[:, :1])], dim=1),
                log_pterm=log_pterm,
                log_r=log_r,
                m_values=m_values,
            )
            sub_json = {}
            for m, data in sub_targets.items():
                stats = grouped_weighted_var(data["keys"], data["targets"])
                if not stats:
                    continue
                prefix = f"test/cond_var_subtb_delta_m{m}"
                log_payload[prefix] = stats["cond_var"]
                log_payload[f"{prefix}_singleton_mass"] = stats["singleton_mass"]
                log_payload[f"{prefix}_max_group_mass"] = stats["max_group_mass"]
                log_payload[f"{prefix}_num_groups"] = stats["num_groups"]
                log_payload[f"{prefix}_finite_rate"] = stats["finite_rate"]
                sub_json[m] = stats
            if sub_json:
                json_payload["SubTB_delta"] = sub_json

        return log_payload, json_payload

    def _test_repeat_suffix(self) -> str:
        return getattr(self, "test_repeat_suffix", "") or ""

    def _repeat_dir(self, base_name: str) -> str:
        suffix = self._test_repeat_suffix()
        name = f"{base_name}{suffix}" if suffix else base_name
        return os.path.join(self.trainer.default_root_dir, name)

    def _repeat_eval_dir(self) -> str:
        """Return repeat-scoped eval directory under trainer.default_root_dir/json."""
        return os.path.join(self.trainer.default_root_dir, "json")

    def _log_validator_core_metrics(
        self,
        prefix: str,
        validator_metric_dict: dict[str, Any] | None,
        *,
        sync_dist: bool,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
    ) -> None:
        if not validator_metric_dict:
            return
        scorer_name = None
        if (
            self.reward is not None
            and getattr(self.reward, "sentence_validator", None) is not None
        ):
            scorer_name = getattr(self.reward.sentence_validator, "scorer_name", None)

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
        if not validator_metric_dict:
            return
        metrics = {}
        skip_keys = {
            "len_tok_hist",
            "len_tok_valid_hist",
            "len_char_hist",
            "len_char_valid_hist",
            "score_hist",
        }
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

    def _log_wandb_prefix_tables(
        self,
        tag: str,
        pos,
        kmet: dict[int, dict[str, float]],
    ) -> None:
        if self.logger is None:
            return
        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return
        try:
            import wandb
        except Exception:
            return

        loggers = self.loggers if getattr(self, "loggers", None) else [self.logger]
        runs = []
        for lg in loggers:
            exp = getattr(lg, "experiment", None)
            if exp is not None and hasattr(exp, "log") and hasattr(exp, "define_metric"):
                runs.append(exp)
        if not runs:
            return

        epoch = int(getattr(self.trainer, "current_epoch", 0))

        # one-time: encourage scalar plots to use epoch as x-axis
        flag_name = f"_wandb_prefix_defined_{tag.replace('/', '_')}"
        if not getattr(self, flag_name, False):
            for run in runs:
                try:
                    run.define_metric(f"{tag}/*", step_metric="epoch")
                except Exception:
                    pass
            setattr(self, flag_name, True)

        payload = {"epoch": epoch}
        local_tag = tag.replace("/", "_")
        _root_dir = self._repeat_dir("prefix_tables")
        os.makedirs(_root_dir, exist_ok=True)

        ranges = getattr(self, "_wandb_prefix_axis_ranges", None)
        if ranges is None:
            ranges = {}
            setattr(self, "_wandb_prefix_axis_ranges", ranges)
        tag_ranges = ranges.setdefault(tag, {})

        def _update_range(key: str, values: list[float]) -> tuple[float, float] | None:
            if not values:
                return None
            vmin = float(min(values))
            vmax = float(max(values))
            cur = tag_ranges.get(key)
            if cur is None:
                tag_ranges[key] = [vmin, vmax]
            else:
                cur[0] = min(cur[0], vmin)
                cur[1] = max(cur[1], vmax)
            return float(tag_ranges[key][0]), float(tag_ranges[key][1])

        def _update_x_range(
            key_min: str, key_max: str, xs: list[int]
        ) -> tuple[float, float] | None:
            if not xs:
                return None
            vmin = float(min(xs))
            vmax = float(max(xs))
            cur_min = float(tag_ranges.get(key_min, vmin))
            cur_max = float(tag_ranges.get(key_max, vmax))
            tag_ranges[key_min] = min(cur_min, vmin)
            tag_ranges[key_max] = max(cur_max, vmax)
            return float(tag_ranges[key_min]), float(tag_ranges[key_max])

        def _apply_ylim(ax, rng: tuple[float, float] | None) -> None:
            if rng is None:
                return
            ymin, ymax = rng
            if ymin == ymax:
                pad = 1e-6 if ymin == 0.0 else abs(ymin) * 0.05
                ymin -= pad
                ymax += pad
            ax.set_ylim(ymin, ymax)

        def _log_pos_variant(
            *,
            suffix: str,  # "" or "_correct"
            top1_mass: list[float],
            entropy: list[float],
            eff_support: list[float],
            unique: list[int] | None,
            support: list[int] | None,
            correct_frac: list[float] | None = None,  # only for correct variant (optional)
        ) -> None:
            if not top1_mass:
                return
            T = min(len(top1_mass), len(entropy), len(eff_support))
            if T <= 0:
                return

            # ---- table ----
            cols = ["epoch", "t", "top1_mass", "entropy", "eff_support"]
            if unique is not None:
                cols.append("unique")
            if support is not None:
                cols.append("support")
            if correct_frac is not None:
                cols.append("correct_frac")

            rows = []
            for t in range(T):
                row = [
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

            pd.DataFrame(rows, columns=cols).to_csv(
                os.path.join(_root_dir, f"{local_tag}_pos{suffix}_{epoch}.csv"),
                index=False,
            )

            # ---- figure: 3 subplots ----
            try:
                import matplotlib.pyplot as plt

                xs = list(range(T))
                top1_vals = [float(v) for v in top1_mass[:T]]
                ent_vals = [float(v) for v in entropy[:T]]
                eff_vals = [float(v) for v in eff_support[:T]]

                fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

                axes[0].plot(xs, top1_vals)
                axes[0].set_ylabel("top1_mass")

                axes[1].plot(xs, ent_vals)
                axes[1].set_ylabel("entropy")

                axes[2].plot(xs, eff_vals)
                axes[2].set_ylabel("eff_support")
                axes[2].set_xlabel("t")

                _apply_ylim(axes[0], _update_range(f"pos{suffix}_top1_mass", top1_vals))
                _apply_ylim(axes[1], _update_range(f"pos{suffix}_entropy", ent_vals))
                _apply_ylim(axes[2], _update_range(f"pos{suffix}_eff_support", eff_vals))
                x_rng = _update_x_range(f"pos{suffix}_x_min", f"pos{suffix}_x_max", xs)
                if x_rng is not None:
                    for ax in axes:
                        ax.set_xlim(x_rng[0], x_rng[1])

                title_extra = "" if suffix == "" else " (correct-only)"
                fig.suptitle(f"{tag} / prefix-by-position{title_extra} (epoch={epoch})")
                fig.tight_layout()
                fig.set_dpi(300)

                key = f"{tag}/pos{suffix}_curves_img" if suffix else f"{tag}/pos_curves_img"
                if suffix:
                    key = f"{tag}/pos_correct_curves_img"
                payload[key] = wandb.Image(fig, file_type="jpg")
                plt.close(fig)
            except Exception:
                pass

        def _log_k_variant(*, suffix: str) -> None:
            if not kmet:
                return
            ks = sorted(kmet.keys())
            if not ks:
                return

            # pick fields
            if suffix == "":
                f_top1, f_top5, f_ent, f_eff, f_n, f_uniq = (
                    "top1",
                    "top5",
                    "entropy",
                    "eff",
                    "n",
                    "unique",
                )
            else:
                f_top1, f_top5, f_ent, f_eff, f_n, f_uniq = (
                    "top1_correct",
                    "top5_correct",
                    "entropy_correct",
                    "eff_correct",
                    "n_correct",
                    "unique_correct",
                )

            # if correct variant but none exists, skip quietly
            if suffix != "":
                any_has = False
                for k in ks:
                    v = kmet[k] or {}
                    if (f_n in v) or (f_top1 in v) or (f_top5 in v):
                        any_has = True
                        break
                if not any_has:
                    return

            cols = ["epoch", "k", "top1", "top5", "entropy", "eff", "n", "unique"]
            rows = []
            for k in ks:
                v = kmet[k] or {}
                rows.append(
                    [
                        epoch,
                        int(k),
                        float(v.get(f_top1, 0.0)),
                        float(v.get(f_top5, 0.0)),
                        float(v.get(f_ent, 0.0)),
                        float(v.get(f_eff, 0.0)),
                        float(v.get(f_n, 0.0)),
                        float(v.get(f_uniq, 0.0)),
                    ]
                )

            pd.DataFrame(rows, columns=cols).to_csv(
                os.path.join(_root_dir, f"{local_tag}_k{suffix}_{epoch}.csv"),
                index=False,
            )

            try:
                import matplotlib.pyplot as plt

                xs = [int(k) for k in ks]
                top1s = [float((kmet[k] or {}).get(f_top1, 0.0)) for k in ks]
                top5s = [float((kmet[k] or {}).get(f_top5, 0.0)) for k in ks]
                ents = [float((kmet[k] or {}).get(f_ent, 0.0)) for k in ks]
                effs = [float((kmet[k] or {}).get(f_eff, 0.0)) for k in ks]

                fig, axes = plt.subplots(4, 1, figsize=(7, 9), sharex=True)

                axes[0].plot(xs, top1s)
                axes[0].set_ylabel("top1")

                axes[1].plot(xs, top5s)
                axes[1].set_ylabel("top5")

                axes[2].plot(xs, ents)
                axes[2].set_ylabel("entropy")

                axes[3].plot(xs, effs)
                axes[3].set_ylabel("eff")
                axes[3].set_xlabel("k")

                _apply_ylim(axes[0], _update_range(f"k{suffix}_top1", top1s))
                _apply_ylim(axes[1], _update_range(f"k{suffix}_top5", top5s))
                _apply_ylim(axes[2], _update_range(f"k{suffix}_entropy", ents))
                _apply_ylim(axes[3], _update_range(f"k{suffix}_eff", effs))
                x_rng = _update_x_range(f"k{suffix}_x_min", f"k{suffix}_x_max", xs)
                if x_rng is not None:
                    for ax in axes:
                        ax.set_xlim(x_rng[0], x_rng[1])

                title_extra = "" if suffix == "" else " (correct-only)"
                fig.suptitle(f"{tag} / prefix-by-k{title_extra} (epoch={epoch})")
                fig.tight_layout()
                fig.set_dpi(300)

                key = f"{tag}/k_curves_img" if suffix == "" else f"{tag}/k_correct_curves_img"
                payload[key] = wandb.Image(fig, file_type="jpg")
                plt.close(fig)
            except Exception:
                pass

        # -------------------------
        # POS: all + correct
        # -------------------------
        if getattr(pos, "top1_mass", None):
            _log_pos_variant(
                suffix="",
                top1_mass=[float(v) for v in pos.top1_mass],
                entropy=[float(v) for v in getattr(pos, "entropy", [])],
                eff_support=[float(v) for v in getattr(pos, "eff_support", [])],
                unique=getattr(pos, "unique", None),
                support=getattr(pos, "support", None),
                correct_frac=None,
            )

        if getattr(pos, "top1_mass_correct", None):
            _log_pos_variant(
                suffix="_correct",
                top1_mass=[float(v) for v in pos.top1_mass_correct],
                entropy=[float(v) for v in getattr(pos, "entropy_correct", [])],
                eff_support=[float(v) for v in getattr(pos, "eff_support_correct", [])],
                unique=getattr(pos, "unique_correct", None),
                support=getattr(pos, "support_correct", None),
                correct_frac=getattr(pos, "correct_frac", None),
            )

        # -------------------------
        # K: all + correct
        # -------------------------
        if kmet:
            _log_k_variant(suffix="")
            _log_k_variant(suffix="_correct")

        # -------------------------
        # Scalars (all + correct)
        # -------------------------
        # if getattr(pos, "top1_auc", None) is not None:
        #     payload[f"{tag}/top1_auc"] = float(pos.top1_auc)

        # if getattr(pos, "top1_auc_correct", None) is not None:
        #     payload[f"{tag}/top1_auc_correct"] = float(pos.top1_auc_correct)

        # drop Nones
        payload = {k: v for k, v in payload.items() if v is not None}
        if len(payload) <= 1:
            return

        # IMPORTANT: do NOT set step manually
        for run in runs:
            try:
                run.log(payload)
            except Exception:
                pass

    def _log_wandb_length_metrics(
        self,
        tag: str,
        length_counts: dict[int, int],
        score_sums: dict[int, float],
        score_counts: dict[int, int],
        diversity_by_len: dict[int, float],
        log_pterm_sums: dict[int, float],
        log_pterm_counts: dict[int, int],
    ) -> None:
        if self.logger is None:
            return
        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return
        try:
            import wandb
        except Exception:
            return

        loggers = self.loggers if getattr(self, "loggers", None) else [self.logger]
        runs = []
        for lg in loggers:
            exp = getattr(lg, "experiment", None)
            if exp is not None and hasattr(exp, "log") and hasattr(exp, "define_metric"):
                runs.append(exp)
        if not runs:
            return

        epoch = int(getattr(self.trainer, "current_epoch", 0))
        flag_name = f"_wandb_length_defined_{tag.replace('/', '_')}"
        if not getattr(self, flag_name, False):
            for run in runs:
                try:
                    run.define_metric(f"{tag}/*", step_metric="epoch")
                except Exception:
                    pass
            setattr(self, flag_name, True)

        scorer_name = "score"
        if (
            self.reward is not None
            and getattr(self.reward, "sentence_validator", None) is not None
        ):
            scorer_name = getattr(self.reward.sentence_validator, "scorer_name", scorer_name)

        payload = {"epoch": epoch}

        ranges = getattr(self, "_wandb_length_axis_ranges", None)
        if ranges is None:
            ranges = {}
            setattr(self, "_wandb_length_axis_ranges", ranges)
        tag_ranges = ranges.setdefault(tag, {})

        def _update_range(key: str, values: list[float]) -> tuple[float, float] | None:
            if not values:
                return None
            vmin = float(min(values))
            vmax = float(max(values))
            cur = tag_ranges.get(key)
            if cur is None:
                tag_ranges[key] = [vmin, vmax]
            else:
                cur[0] = min(cur[0], vmin)
                cur[1] = max(cur[1], vmax)
            return float(tag_ranges[key][0]), float(tag_ranges[key][1])

        def _update_x_range(
            key_min: str, key_max: str, xs: list[int]
        ) -> tuple[float, float] | None:
            if not xs:
                return None
            vmin = float(min(xs))
            vmax = float(max(xs))
            cur_min = float(tag_ranges.get(key_min, vmin))
            cur_max = float(tag_ranges.get(key_max, vmax))
            tag_ranges[key_min] = min(cur_min, vmin)
            tag_ranges[key_max] = max(cur_max, vmax)
            return float(tag_ranges[key_min]), float(tag_ranges[key_max])

        def _apply_ylim(ax, rng: tuple[float, float] | None) -> None:
            if rng is None:
                return
            ymin, ymax = rng
            if ymin == ymax:
                pad = 1e-6 if ymin == 0.0 else abs(ymin) * 0.05
                ymin -= pad
                ymax += pad
            ax.set_ylim(ymin, ymax)

        def _plot_line(
            xs: list[int],
            ys: list[float],
            xlabel: str,
            ylabel: str,
            title: str,
            *,
            key_prefix: str,
        ):
            if not xs:
                return None
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(xs, ys)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                _apply_ylim(ax, _update_range(f"{key_prefix}_y", ys))
                x_rng = _update_x_range(f"{key_prefix}_x_min", f"{key_prefix}_x_max", xs)
                if x_rng is not None:
                    ax.set_xlim(x_rng[0], x_rng[1])
                fig.tight_layout()
                # set dpi to 300
                fig.set_dpi(300)
                return fig
            except Exception:
                return None

        if length_counts:
            lengths = sorted(length_counts.keys())
            total = float(sum(length_counts.values()))
            if total > 0:
                counts = [float(length_counts[length]) / total * 100.0 for length in lengths]
            else:
                counts = [0.0 for _ in lengths]
            fig = _plot_line(
                lengths,
                counts,
                "length",
                "percent",
                f"{tag} count by length (total={int(total)}) (epoch={epoch})",
                key_prefix="count_by_len",
            )
            if fig is not None:
                payload[f"{tag}/count_by_len"] = wandb.Image(fig, file_type="jpg")
                plt.close(fig)

            if score_counts:
                avg_scores = [
                    float(score_sums.get(length, 0.0) / max(1, score_counts.get(length, 0)))
                    for length in lengths
                ]
                fig = _plot_line(
                    lengths,
                    avg_scores,
                    "length",
                    f"{scorer_name}",
                    f"{tag} {scorer_name} by length (epoch={epoch})",
                    key_prefix="score_by_len",
                )
                if fig is not None:
                    payload[f"{tag}/{scorer_name}_by_len"] = wandb.Image(fig, file_type="jpg")
                    plt.close(fig)

        if diversity_by_len:
            lengths = sorted(diversity_by_len.keys())
            divs = [float(diversity_by_len[length]) for length in lengths]
            fig = _plot_line(
                lengths,
                divs,
                "length",
                "diversity",
                f"{tag} diversity by length (epoch={epoch})",
                key_prefix="diversity_by_len",
            )
            if fig is not None:
                payload[f"{tag}/diversity_by_len"] = wandb.Image(fig, file_type="jpg")
                plt.close(fig)

        if log_pterm_sums and log_pterm_counts:
            lengths = sorted(log_pterm_sums.keys())
            log_vals = [
                float(log_pterm_sums.get(length, 0.0) / log_pterm_counts.get(length, 1))
                if log_pterm_counts.get(length, 0) > 0
                else 0.0
                for length in lengths
            ]
            fig = _plot_line(
                lengths,
                log_vals,
                "length",
                "log_pterm",
                f"{tag} log_pterm by length (epoch={epoch})",
                key_prefix="log_pterm_by_len",
            )
            if fig is not None:
                payload[f"{tag}/log_pterm_by_len"] = wandb.Image(fig, file_type="jpg")
                plt.close(fig)
            pterm_vals = [float(np.exp(v)) for v in log_vals]
            fig = _plot_line(
                lengths,
                pterm_vals,
                "length",
                "pterm",
                f"{tag} pterm by length (epoch={epoch})",
                key_prefix="pterm_by_len",
            )
            if fig is not None:
                payload[f"{tag}/pterm_by_len"] = wandb.Image(fig, file_type="jpg")
                plt.close(fig)

        payload = {k: v for k, v in payload.items() if v is not None}
        if len(payload) <= 1:
            return
        for run in runs:
            try:
                run.log(payload)
            except Exception:
                pass

    def _get_valid_flags(
        self,
        validator_dict: dict[str, Any] | None,
        generated_text: torch.Tensor,
        prompt_len: int,
    ) -> list[bool] | None:
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

    def _log_metrics(self, metrics: dict[str, Any], **common_kwargs) -> None:
        for name, value in metrics.items():
            if isinstance(value, tuple):
                metric_value, overrides = value
                kwargs = {**common_kwargs, **overrides}
            else:
                metric_value = value
                kwargs = common_kwargs
            metric_value = self._prepare_metric(metric_value, bool(kwargs.get("sync_dist", False)))
            self.log(name, metric_value, **kwargs)

    def _log_agreement_metrics(self, agree_list) -> None:
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

        metrics = {
            "train/agree_mean": torch.stack(agree_means).mean(),
            "train/agree_start": agree_means[0],
            "train/agree_midd": agree_means[mid_index],
            "train/agree_end": agree_means[-1],
        }
        self._log_metrics(metrics, on_step=True, sync_dist=True)

    def _sample_pf_temperature(self) -> float:
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

    def generate_from_replay_buffer(self, item, encoded_prompt):
        """Optionally generate from replay buffer samples (Buffer 1).

        Uses a dynamic probability schedule to decide whether to sample
        from the replay buffer.

        Args:
            item: Data item dictionary.
            encoded_prompt: Encoded prompt tensor.

        Returns:
            Tuple of (action_seq, result_dict) if buffer sampling is used, None otherwise.
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
        # Ensure buffer samples are on the same device as prompt
        buffer_sentences = buffer_sentences.to(device, non_blocking=True)
        prompt_expanded = prompt_tensor.expand(buffer_sentences.size(0), -1)
        prompt_prefix = prompt_expanded[:, :-1]
        action_seq = torch.cat([prompt_prefix, buffer_sentences], dim=1)
        result_dict = self.forward(
            item,
            action_seq=action_seq,
            pf_temperature=1.0,  # no temperature sampling for buffer sampling
            reward_temperature=self.reward.temperature,
            scaling_factor=self.get_scaling_factor_at_step(self.global_step),
            reference_logits_scale=self.get_reference_logits_scale_at_step(self.global_step),
        )
        return action_seq, result_dict

    def _decode_generated_tokens(
        self, tokens: torch.Tensor, skip_special_tokens: bool = True
    ) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _ema_cfg(self, key: str, default: Any) -> Any:
        cfg = self.ema_config
        if cfg is None:
            return default
        if hasattr(cfg, "get"):
            value = cfg.get(key, default)
        else:
            value = getattr(cfg, key, default)
        return default if value is None else value

    def _maybe_update_ema(self) -> None:
        if not self._ema_enabled:
            return
        epoch = int(getattr(self.trainer, "current_epoch", 0))
        completed_epochs = epoch + 1
        ema_updated = False
        reference_updated = False
        reference_delta = None
        if completed_epochs >= self._ema_start_epoch:
            self._update_ema_state()
            ema_updated = True
            self._maybe_store_ema_snapshot(completed_epochs)
            if self._should_update_reference(completed_epochs):
                ema_state_to_apply = self._get_delayed_ema_state(completed_epochs)
                if ema_state_to_apply is not None:
                    reference_delta = self._apply_ema_to_reference(ema_state_to_apply)
                    self._ema_reference_update_count += 1
                    reference_updated = True
        self._log_ema_metrics(ema_updated, reference_updated, reference_delta)

    def _maybe_store_ema_snapshot(self, completed_epochs: int) -> None:
        if self._ema_state_history is None:
            return
        self._ema_state_history.append((completed_epochs, self._capture_ema_state()))

    def _get_delayed_ema_state(self, completed_epochs: int) -> dict[str, torch.Tensor] | None:
        if self._ema_reference_delay_epochs <= 0:
            return self._ema_state
        if self._ema_state_history is None:
            return None
        if len(self._ema_state_history) < self._ema_reference_delay_epochs + 1:
            return None
        target_epoch = completed_epochs - self._ema_reference_delay_epochs
        for epoch, state in self._ema_state_history:
            if epoch == target_epoch:
                return state
        return None

    def _update_ema_state(self) -> None:
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                if not param.requires_grad:
                    continue
                param_cpu = param.detach().cpu()
                if name not in self._ema_state:
                    self._ema_state[name] = param_cpu.clone()
                else:
                    self._ema_state[name].mul_(self._ema_decay).add_(
                        param_cpu, alpha=1 - self._ema_decay
                    )

    def _should_update_reference(self, completed_epochs: int) -> bool:
        if completed_epochs < self._ema_reference_start_epoch:
            return False
        if self._ema_reference_interval <= 0:
            return False
        return (
            (completed_epochs - self._ema_reference_start_epoch) % self._ema_reference_interval
        ) == 0

    def _apply_ema_to_reference(
        self, ema_state: dict[str, torch.Tensor]
    ) -> tuple[float, float] | None:
        if not ema_state:
            return None
        target_model = self._net_frozen_raw
        reference_delta = None
        if self._ema_log_reference_delta:
            if self.disable_peft:
                reference_delta = self._compute_reference_delta_vs_model(target_model, ema_state)
            else:
                reference_delta = self._compute_reference_delta_vs_prev(ema_state)
        if self.disable_peft:
            target_model.load_state_dict(ema_state, strict=False)
        else:
            if hasattr(target_model, "merge_and_unload"):
                ema_peft = target_model
            else:
                if hasattr(target_model, "peft_config"):
                    try:
                        delattr(target_model, "peft_config")
                    except Exception:
                        pass
                ema_peft = get_peft_model(target_model, self.lora_config)
            ema_peft.load_state_dict(ema_state, strict=False)
            if hasattr(ema_peft, "merge_and_unload"):
                merged_model = ema_peft.merge_and_unload()
                if merged_model is not target_model:
                    target_model.load_state_dict(merged_model.state_dict())
            else:
                if hasattr(ema_peft, "base_model"):
                    ema_peft.base_model.disable_adapter_layers()
        target_model.eval()
        target_model.requires_grad_(False)
        if self._ema_log_reference_delta and not self.disable_peft:
            self._ema_reference_prev_state = self._capture_ema_state(ema_state)
        return reference_delta

    def _log_ema_metrics(
        self,
        ema_updated: bool,
        reference_updated: bool,
        reference_delta: tuple[float, float] | None,
    ) -> None:
        metrics = {
            "ema/updated": float(ema_updated),
            "ema/reference_updated": float(reference_updated),
            "ema/reference_update_count": float(self._ema_reference_update_count),
            "ema/decay": float(self._ema_decay),
        }
        if self._ema_log_param_delta:
            stats = self._compute_ema_param_delta()
            if stats is not None:
                mean, var = stats
                metrics["ema/param_delta_mean_abs"] = mean
                metrics["ema/param_delta_var_abs"] = var
        if reference_delta is not None:
            mean, var = reference_delta
            metrics["ema/reference_param_delta_mean_abs"] = mean
            metrics["ema/reference_param_delta_var_abs"] = var
        self._log_metrics(metrics, on_epoch=True, sync_dist=True)

    def _compute_ema_param_delta(self) -> tuple[float, float] | None:
        if not self._ema_state:
            return None
        total = 0.0
        total_sq = 0.0
        count = 0
        param_map = dict(self.net.named_parameters())
        for name, ema_tensor in self._ema_state.items():
            if not self._should_track_ema_name(name):
                continue
            param = param_map.get(name)
            if param is None:
                continue
            param_cpu = param.detach().cpu()
            delta = (param_cpu - ema_tensor).float().abs()
            total += delta.sum().item()
            total_sq += (delta * delta).sum().item()
            count += delta.numel()
        if count == 0:
            return None
        mean = total / count
        var = total_sq / count - mean * mean
        if var < 0:
            var = 0.0
        return mean, var

    def _compute_reference_delta_vs_prev(
        self, ema_state: dict[str, torch.Tensor]
    ) -> tuple[float, float] | None:
        if not self._ema_reference_prev_state:
            return None
        total = 0.0
        total_sq = 0.0
        count = 0
        for name, ema_tensor in ema_state.items():
            if not self._should_track_ema_name(name):
                continue
            prev_tensor = self._ema_reference_prev_state.get(name)
            if prev_tensor is None:
                continue
            delta = (ema_tensor - prev_tensor).float().abs()
            total += delta.sum().item()
            total_sq += (delta * delta).sum().item()
            count += delta.numel()
        if count == 0:
            return None
        mean = total / count
        var = total_sq / count - mean * mean
        if var < 0:
            var = 0.0
        return mean, var

    def _compute_reference_delta_vs_model(
        self, model, ema_state: dict[str, torch.Tensor]
    ) -> tuple[float, float] | None:
        total = 0.0
        total_sq = 0.0
        count = 0
        param_map = dict(model.named_parameters())
        for name, ema_tensor in ema_state.items():
            if not self._should_track_ema_name(name):
                continue
            param = param_map.get(name)
            if param is None:
                continue
            param_cpu = param.detach().cpu()
            delta = (ema_tensor - param_cpu).float().abs()
            total += delta.sum().item()
            total_sq += (delta * delta).sum().item()
            count += delta.numel()
        if count == 0:
            return None
        mean = total / count
        var = total_sq / count - mean * mean
        if var < 0:
            var = 0.0
        return mean, var

    def _capture_ema_state(
        self, ema_state: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        ema_state = self._ema_state if ema_state is None else ema_state
        snapshot: dict[str, torch.Tensor] = {}
        for name, ema_tensor in ema_state.items():
            if self._should_track_ema_name(name):
                snapshot[name] = ema_tensor.clone()
        return snapshot

    def _should_track_ema_name(self, name: str) -> bool:
        if self.disable_peft:
            return True
        return "lora_" in name
