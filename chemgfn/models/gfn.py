from __future__ import annotations

import os
import random
import sys
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

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
    GrammarIncrementalLogitsProcessorForNumberOnly,
    GrammarIncrementalLogitsProcessorGeneral,
    GrammarIncrementalLogitsProcessorSampleEnhanced,
    GrammarLogitsProcessorPartheseness,
)
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer

from chemgfn.models.losses import GFNLoss
from chemgfn.utils.gfn_utils import (
    ReplayBuffer,
    base_to_lora,
    calculate_diversity,
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
from chemgfn.utils.schedulers import Scheduler

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
        constraint_config: dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        factor_schedulers: dict[str, Any],
        compile: bool,
        disable_peft: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["net", "loss_fn"])
        model = AutoModelForCausalLM.from_pretrained(net_config.pretrained_model_name_or_path)
        model.train()

        model_frozen = AutoModelForCausalLM.from_pretrained(
            net_config.pretrained_model_name_or_path
        )
        model_frozen.eval()
        model_frozen.requires_grad_(False)

        if not disable_peft:
            self.net = get_peft_model(model, lora_config)
        else:
            self.net = model

        self.net_frozen = model_frozen
        self.tokenizer = tokenizer

        self.reward_config = reward_config
        self.constraint_config = constraint_config
        self.training_mixed_config = training_mixed_config
        self.end_of_sentence_token_id = self.tokenizer.eos_token_id

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

        self.buffer_mixture_ratio = training_mixed_config["buffer_mixture_ratio"]

        self.reward = reward
        self.reward_buffer: ReplayBuffer = reward_buffer
        self.reward_buffer.set_termination_token_id(self.end_of_sentence_token_id)
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
        self.compile = compile

        self.train_sentence_length: list = []
        self.train_samples: list = []
        self.val_samples: list = []
        self.test_samples: list = []
        self.train_samples_ids: list = []
        self.val_samples_ids: list = []
        self.test_samples_ids: list = []
        self.val_samples_table: list = []
        self.val_log_rs: list = []
        self.val_log_pfss: list = []
        self.test_samples_table: list = []
        self.test_log_rs: list = []
        self.test_log_pfss: list = []

        self.skip_baseline_sampling = self.training_mixed_config.skip_baseline_sampling

        # Initialize loss function from config
        self.loss_fn: GFNLoss = loss_fn

        # Optional: loss weight schedulers
        self.loss_weight_schedulers = dict(self.factor_schedulers)
        if hasattr(self.loss_fn, "set_weight_schedulers"):
            self.loss_fn.set_weight_schedulers(self.loss_weight_schedulers)

        if hasattr(self.loss_fn, "set_alpha_reference"):
            self.loss_fn.set_alpha_reference(self.get_alpha_reference_at_step(self.global_step))

        try:
            if self.compile:
                self.net = torch.compile(self.net, mode="max-autotune", fullgraph=False)
                self.net_frozen = torch.compile(
                    self.net_frozen, mode="max-autotune", fullgraph=False
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"torch.compile failed, continuing without compilation: {exc}")

        # phi cache
        self._pv_probe_cache = None
        self._pv_report_epoch = -1

        # debug flags
        debug_shapes = os.environ.get("CHEMGFN_DEBUG_SHAPES", "0") == "1"
        debug_steps = int(os.environ.get("CHEMGFN_DEBUG_SHAPES_STEPS", "1"))
        self.debug_shapes = debug_shapes
        self._debug_shapes_remaining = debug_steps if debug_shapes else 0

    def set_probes(self, train_probes, val_probes):
        self.train_probes = train_probes
        self.val_probes = val_probes

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

        if self._debug_shapes_remaining > 0:
            self._debug_shapes_remaining -= 1
            state = result["state"]
            log_pf = result["log_pf"]
            log_pterm = result["log_pterm"]
            log_r = result.get("log_r")
            log_r_unpenalized = result.get("log_r_unpenalized")
            log_pf_ref = result.get("log_pf_ref")
            log_pterm_ref = result.get("log_pterm_ref")

            B = state.shape[0]
            T_tok = log_pf.shape[1] if log_pf is not None else None
            L_state = (
                log_r.shape[1]
                if log_r is not None
                else (log_pterm.shape[1] if log_pterm is not None else None)
            )
            gen_tokens = state[:, prompt_len : prompt_len + T_tok] if T_tok is not None else None

            assert log_pf is not None and log_pf.ndim == 2
            assert log_pterm is not None and log_pterm.ndim == 2
            assert log_pf.shape[0] == B and log_pterm.shape[0] == B
            if log_r is not None:
                assert log_r.shape[0] == B

            print(
                "[forward debug] step="
                f"{int(self.global_step)} B={B} T_tok={T_tok} L_state={L_state} "
                f"state={tuple(state.shape)} "
                f"gen_tokens={None if gen_tokens is None else tuple(gen_tokens.shape)} "
                f"log_pf={tuple(log_pf.shape)} log_pterm={tuple(log_pterm.shape)} "
                f"log_r={None if log_r is None else tuple(log_r.shape)} "
                f"log_r_unpenalized="
                f"{None if log_r_unpenalized is None else tuple(log_r_unpenalized.shape)} "
                f"log_pf_ref={None if log_pf_ref is None else tuple(log_pf_ref.shape)} "
                f"log_pterm_ref={None if log_pterm_ref is None else tuple(log_pterm_ref.shape)}"
            )

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

        log_r_reference = result_dict["log_r_reference"]
        log_r_target = result_dict["log_r_target"]

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

        weight_overrides = None
        if getattr(self, "loss_weight_schedulers", None):
            # Evaluate all loss-weight schedulers at the current step so losses can
            # resolve their weights without reaching back into the module.
            weight_overrides = {
                name: sched(self.global_step)
                for name, sched in self.loss_weight_schedulers.items()
            }

        loss_output = self.loss_fn(
            log_pf=log_pf,
            log_r=log_r,  # compatibility with old loss functions
            log_r_reference=log_r_reference,
            log_r_target=log_r_target,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=prompt_len,
            global_step=self.global_step,
            weight_overrides=weight_overrides,
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

        validator_dict = result_dict.get("validator_dict")
        valid_flags = self._get_valid_flags(validator_dict, generated_text, prompt_len)
        if self.reward.sentence_validator is None:
            validator_metric_dict = {}
        else:
            tokens = generated_text[:, prompt_len:]
            try:
                validator_metric_dict = self.reward.sentence_validator.accuracy(
                    tokens,
                    self.tokenizer,
                    item.get("molecule", None),
                )
            except TypeError:
                validator_metric_dict = self.reward.sentence_validator.accuracy(
                    tokens,
                    self.tokenizer,
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
                    f"train/replay_buffer_{key}": value
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
                f"train/validator_{key}": (value, {"prog_bar": True})
                for key, value in validator_metric_dict.items()
            },
            on_step=True,
            sync_dist=True,
        )

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
        result_dict = self.forward(batch, reward_temperature=1.0, pf_temperature=1.0)
        generated_text = result_dict["state"]
        log_pf = result_dict["log_pf"]
        log_pterm = result_dict["log_pterm"]
        log_r = result_dict["log_r"]
        log_pf_ref = result_dict["log_pf_ref"]
        log_pterm_ref = result_dict["log_pterm_ref"]
        validator_dict = result_dict.get("validator_dict")

        log_r_reference = result_dict["log_r_reference"]
        log_r_target = result_dict["log_r_target"]
        log_r_unpenalized = result_dict["log_r_unpenalized"]
        self.val_samples.extend(generated_text[:, prompt_len:].tolist())
        self.val_samples_ids.extend(generated_text[:, prompt_len:].tolist())

        if hasattr(self.loss_fn, "set_global_step"):
            self.loss_fn.set_global_step(self.global_step)
        weight_overrides = None
        if getattr(self, "loss_weight_schedulers", None):
            weight_overrides = {
                name: sched(self.global_step)
                for name, sched in self.loss_weight_schedulers.items()
            }
        loss_output = self.loss_fn(
            log_pf=log_pf,
            log_r=log_r,  # compatibility with old loss functions
            log_r_reference=log_r_reference,
            log_r_target=log_r_target,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=prompt_len,
            global_step=self.global_step,
            weight_overrides=weight_overrides,
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
        if log_pfs is not None:
            self.val_log_rs.append(last_log_r.detach().cpu())
            self.val_log_pfss.append(log_pfs.detach().cpu())

        validator_dict = result_dict.get("validator_dict")
        valid_flags = self._get_valid_flags(validator_dict, generated_text, prompt_len)
        if self.reward.sentence_validator is None:
            validator_metric_dict = {}
        else:
            tokens = generated_text[:, prompt_len:]
            try:
                validator_metric_dict = self.reward.sentence_validator.accuracy(
                    tokens,
                    self.tokenizer,
                    batch.get("molecule", None),
                )
            except TypeError:
                validator_metric_dict = self.reward.sentence_validator.accuracy(
                    tokens,
                    self.tokenizer,
                )
        self._log_metrics(
            {f"val/validator_{key}": value for key, value in validator_metric_dict.items()},
            sync_dist=True,
            on_epoch=True,
        )

        log_ps = last_log_r * self.reward.temperature
        log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature
        self._log_metrics(
            {
                "val/loss": (loss, {"prog_bar": True}),
                "val/logR": last_log_r.mean(),
                "val/logP(s) (avg)": log_ps.mean(),
                "val/logP(s) (max)": log_ps.max(),
                "val/logP(s) unpenalized (avg)": log_ps_unpenalized.mean(),
                "val/logP(s) unpenalized (max)": log_ps_unpenalized.max(),
                "val/Mean(log_pterm - log_pterm_ref)": (log_pterm - log_pterm_ref).mean(),
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
        result_dict = self.forward(batch, reward_temperature=1.0, pf_temperature=1.0)
        generated_text = result_dict["state"]
        log_pf = result_dict["log_pf"]
        log_pterm = result_dict["log_pterm"]
        log_r = result_dict["log_r"]
        log_pf_ref = result_dict["log_pf_ref"]
        log_pterm_ref = result_dict["log_pterm_ref"]
        validator_dict = result_dict.get("validator_dict")

        log_r_reference = result_dict["log_r_reference"]
        log_r_target = result_dict["log_r_target"]
        log_r_unpenalized = result_dict["log_r_unpenalized"]
        self.test_samples.extend(generated_text[:, prompt_len:].tolist())
        self.test_samples_ids.extend(generated_text[:, prompt_len:].tolist())

        if hasattr(self.loss_fn, "set_global_step"):
            self.loss_fn.set_global_step(self.global_step)
        weight_overrides = None
        if getattr(self, "loss_weight_schedulers", None):
            weight_overrides = {
                name: sched(self.global_step)
                for name, sched in self.loss_weight_schedulers.items()
            }
        loss_output = self.loss_fn(
            log_pf=log_pf,
            log_r=log_r,  # compatibility with old loss functions
            log_r_reference=log_r_reference,
            log_r_target=log_r_target,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=prompt_len,
            global_step=self.global_step,
            weight_overrides=weight_overrides,
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
        if log_pfs is not None:
            self.test_log_rs.append(last_log_r.detach().cpu())
            self.test_log_pfss.append(log_pfs.detach().cpu())

        validator_dict = result_dict.get("validator_dict")
        valid_flags = self._get_valid_flags(validator_dict, generated_text, prompt_len)
        if self.reward.sentence_validator is None:
            validator_metric_dict = {}
        else:
            tokens = generated_text[:, prompt_len:]
            try:
                validator_metric_dict = self.reward.sentence_validator.accuracy(
                    tokens,
                    self.tokenizer,
                    batch.get("molecule", None),
                )
            except TypeError:
                validator_metric_dict = self.reward.sentence_validator.accuracy(
                    tokens,
                    self.tokenizer,
                )
        self._log_metrics(
            {f"test/validator_{key}": value for key, value in validator_metric_dict.items()},
            sync_dist=True,
            on_epoch=True,
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
        lr = self.lr_schedulers().get_lr()[0]
        self.reward.temperature = reward_temp
        self.reward.scaling_factor = scaling_factor
        self._log_metrics({"train/reward_temp": reward_temp}, sync_dist=True, on_step=True)

        for pg in self.optimizers().param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------ #
    # Epoch hooks
    # ------------------------------------------------------------------ #

    def on_train_epoch_end(self):
        if hasattr(self, "global_rank") and self.global_rank != 0:
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

        # log prefix collapse metrics
        eos = self.end_of_sentence_token_id
        seqs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in self.train_samples_ids],
            batch_first=True,
            padding_value=eos,
        )
        active_before = compute_active_before(seqs, eos=eos)
        non_eos = seqs != eos
        mask_noeos = active_before & non_eos
        seqs_list = seqs.detach().cpu().tolist()
        mask_list = mask_noeos.detach().cpu().tolist()
        pos = prefix_collapse_by_position(seqs_list, mask_list, collapse_thr=0.95)
        kmet = prefix_collapse_by_k(seqs_list, mask_list, k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        log = {
            "prefix_pos_train/top1_auc": float(pos.top1_auc),
            "prefix_pos_train/collapse_depth_095": float(pos.collapse_depth),
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

        # log replay buffer
        self.reward_buffer.save_csv(
            os.path.join(
                self.trainer.default_root_dir,
                "replay_buffer",
                f"replay_{self.trainer.current_epoch}.csv",
            ),
            self.tokenizer,
        )

        # if self.logger is not None:
        #     self.logger.log_table("train/samples_latest", dataframe=df)

        self.train_samples.clear()
        self.train_samples_ids.clear()
        self.train_sentence_length.clear()

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

    def on_train_epoch_start(self):
        self._log_metrics(
            {"scheduled/R_temperature": self.get_reward_temp_at_step(self.global_step)},
            sync_dist=True,
        )
        self._log_metrics(
            {"scheduled/lr": self.lr_schedulers().get_lr()[0]},
            sync_dist=True,
        )

    def on_validation_epoch_start(self):
        """Prepare validation probes and reset cached samples."""
        val_dataset = self.trainer.datamodule.val_dataloader().dataset
        self.val_probes = torch.utils.data.Subset(
            val_dataset, random.sample(range(len(val_dataset)), 10)
        )
        self.val_samples.clear()
        self.val_samples_ids.clear()
        self.val_samples_table.clear()
        self.val_log_rs.clear()
        self.val_log_pfss.clear()

    def on_validation_epoch_end(self):
        diversity = calculate_diversity(torch.tensor(self.val_samples))
        self._log_metrics({"val/diversity": diversity}, sync_dist=True, on_epoch=True)

        # log prefix collapse metrics
        eos = self.end_of_sentence_token_id
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
        pos = prefix_collapse_by_position(seqs_list, mask_list, collapse_thr=0.95)
        kmet = prefix_collapse_by_k(seqs_list, mask_list, k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        log = {
            "prefix_pos_val/top1_auc": float(pos.top1_auc),
            "prefix_pos_val/collapse_depth_095": float(pos.collapse_depth),
        }
        self._log_metrics(log, on_epoch=True, sync_dist=True)
        self._log_wandb_prefix_tables("prefix_pos_val", pos, kmet)

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
        self.test_samples.clear()
        self.test_samples_ids.clear()
        self.test_samples_table.clear()
        self.test_log_rs.clear()
        self.test_log_pfss.clear()

    def on_test_epoch_end(self):
        diversity = calculate_diversity(torch.tensor(self.test_samples))
        self._log_metrics({"test/diversity": diversity}, sync_dist=True, on_epoch=True)

        eos = self.end_of_sentence_token_id
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
        pos = prefix_collapse_by_position(seqs_list, mask_list, collapse_thr=0.95)
        kmet = prefix_collapse_by_k(seqs_list, mask_list, k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        log = {
            "prefix_pos_test/top1_auc": float(pos.top1_auc),
            "prefix_pos_test/collapse_depth_095": float(pos.collapse_depth),
        }
        self._log_metrics(log, on_epoch=True, sync_dist=True)
        self._log_wandb_prefix_tables("prefix_pos_test", pos, kmet)

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
        _root_dir = os.path.join(self.trainer.default_root_dir, "test_samples")
        os.makedirs(_root_dir, exist_ok=True)
        samples_table.to_csv(
            os.path.join(
                _root_dir,
                f"samples_test_{self.trainer.global_step}.csv",
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

    def on_train_start(self):
        val_dataset = self.trainer.datamodule.val_dataloader().dataset
        val_probes = torch.utils.data.Subset(
            val_dataset, random.sample(range(len(val_dataset)), 10)
        )
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

    def sample_probes_baselines(self, probes, n_samples=4):
        assert isinstance(probes, list) and probes[0].ndim == 1
        samples = []
        for probe in probes:
            probe_str = self.tokenizer.decode(probe)
            probe_samples = self.sample_baselines(probe.to(self.device), n_samples=n_samples)
            for idx in range(n_samples):
                sample = {"Prompt": probe_str}
                for baseline in probe_samples:
                    sample[f"Sampled sentence ({baseline})"] = probe_samples[baseline]["sample"][
                        idx
                    ]
                    sample[f"logP(s) ({baseline})"] = probe_samples[baseline]["logP(s)"][
                        idx
                    ].item()
                    sample[f"logP(s) unpenalized ({baseline})"] = probe_samples[baseline][
                        "logP(s) unpenalized"
                    ][idx].item()
                samples.append(sample)

        df = pd.DataFrame(samples)
        return df.sort_values(by=["Prompt"], ascending=False)

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
                generated_text = [text.replace(".", "") for text in generated_text]
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
        if self.compile and stage == "fit":
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

    @staticmethod
    def _normalize_scalar(value: Any) -> float:
        if value is None:
            return 0.0
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _load_token_masks(self):
        tokens_path = getattr(self.constraint_config, "legal_tokens", None)
        if tokens_path and os.path.exists(tokens_path):
            return prepare_token_mask(self.tokenizer, tokens_path)

        if tokens_path:
            print(f"Legal tokens file not found: {tokens_path}")
        return None, None, None

    def _build_pre_grammar_processor(self, parsed_grammar):
        processor_type = getattr(self.constraint_config, "processor_type", "none")
        if processor_type == "none":
            return None
        if processor_type == "general":
            if self.grammar is None:
                print("Grammar parsing failed with current tokenizer, disable general processor")
                return None
            return GrammarConstrainedLogitsProcessor(self.grammar)

        processor_map = {
            "prefix": GrammarIncrementalLogitsProcessorGeneral,
            "prefix_enhanced": GrammarIncrementalLogitsProcessorSampleEnhanced,
            "parenthese": GrammarLogitsProcessorPartheseness,
            "number_only": GrammarIncrementalLogitsProcessorForNumberOnly,
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
        # we have a elegant way to disable grammar, but it is not used in the code

        # if not getattr(self.constraint_config, "apply_grammar", False):
        #     self.string_grammar = None
        #     self.grammar = None
        #     self.pre_grammar_processor = None
        #     self.grammar_processor = None
        #     return

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

    def _strip_special_token_ids(self, token_ids: list[int]) -> list[int]:
        special_ids = set(self.tokenizer.all_special_ids)
        return [tok for tok in token_ids if tok not in special_ids]

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

        # one-time: encourage scalar plots to use epoch as x-axis (doesn't touch wandb step monotonicity)
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
        _root_dir = os.path.join(self.trainer.default_root_dir, "prefix_tables")
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

        # -------------------------
        # POS: local table + 3-subplot figure
        # -------------------------
        pos_rows = []
        if getattr(pos, "top1_mass", None):
            T = min(
                len(getattr(pos, "top1_mass", [])),
                len(getattr(pos, "entropy", [])),
                len(getattr(pos, "eff_support", [])),
            )
            if T > 0:
                cols = ["epoch", "t", "top1_mass", "entropy", "eff_support"]
                if getattr(pos, "unique", None) is not None:
                    cols.append("unique")
                if getattr(pos, "support", None) is not None:
                    cols.append("support")

                for t in range(T):
                    row = [
                        epoch,
                        int(t),
                        float(pos.top1_mass[t]),
                        float(pos.entropy[t]),
                        float(pos.eff_support[t]),
                    ]
                    if getattr(pos, "unique", None) is not None:
                        row.append(int(pos.unique[t]))
                    if getattr(pos, "support", None) is not None:
                        row.append(int(pos.support[t]))
                    pos_rows.append(row)

                pd.DataFrame(pos_rows, columns=cols).to_csv(
                    os.path.join(_root_dir, f"{local_tag}_pos_{epoch}.csv"),
                    index=False,
                )

                # 3 subplots in ONE figure
                try:
                    import matplotlib.pyplot as plt

                    xs = list(range(T))
                    top1_vals = [float(v) for v in pos.top1_mass[:T]]
                    ent_vals = [float(v) for v in pos.entropy[:T]]
                    eff_vals = [float(v) for v in pos.eff_support[:T]]
                    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

                    axes[0].plot(xs, top1_vals)
                    axes[0].set_ylabel("top1_mass")

                    axes[1].plot(xs, ent_vals)
                    axes[1].set_ylabel("entropy")

                    axes[2].plot(xs, eff_vals)
                    axes[2].set_ylabel("eff_support")
                    axes[2].set_xlabel("t")

                    _apply_ylim(axes[0], _update_range("pos_top1_mass", top1_vals))
                    _apply_ylim(axes[1], _update_range("pos_entropy", ent_vals))
                    _apply_ylim(axes[2], _update_range("pos_eff_support", eff_vals))
                    x_rng = _update_x_range("pos_x_min", "pos_x_max", xs)
                    if x_rng is not None:
                        for ax in axes:
                            ax.set_xlim(x_rng[0], x_rng[1])

                    fig.suptitle(f"{tag} / prefix-by-position (epoch={epoch})")
                    fig.tight_layout()

                    payload[f"{tag}/pos_curves_img"] = wandb.Image(fig)
                    plt.close(fig)
                except Exception:
                    pass

        # -------------------------
        # K: local table + 4-subplot figure
        # -------------------------
        k_rows = []
        if kmet:
            ks = sorted(kmet.keys())
            if ks:
                cols = ["epoch", "k", "top1", "top5", "entropy", "eff", "n", "unique"]
                for k in ks:
                    v = kmet[k] or {}
                    k_rows.append(
                        [
                            epoch,
                            int(k),
                            float(v.get("top1", 0.0)),
                            float(v.get("top5", 0.0)),
                            float(v.get("entropy", 0.0)),
                            float(v.get("eff", 0.0)),
                            float(v.get("n", 0.0)),
                            float(v.get("unique", 0.0)),
                        ]
                    )
                pd.DataFrame(k_rows, columns=cols).to_csv(
                    os.path.join(_root_dir, f"{local_tag}_k_{epoch}.csv"),
                    index=False,
                )

                # 4 subplots in ONE figure
                try:
                    import matplotlib.pyplot as plt

                    xs = [int(k) for k in ks]
                    top1s = [float(kmet[k].get("top1", 0.0)) for k in ks]
                    top5s = [float(kmet[k].get("top5", 0.0)) for k in ks]
                    ents = [float(kmet[k].get("entropy", 0.0)) for k in ks]
                    effs = [float(kmet[k].get("eff", 0.0)) for k in ks]

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

                    _apply_ylim(axes[0], _update_range("k_top1", top1s))
                    _apply_ylim(axes[1], _update_range("k_top5", top5s))
                    _apply_ylim(axes[2], _update_range("k_entropy", ents))
                    _apply_ylim(axes[3], _update_range("k_eff", effs))
                    x_rng = _update_x_range("k_x_min", "k_x_max", xs)
                    if x_rng is not None:
                        for ax in axes:
                            ax.set_xlim(x_rng[0], x_rng[1])

                    fig.suptitle(f"{tag} / prefix-by-k (epoch={epoch})")
                    fig.tight_layout()

                    payload[f"{tag}/k_curves_img"] = wandb.Image(fig)
                    plt.close(fig)
                except Exception:
                    pass

        # optional scalars
        if getattr(pos, "top1_auc", None) is not None:
            payload[f"{tag}/top1_auc"] = float(pos.top1_auc)
        if getattr(pos, "collapse_depth", None) is not None:
            payload[f"{tag}/collapse_depth_095"] = float(pos.collapse_depth)

        # drop Nones
        payload = {k: v for k, v in payload.items() if v is not None}
        if len(payload) <= 1:
            return

        # IMPORTANT: do NOT set step manually (avoid non-monotonic step warnings)
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
        if self.constraint_config.processor_type == "number_only":
            pieces = []
            for token in tokens:
                if token.item() == self.tokenizer.eos_token_id:
                    break
                pieces.append(self.tokenizer.decode(token, skip_special_tokens=False))
            return ",".join(pieces)

        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
