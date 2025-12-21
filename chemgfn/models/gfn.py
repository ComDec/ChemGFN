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
    generate_and_return_termination_logprob_for_sidechain_opt,
    get_termination_vals,
    lora_to_base,
    prepare_token_mask,
)
from chemgfn.utils.phi_utils import PrefixValueMemory, compute_active_before
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
            self.illegal_vocab_penalty = -99

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile = compile

        self.train_sentence_length: list = []
        self.train_samples: list = []
        self.val_samples: list = []

        self.opt_task = self.training_mixed_config.get("opt_task", False)
        self.skip_baseline_sampling = self.training_mixed_config.skip_baseline_sampling

        # Initialize loss function from config
        self.loss_fn: GFNLoss = loss_fn
        self.return_policy_logits = bool(getattr(self.loss_fn, "requires_policy_logits", False))

        # Optional: loss weight schedulers（直接复用 factor_schedulers，由具体 loss 自行取用）
        self.loss_weight_schedulers = dict(self.factor_schedulers)
        if hasattr(self.loss_fn, "set_weight_schedulers"):
            self.loss_fn.set_weight_schedulers(self.loss_weight_schedulers)

        self.return_policy_logits = True

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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

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
            "return_policy_logits": self.return_policy_logits,
            "disable_grammar": getattr(self.constraint_config, "disable_grammar", False),
            "grammar_disagree_penalty": getattr(
                self.training_mixed_config, "grammar_disagree_penalty", -80
            ),
        }

        generator = (
            generate_and_return_termination_logprob_for_sidechain_opt
            if self.opt_task
            else generate_and_return_termination_logprob
        )
        return generator(**generation_config)

    # ------------------------------------------------------------------ #
    # Training / validation loops
    # ------------------------------------------------------------------ #

    def training_step(self, item, batch_idx) -> torch.Tensor:
        encoded_prompt = item["encoded_prompt"]
        prompt_len = encoded_prompt.shape[-1]
        buffer_sample = item["buffer_encoded_sample"]
        use_dataset_buffer = False

        # Hot patch for prefix value memory
        if self.reward.pv_mem is not None:
            self.reward.set_step(self.global_step)
            nums_pv_keys = len(self.reward.pv_mem.stats.keys())
            self.log("train/nums_pv_keys", nums_pv_keys, sync_dist=True, on_step=True)

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
        phi_state = getattr(result_dict, "phi_state", torch.zeros_like(log_pf))
        phi_tok = getattr(result_dict, "phi_tok", torch.zeros_like(log_pf))
        pv = getattr(result_dict, "pv", torch.zeros_like(log_pf))

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
        self.log_dict(
            {f"scheduled/{key}": val for key, val in scheduled_values.items()},
            sync_dist=True,
            on_step=True,
        )
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
                self.log_dict(component_losses, on_step=True, sync_dist=True, prog_bar=True)
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
        validator_metric_dict = self.reward.sentence_validator.accuracy(
            generated_text[:, prompt_len:],
            self.tokenizer,
            item.get("molecule", None),
        )
        self.train_sentence_length.append(sentence_len.detach().cpu())

        log_ps = last_log_r * self.reward.temperature
        log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature

        if batch_idx % 5 == 0:
            decoded = self._decode_generated_tokens(generated_text[0, prompt_len:])
            acc = self.reward.sentence_validator.accuracy(
                generated_text[0, prompt_len:].unsqueeze(0),
                self.tokenizer,
                item.get("molecule", None),
            )
            acc_val = acc["acc"] if isinstance(acc, dict) and "acc" in acc else float(acc)

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
                    "phi_state_list": [float(x) for x in phi_state[0].detach().cpu().tolist()],
                    "phi_tok_list": [float(x) for x in phi_tok[0].detach().cpu().tolist()],
                    "pv_list": [float(x) for x in pv[0].detach().cpu().tolist()],
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

        prefix_diag = result_dict.get("prefix_diag", None)
        if prefix_diag is not None:
            for key, value in prefix_diag.items():
                self._log_metrics(
                    {
                        f"prefix_diag/{key}": value,
                    },
                    on_step=True,
                    sync_dist=True,
                )

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

        log_r_reference = result_dict["log_r_reference"]
        log_r_target = result_dict["log_r_target"]
        log_r_unpenalized = result_dict["log_r_unpenalized"]
        self.val_samples.extend(generated_text[:, prompt_len:].tolist())

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
                self.log_dict(component_losses, on_step=False, sync_dist=True)
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

        validator_metric_dict = self.reward.sentence_validator.accuracy(
            generated_text[:, prompt_len:],
            self.tokenizer,
            batch.get("molecule", None),
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

    def on_train_batch_start(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        reward_temp = self.get_reward_temp_at_step(self.global_step)
        scaling_factor = self.get_scaling_factor_at_step(self.global_step)
        lr = self.lr_schedulers().get_lr()[0]
        self.reward.temperature = reward_temp
        self.reward.scaling_factor = scaling_factor
        self.log("train/reward_temp", reward_temp, sync_dist=True, on_step=True)

        for pg in self.optimizers().param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------ #
    # Epoch hooks
    # ------------------------------------------------------------------ #

    def on_train_epoch_end(self):
        if hasattr(self, "global_rank") and self.global_rank != 0:
            return

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

        plt.hist(self.train_sentence_length, bins=50)
        _root_dir = os.path.join(self.trainer.default_root_dir, "train_sentence_length")
        os.makedirs(_root_dir, exist_ok=True)
        plt.savefig(
            os.path.join(_root_dir, f"train_sentence_length_{self.trainer.current_epoch}.png")
        )
        plt.close()

        self.reward_buffer.save_csv(
            os.path.join(
                self.trainer.default_root_dir,
                "replay_buffer",
                f"replay_{self.trainer.current_epoch}.csv",
            ),
            self.tokenizer,
        )

        if self.logger is not None:
            self.logger.log_table("train/samples_latest", dataframe=df)
        self.train_samples.clear()
        self.train_sentence_length.clear()

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
            )

        self.log_dict({f"pv_report/{k}": v for k, v in rep.items()}, on_epoch=True, sync_dist=True)

    def on_train_epoch_start(self):
        self.log(
            "scheduled/R_temperature",
            self.get_reward_temp_at_step(self.global_step),
            sync_dist=True,
        )
        self.log("scheduled/lr", self.lr_schedulers().get_lr()[0], sync_dist=True)

    def on_validation_epoch_start(self):
        """Compute validation metrics at the start of validation epoch."""
        log_rs, log_pfss = [], []
        val_dataset = self.trainer.datamodule.val_dataloader().dataset
        self.val_probes = torch.utils.data.Subset(
            val_dataset, random.sample(range(len(val_dataset)), 10)
        )
        device = self.device
        for idx, item in enumerate(self.trainer.datamodule.val_dataloader()):
            encoded_prompt = item["encoded_prompt"]
            # Move all tensors to device in one pass
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    item[key] = value.to(device, non_blocking=True)
            result_dict = self.forward(item, pf_temperature=1.0)
            generated_text = result_dict["state"]
            log_pf = result_dict["log_pf"]
            log_pterm = result_dict["log_pterm"]
            log_r = result_dict["log_r"]
            log_r_unpenalized = result_dict["log_r_unpenalized"]

            log_pfs, log_r_val, _, _ = get_termination_vals(
                generated_text=generated_text,
                log_pf=log_pf,
                log_pterm=log_pterm,
                log_r=log_r,
                log_r_unpenalized=log_r_unpenalized,
                termination_token_id=self.end_of_sentence_token_id,
                prompt_len=len(encoded_prompt[0]),
            )
            log_rs.append(log_r_val)
            log_pfss.append(log_pfs)

            if idx == 10:
                break

        log_rs, log_pfss = torch.cat(log_rs), torch.cat(log_pfss)
        self.log("val/Var(logR - logPf(s))", (log_rs - log_pfss).var(), sync_dist=True)

        if self.val_probes is not None and self.logger is not None:
            samples_table = self.sample_probes(self.val_probes)
            _root_dir = os.path.join(self.trainer.default_root_dir, "validation_samples")
            os.makedirs(_root_dir, exist_ok=True)
            samples_table.to_csv(
                os.path.join(
                    _root_dir,
                    f"samples_val_probes_{self.trainer.global_step}.csv",
                ),
                index=False,
            )

        self.val_samples = []

    def on_validation_epoch_end(self):
        diversity = calculate_diversity(torch.tensor(self.val_samples))
        self.log("val/diversity", diversity, sync_dist=True, on_epoch=True)
        self.val_samples = []

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

    def sample_probes(self, probes, n_samples=4):
        """Sample and decode probe sequences for logging.

        Args:
            probes: List of probe data items.
            n_samples: Number of samples per probe (default: 4).

        Returns:
            DataFrame with sampled sequences and their metrics.
        """
        samples = []
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

            log_ps, log_ps_unpenalized = get_termination_vals(
                generated_text=generated_text,
                log_pf=None,
                log_pterm=None,
                log_r=log_r,
                log_r_unpenalized=log_r_unpenalized,
                termination_token_id=self.end_of_sentence_token_id,
                prompt_len=prompt_len,
            )[1:3]

            log_ps *= self.get_reward_temp_at_step(self.global_step)
            log_ps_unpenalized *= self.get_reward_temp_at_step(self.global_step)

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
        return pd.DataFrame(samples)

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

    def _log_metrics(self, metrics: dict[str, Any], **common_kwargs) -> None:
        for name, value in metrics.items():
            if isinstance(value, tuple):
                metric_value, overrides = value
                kwargs = {**common_kwargs, **overrides}
            else:
                metric_value = value
                kwargs = common_kwargs
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
