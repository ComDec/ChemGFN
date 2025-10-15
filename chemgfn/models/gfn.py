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
import torch.utils  # type: ignore
import torch.utils.data  # type: ignore
import wandb  # type: ignore
from lightning import LightningModule
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from transformers_cfg.generation.logits_process import (
    GrammarConstrainedLogitsProcessor,
    GrammarIncrementalLogitsProcessorForNumberOnly,
    GrammarIncrementalLogitsProcessorGeneral,
    GrammarIncrementalLogitsProcessorSampleEnhanced,
    GrammarLimitedOneTimeLogitsProcessor,
    GrammarLogitsProcessorPartheseness,
)
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer

from chemgfn.utils.gfn_utils import (
    ReplayBuffer,
    base_to_lora,
    calculate_diversity,
    generate_and_return_termination_logprob,
    generate_and_return_termination_logprob_for_sidechain_opt,
    get_termination_vals,
    lora_to_base,
    modified_subtb_loss,
    prepare_token_mask,
)

sys.setrecursionlimit(1500)


class ChemGFNModule(LightningModule):
    """Main Lightning module wrapping model, reward and optimisation logic."""

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

    @classmethod
    def _build_linear_schedule(
        cls, start: float, end: float, horizon: Any
    ) -> Callable[[Any], float]:
        horizon_value = cls._normalize_scalar(horizon)
        if horizon_value <= 0:
            return lambda *_: end

        delta = end - start

        def schedule(step: Any) -> float:
            step_value = cls._normalize_scalar(step)
            progress = min(1.0, step_value / horizon_value)
            return start + delta * progress

        return schedule

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
        if not getattr(self.constraint_config, "apply_grammar", False):
            self.string_grammar = None
            self.grammar = None
            self.pre_grammar_processor = None
            self.grammar_processor = None
            return

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

        agree_sum = [torch.mean(x.sum(-1).float()).item() for x in tensors]
        mid_index = len(agree_sum) // 2
        metrics = {
            "train/agree_mean": torch.tensor(agree_sum, device=self.device).mean(),
            "train/agree_start": torch.tensor(agree_sum[0], device=self.device),
            "train/agree_midd": torch.tensor(agree_sum[mid_index], device=self.device),
            "train/agree_end": torch.tensor(agree_sum[-1], device=self.device),
        }
        self._log_metrics(metrics, on_step=True, sync_dist=True)

    def _sample_pf_temperature(self) -> float:
        if random.random() >= self.training_mixed_config.pf_temp_prob:
            return 1.0
        pf_low = self.training_mixed_config.pf_temp_low
        pf_high = self.training_mixed_config.pf_temp_high
        return random.random() * (pf_high - pf_low) + pf_low

    def _maybe_generate_from_buffer(self, item, encoded_prompt):
        if random.random() >= self.get_use_buffer_sample_at_step(self.global_step):
            return None

        prompt_tensor = encoded_prompt if encoded_prompt.ndim == 2 else encoded_prompt.unsqueeze(0)

        buffer_sentences, _ = self.reward_buffer.sample(
            self.training_mixed_config.n_samples,
            prompt_tensor,
            self.tokenizer,
        )
        if buffer_sentences is None:
            return None
        buffer_sentences = buffer_sentences.to(encoded_prompt.device)
        prompt_expanded = prompt_tensor.expand(buffer_sentences.size(0), -1).to(
            encoded_prompt.device
        )
        prompt_prefix = prompt_expanded[:, :-1]
        action_seq = torch.cat([prompt_prefix, buffer_sentences], dim=1)
        result_dict = self.forward(item, action_seq=action_seq)
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

    # ------------------------------------------------------------------ #
    # Core initialisation
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        net_config: dict[str, Any],
        lora_config: LoraConfig,
        tokenizer: PreTrainedTokenizer,
        reward,
        reward_buffer,
        reward_config: dict[str, Any],
        training_mixed_config: dict[str, Any],
        constraint_config: dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        disable_peft: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["net"])
        model = AutoModelForCausalLM.from_pretrained(net_config.pretrained_model_name_or_path)
        if not disable_peft:
            self.net = get_peft_model(model, lora_config)
        else:
            self.net = model

        self.tokenizer = tokenizer

        self.reward_config = reward_config
        self.constraint_config = constraint_config
        self.training_mixed_config = training_mixed_config
        self.end_of_sentence_token_id = self.tokenizer.eos_token_id

        self.get_reward_temp_at_step = self._build_linear_schedule(
            reward_config["reward_temp_start"],
            reward_config["reward_temp_end"],
            reward_config["reward_temp_horizon"],
        )
        self.get_advantage_alpha_at_step = self._build_linear_schedule(
            reward_config["advantage_alpha_start"],
            reward_config["advantage_alpha_end"],
            reward_config["advantage_alpha_horizon"],
        )
        self.get_scaling_factor_at_step = self._build_linear_schedule(
            reward_config["scaling_factor_start"],
            reward_config["scaling_factor_end"],
            reward_config["scaling_factor_horizon"],
        )
        self.get_use_buffer_sample_at_step = self._build_linear_schedule(
            training_mixed_config["use_buffer_sample_start_prob"],
            training_mixed_config["use_buffer_sample_end_prob"],
            training_mixed_config["buffer_sample_steps"],
        )
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

        self.naughty_vocab_alpha = float(self.constraint_config.naughty_vocab_alpha)
        self.invalid_vocab_alpha = float(self.reward_config.invalid_vocab_alpha)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile = compile

        self.train_sentence_length: list = []
        self.train_samples: list = []
        self.val_samples: list = []

        self.opt_task = self.training_mixed_config.get("opt_task", False)
        self.skip_baseline_sampling = self.training_mixed_config.skip_baseline_sampling

        try:
            if self.compile:
                self.net = torch.compile(self.net, mode="max-autotune", fullgraph=False)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"torch.compile failed, continuing without compilation: {exc}")

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
        advantage_alpha=0.01,
        scaling_factor=50,
        action_seq=None,
        use_buffer_sample: bool = False,
        buffer_sample: torch.Tensor | None = None,
        buffer_mixture_ratio: float = 0.5,
    ):
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
                model=self.net,
                tokenizer=self.tokenizer,
            ),
            "termination_token_id": self.end_of_sentence_token_id,
            "min_len": self.constraint_config.min_sentence_len,
            "max_len": self.constraint_config.max_sentence_len,
            "temperature": pf_temperature,
            "reward_temperature": reward_temperature,
            "advantage_alpha": advantage_alpha,
            "scaling_factor": scaling_factor,
            "skip_rewards": False,
            "action_seq": action_seq,
            "vocab_nice_mask": self.legal_tokens_mask,
            "vocab_naughty_mask": self.illegal_tokens_mask,
            "naughty_vocab_alpha": self.naughty_vocab_alpha,
            "invalid_vocab_alpha": self.invalid_vocab_alpha,
            "use_buffer_sample": use_buffer_sample,
            "buffer_sample": buffer_sample,
            "buffer_mixture_ratio": buffer_mixture_ratio,
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
        buffer_result = self._maybe_generate_from_buffer(item, encoded_prompt)
        used_buffer = buffer_result is not None

        if used_buffer:
            _, result_dict = buffer_result
            pf_temp = 1.0
        else:
            pf_temp = self._sample_pf_temperature()
            result_dict = self.forward(
                item,
                pf_temperature=pf_temp,
                reward_temperature=self.reward.temperature,
                scaling_factor=self.get_scaling_factor_at_step(self.global_step),
                use_buffer_sample=False,
                buffer_sample=buffer_sample,
                buffer_mixture_ratio=self.buffer_mixture_ratio,
            )

        generated_text = result_dict["state"]
        log_pf = result_dict["log_pf"]
        log_pterm = result_dict["log_pterm"]
        model_log_r = result_dict["log_r"]
        log_r_unpenalized = result_dict["log_r_unpenalized"]
        agree_list = result_dict["agree_list"]

        if used_buffer:
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
                "buffer_prob": torch.tensor(
                    self.get_use_buffer_sample_at_step(self.global_step),
                    device=self.device,
                )
            },
            on_step=True,
            sync_dist=True,
        )
        self._log_agreement_metrics(agree_list)

        loss = modified_subtb_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=prompt_len,
            subtb_lambda=self.training_mixed_config.subtb_lambda,
        )

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

        if batch_idx % 10 == 0:
            decoded = self._decode_generated_tokens(generated_text[0, prompt_len:])
            self.train_samples.append(f"{decoded}: pf_temp={pf_temp:.2f}")

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
                "train/buffer_ratio": torch.tensor(
                    self.get_use_buffer_sample_at_step(self.global_step),
                    device=self.device,
                ),
                "train/reward_var": log_r.var(dim=0).mean(),
                "train/loss": (loss, {"prog_bar": True}),
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
        log_r_unpenalized = result_dict["log_r_unpenalized"]
        self.val_samples.extend(generated_text[:, prompt_len:].tolist())

        loss = modified_subtb_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=prompt_len,
            subtb_lambda=self.training_mixed_config.subtb_lambda,
        )

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

        self.log("train/scaling_factor", scaling_factor, sync_dist=True, on_step=True)
        self.log("train/reward_temp", reward_temp, sync_dist=True, on_step=True)

        for pg in self.optimizers().param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------ #
    # Epoch hooks
    # ------------------------------------------------------------------ #

    def on_train_epoch_end(self):
        df = pd.DataFrame(self.train_samples, columns=["Sampled sentence"])
        df.to_csv(
            os.path.join(
                self.trainer.default_root_dir,
                f"samples_train_probes_{self.trainer.current_epoch}.csv",
            ),
            index=False,
        )

        plt.hist(self.train_sentence_length, bins=50)
        plt.savefig(
            os.path.join(
                self.trainer.default_root_dir,
                f"train_sentence_length_{self.trainer.current_epoch}.png",
            )
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

    def on_train_epoch_start(self):
        self.log("scheduled/R_temperature", self.reward.temperature, sync_dist=True)
        self.log("scheduled/lr", self.lr_schedulers().get_lr()[0], sync_dist=True)

    def on_validation_epoch_start(self):
        log_rs, log_pfss = [], []
        val_dataset = self.trainer.datamodule.val_dataloader().dataset
        self.val_probes = torch.utils.data.Subset(
            val_dataset, random.sample(range(len(val_dataset)), 10)
        )
        for idx, item in enumerate(self.trainer.datamodule.val_dataloader()):
            encoded_prompt = item["encoded_prompt"]
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    item[key] = value.to(self.device)
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
            samples_table.to_csv(
                os.path.join(
                    self.trainer.default_root_dir,
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
        samples = []
        for probe in probes:
            encoded_prompt = probe["encoded_prompt"]
            prompt_len = encoded_prompt.shape[-1]
            for key, value in probe.items():
                if isinstance(value, torch.Tensor):
                    probe[key] = value.to(self.device)
            with torch.no_grad():
                result_dict = self.forward(
                    probe, n_samples=n_samples, pf_temperature=1.0, reward_temperature=1.0
                )
                generated_text = result_dict["state"]
                log_r = result_dict["log_r"]
                log_r_unpenalized = result_dict["log_r_unpenalized"]
                log_pf = result_dict["log_pf"]
                log_pterm = result_dict["log_pterm"]
                log_pf_ref = result_dict.get("log_pf_ref", None)

            log_ps, log_ps_unpenalized = get_termination_vals(
                generated_text=generated_text,
                log_pf=None,
                log_pterm=None,
                log_r=log_r,
                log_r_unpenalized=log_r_unpenalized,
                termination_token_id=self.end_of_sentence_token_id,
                prompt_len=prompt_len,
            )[1:3]

            log_ps *= self.reward.temperature
            log_ps_unpenalized *= self.reward.temperature

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
                        "logPf": log_pf[idx].tolist(),
                        "logPf_ref": log_pf_ref[idx].tolist() if log_pf_ref is not None else [],
                        "logPterm": log_pterm[idx].tolist(),
                        "logR": log_r[idx].tolist(),
                        "logR unpenalized": log_r_unpenalized[idx].tolist(),
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
