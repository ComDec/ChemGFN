import os
import random
import sys
from functools import partial
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
import wandb
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
    base_to_lora,
    generate_and_return_termination_logprob,
    generate_and_return_termination_logprob_for_sidechain_opt,
    get_termination_vals,
    lora_to_base,
    modified_subtb_loss,
    prepare_token_mask,
)

sys.setrecursionlimit(1500)


class ChemGFNModule(LightningModule):
    def __init__(
        self,
        net_config: Dict[str, Any],
        lora_config: LoraConfig,
        tokenizer: PreTrainedTokenizer,
        reward,
        reward_buffer,
        reward_config: Dict[str, Any],
        training_mixed_config: Dict[str, Any],
        constraint_config: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `ChemGFNLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"])
        model = AutoModelForCausalLM.from_pretrained(net_config.pretrained_model_name_or_path)
        self.net = get_peft_model(model, lora_config)
        self.tokenizer = tokenizer

        self.reward_config = reward_config
        self.constraint_config = constraint_config
        self.training_mixed_config = training_mixed_config
        self.end_of_sentence_token_id = self.tokenizer.eos_token_id

        # set reward temp at certain step
        self.get_reward_temp_at_step = lambda step: reward_config["reward_temp_start"] + (
            reward_config["reward_temp_end"] - reward_config["reward_temp_start"]
        ) * min(1, step / reward_config["reward_temp_horizon"])

        self.get_advantage_alpha_at_step = lambda step: (
            reward_config["advantage_alpha_start"]
            - (reward_config["advantage_alpha_start"] - reward_config["advantage_alpha_end"])
            * min(1, step / reward_config["advantage_alpha_horizon"])
        )

        # set use buffer sample at certain step
        self.get_use_buffer_sample_at_step = lambda step: (
            training_mixed_config["use_buffer_sample_start_prob"]
            - (
                training_mixed_config["use_buffer_sample_start_prob"]
                - training_mixed_config["use_buffer_sample_end_prob"]
            )
            * min(1, step / training_mixed_config["buffer_sample_steps"])
        )
        self.buffer_mixture_ratio = training_mixed_config["buffer_mixture_ratio"]

        self.reward = reward
        self.reward_buffer = reward_buffer
        self.reward_buffer.set_termination_token_id(self.end_of_sentence_token_id)
        if os.path.exists(self.constraint_config.legal_tokens):
            (
                self.legal_tokens_mask,
                self.illegal_tokens_mask,
                self.legal_token_ids_list,
            ) = prepare_token_mask(self.tokenizer, self.constraint_config.legal_tokens)
        else:
            # self.legal_tokens_mask = torch.zeros(self.tokenizer.vocab_size, dtype=torch.bool)
            self.legal_tokens_mask = None
            self.illegal_tokens_mask = None
            self.legal_token_ids_list = None
            print(f"Legal tokens file not found: {self.constraint_config.legal_tokens}")

        if constraint_config.apply_grammar:
            with open(constraint_config.grammar_path) as file:
                grammar_str = file.read()

            parsed_grammar = parse_ebnf(grammar_str)
            self.string_grammar = StringRecognizer(
                parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"]
            )
            try:
                self.grammar = IncrementalGrammarConstraint(grammar_str, "root", self.tokenizer)
            except:
                self.grammar = None
                print("Grammar parsing failed with current tokenizer, disable general processor")
            self.grammar_processor = GrammarLimitedOneTimeLogitsProcessor(
                parsed_grammar,
                tokenizer=self.tokenizer,
                nice_token_ids_list=self.legal_token_ids_list,
                execution_mode=self.constraint_config.parse_mode,
            )
            if self.constraint_config.processor_type == "prefix":
                self.pre_grammar_processor = GrammarIncrementalLogitsProcessorGeneral(
                    parsed_grammar,
                    tokenizer=self.tokenizer,
                    nice_token_ids_list=self.legal_token_ids_list,
                    execution_mode=self.constraint_config.parse_mode,
                )

            elif self.constraint_config.processor_type == "prefix_enhanced":
                self.pre_grammar_processor = GrammarIncrementalLogitsProcessorSampleEnhanced(
                    parsed_grammar,
                    tokenizer=self.tokenizer,
                    nice_token_ids_list=self.legal_token_ids_list,
                    execution_mode=self.constraint_config.parse_mode,
                )

            elif self.constraint_config.processor_type == "parenthese":
                self.pre_grammar_processor = GrammarLogitsProcessorPartheseness(
                    parsed_grammar,
                    tokenizer=self.tokenizer,
                    nice_token_ids_list=self.legal_token_ids_list,
                    execution_mode=self.constraint_config.parse_mode,
                )
            elif self.constraint_config.processor_type == "number_only":
                self.pre_grammar_processor = GrammarIncrementalLogitsProcessorForNumberOnly(
                    parsed_grammar,
                    tokenizer=self.tokenizer,
                    nice_token_ids_list=self.legal_token_ids_list,
                    execution_mode=self.constraint_config.parse_mode,
                )
            elif self.constraint_config.processor_type == "general":
                self.pre_grammar_processor = GrammarConstrainedLogitsProcessor(
                    self.grammar,
                )
            elif self.constraint_config.processor_type == "none":
                self.pre_grammar_processor = None

        else:
            self.grammar_processor = None

        self.naughty_vocab_alpha = float(self.constraint_config.naughty_vocab_alpha)
        self.invalid_vocab_alpha = float(self.reward_config.invalid_vocab_alpha)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile = compile

        # metrics
        self.train_sentence_length = []
        self.train_samples = []

        # other
        self.opt_task = self.training_mixed_config.get("opt_task", False)
        self.skip_baseline_sampling = self.training_mixed_config.skip_baseline_sampling

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
        action_seq=None,
        use_buffer_sample: bool = False,
        buffer_sample: Optional[torch.Tensor] = None,
        buffer_mixture_ratio: float = 0.5,
    ):
        encoded_prompt = encoded_data["encoded_prompt"]
        encoded_prompt = encoded_prompt.squeeze(0) if encoded_prompt.ndim != 1 else encoded_prompt
        n_samples = n_samples or self.training_mixed_config.n_samples
        encoded_prompt = encoded_prompt.expand(n_samples, -1)  # 批量扩展

        # update encoded_data with the expanded prompt
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

        result = generator(**generation_config)

        return result

    def training_step(self, item, batch_idx) -> torch.Tensor:
        encoded_prompt = item["encoded_prompt"]
        buffer_sample = item["buffer_encoded_sample"]
        # Sample a sentence and get the reward

        if (
            random.random() < self.training_mixed_config.use_buffer_prob
            and self.reward_buffer.sample(self.training_mixed_config.n_samples, encoded_prompt)[0]
            is not None
        ):
            # Using a sample from the reward buffer
            action_seq, log_r = self.reward_buffer.sample(
                self.training_mixed_config.n_samples, encoded_prompt
            )
            result_dict = self.forward(item, action_seq=action_seq)
            generated_text = result_dict["state"]
            log_pf = result_dict["log_pf"]
            log_pterm = result_dict["log_pterm"]
            log_r = result_dict["log_r"]
            log_r_unpenalized = result_dict["log_r_unpenalized"]
            agree_list = result_dict["agree_list"]

            log_r = log_r[
                :, : generated_text.shape[1] - len(encoded_prompt)
            ]  # Undo padding from buffer
            # log_r *= 1 / self.reward.temperature  # redo the effect of reward tempering
            pf_temp = 1.0

        else:
            # Using the forward policy

            if random.random() < self.get_use_buffer_sample_at_step(self.global_step):
                use_buffer_sample = True
            else:
                use_buffer_sample = False

            if random.random() < self.training_mixed_config.pf_temp_prob:  # With tempering
                pf_temp = (
                    random.random()
                    * (
                        self.training_mixed_config.pf_temp_high
                        - self.training_mixed_config.pf_temp_low
                    )
                    + self.training_mixed_config.pf_temp_low
                )
            else:  # Without tempering
                pf_temp = 1.0

            result_dict = self.forward(
                item,
                pf_temperature=pf_temp,
                reward_temperature=self.reward.temperature,
                advantage_alpha=self.get_advantage_alpha_at_step(self.global_step),
                use_buffer_sample=use_buffer_sample,
                buffer_sample=buffer_sample,
                buffer_mixture_ratio=self.buffer_mixture_ratio,
            )

            generated_text = result_dict["state"]
            log_pf = result_dict["log_pf"]
            log_pterm = result_dict["log_pterm"]
            log_r = result_dict["log_r"]
            log_r_unpenalized = result_dict["log_r_unpenalized"]
            agree_list = result_dict["agree_list"]

            self.reward_buffer.add_batch(
                prompt=encoded_prompt,
                sentences=generated_text[:, len(encoded_prompt) :],
                logrewards=log_r * self.reward.temperature,  # undo the effect of reward tempering
                tokenizer=self.tokenizer,
            )

        # sum of agree list
        agree_sum = [torch.mean(x.sum(-1).float()).item() for x in agree_list]

        self.log(
            "train/agree_mean",
            torch.tensor(agree_sum, device=self.device).mean(),
            sync_dist=True,
            on_step=True,
        )

        self.log(
            "train/agree_start",
            torch.tensor(agree_sum[0], device=self.device),
            sync_dist=True,
            on_step=True,
        )

        self.log(
            "train/agree_midd",
            torch.tensor(agree_sum[int(len(agree_sum) / 2)], device=self.device),
            sync_dist=True,
            on_step=True,
        )

        self.log(
            "train/agree_end",
            torch.tensor(agree_sum[-1], device=self.device),
            sync_dist=True,
            on_step=True,
        )

        # Get the GFN loss
        loss = modified_subtb_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(encoded_prompt[0]),
            subtb_lambda=self.training_mixed_config.subtb_lambda,
        )

        # Log metrics
        _, last_log_r, last_log_r_unpenalized, sentence_len = get_termination_vals(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(encoded_prompt[0]),
        )

        validator_metric_dict = self.reward.sentence_validator.accuracy(
            generated_text[:, encoded_prompt.shape[-1] :], self.tokenizer
        )
        self.train_sentence_length.append(sentence_len.detach().cpu())

        log_ps = last_log_r * self.reward.temperature
        log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature

        if batch_idx % 10 == 0:
            if self.constraint_config.processor_type == "number_only":
                text = []
                raw_tokens = generated_text[0, encoded_prompt.shape[-1] :]
                for t in raw_tokens:
                    if t.item() != self.tokenizer.eos_token_id:
                        text.append(self.tokenizer.decode(t))
                    else:
                        break
                self.train_samples.append(",".join(text) + f": pf_temp={pf_temp:.2f}")
            else:
                self.train_samples.append(
                    self.tokenizer.decode(
                        generated_text[0, encoded_prompt.shape[-1] :], skip_special_tokens=True
                    )
                    + f": pf_temp={pf_temp:.2f}"
                )

        self.log(
            "train/buffer_ratio",
            self.get_use_buffer_sample_at_step(self.global_step),
            sync_dist=True,
            on_step=True,
        )

        self.log(
            "train/reward_var",
            log_r.var(dim=0).mean(),
            sync_dist=True,
            on_step=True,
        )

        self.log(
            "train/loss",
            loss,
            on_step=True,
            sync_dist=True,
            prog_bar=True,
        )

        for key, value in validator_metric_dict.items():
            self.log(
                f"train/validator_{key}",
                value,
                on_step=True,
                sync_dist=True,
                prog_bar=True,
            )

        self.log(
            "train/logR",
            last_log_r.mean(),
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) (avg)",
            log_ps.mean(),
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) (max)",
            log_ps.max(),
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) unpenalized (avg)",
            log_ps_unpenalized.mean(),
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) unpenalized (max)",
            log_ps_unpenalized.max(),
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "train/sentence_len",
            sentence_len.float().mean(),
            on_step=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        # Should always be (1, prompt_len)
        encoded_prompt = batch["encoded_prompt"]
        # Sample a sentence and get the reward
        result_dict = self.forward(batch, reward_temperature=1.0, pf_temperature=1.0)
        generated_text = result_dict["state"]
        log_pf = result_dict["log_pf"]
        log_pterm = result_dict["log_pterm"]
        log_r = result_dict["log_r"]
        log_r_unpenalized = result_dict["log_r_unpenalized"]

        # Get the GFN loss
        loss = modified_subtb_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(encoded_prompt[0]),
            subtb_lambda=self.training_mixed_config.subtb_lambda,
        )

        # Log metrics
        _, last_log_r, last_log_r_unpenalized, sentence_len = get_termination_vals(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(encoded_prompt[0]),
        )

        validator_metric_dict = self.reward.sentence_validator.accuracy(
            generated_text[:, encoded_prompt.shape[-1] :], self.tokenizer
        )

        log_ps = last_log_r * self.reward.temperature
        log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature

        for key, value in validator_metric_dict.items():
            self.log(
                f"val/validator_{key}",
                value,
                sync_dist=True,
                on_epoch=True,
            )

        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val/logR",
            last_log_r.mean(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) (avg)",
            log_ps.mean(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) (max)",
            log_ps.max(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) unpenalized (avg)",
            log_ps_unpenalized.mean(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) unpenalized (max)",
            log_ps_unpenalized.max(),
            sync_dist=True,
        )

    def on_train_batch_start(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        reward_temp = self.get_reward_temp_at_step(self.global_step)
        advantage_alpha = self.get_advantage_alpha_at_step(self.global_step)
        lr = self.lr_schedulers().get_lr()[0]
        self.reward.temperature = reward_temp
        self.reward.advantage_alpha = advantage_alpha

        self.log("train/advantage_alpha", advantage_alpha, sync_dist=True, on_step=True)
        self.log("train/reward_temp", reward_temp, sync_dist=True, on_step=True)

        for pg in self.optimizers().param_groups:
            pg["lr"] = lr

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

        self.logger.log_table(
            "train/samples_latest",
            dataframe=df,
        )
        self.train_samples.clear()
        self.train_sentence_length.clear()

    def on_train_epoch_start(self):
        # Log scheduled quantities
        self.log("scheduled/R_temperature", self.reward.temperature, sync_dist=True)
        self.log("scheduled/lr", self.lr_schedulers().get_lr()[0], sync_dist=True)

        # Log probe samples
        # There is no need to sample probes during training
        # if (
        #     self.train_probes is not None
        #     and self.logger is not None
        #     and self.trainer.current_epoch % 1 == 0
        # ):
        #     samples_table = self.sample_probes(self.train_probes)
        #     self.logger.log_table("samples/train_probes", dataframe=samples_table)

    def on_validation_epoch_start(self):
        # Log variance of (logR - logP(s)) using exploration, which should be 0.0
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
            agree_list = result_dict["agree_list"]

            # Get the GFN loss
            log_pfs, log_r, _, _ = get_termination_vals(
                generated_text=generated_text,
                log_pf=log_pf,
                log_pterm=log_pterm,
                log_r=log_r,
                log_r_unpenalized=log_r_unpenalized,
                termination_token_id=self.end_of_sentence_token_id,
                prompt_len=len(encoded_prompt[0]),
            )
            log_rs.append(log_r)
            log_pfss.append(log_pfs)

            if idx == 10:
                break

        log_rs, log_pfss = torch.cat(log_rs), torch.cat(log_pfss)
        self.log("val/Var(logR - logPf(s))", (log_rs - log_pfss).var(), sync_dist=True)

        # Log probe samples
        if self.val_probes is not None and self.logger is not None:
            samples_table = self.sample_probes(self.val_probes)
            samples_table.to_csv(
                os.path.join(
                    self.trainer.default_root_dir,
                    f"samples_val_probes_{self.trainer.global_step}.csv",
                ),
                index=False,
            )

    def on_train_start(self):
        if self.skip_baseline_sampling:
            return
        # Log baseline metrics
        val_data = self.trainer.datamodule.val_dataloader().dataset
        # baseline_performance = None
        samples = {}
        for idx, prompt in enumerate(val_data):
            prompt = prompt["encoded_prompt"]
            samples_ = self.sample_baselines(prompt.to(self.device), n_samples=8)

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
        # if baseline_performance is None:
        # baseline_performance = pd.DataFrame(
        #     data=np.zeros((5, len(samples))),
        #     columns=samples.keys(),
        #     index=[
        #         "logP(s) (avg)",
        #         "logP(s) (max)",
        #         "logP(s) unpenalized (avg)",
        #         "logP(s) unpenalized (max)",
        #         "sentence length",
        #     ],
        # )
        # for baseline in samples:
        # baseline_performance.loc["logP(s) (avg)", baseline] += samples[baseline]["logP(s)"].mean().item() / len(
        #     val_data
        # )
        # baseline_performance.loc["logP(s) (max)", baseline] += samples[baseline]["logP(s)"].max().item() / len(
        #     val_data
        # )
        # baseline_performance.loc["logP(s) unpenalized (avg)", baseline] += samples[baseline][
        #     "logP(s) unpenalized"
        # ].mean().item() / len(val_data)
        # baseline_performance.loc["logP(s) unpenalized (max)", baseline] += samples[baseline][
        #     "logP(s) unpenalized"
        # ].max().item() / len(val_data)
        # if samples[baseline][self.diversity_metric_name] is None:
        #     baseline_performance.loc[self.diversity_metric_name, baseline] = None
        # else:
        #     baseline_performance.loc[self.diversity_metric_name, baseline] += samples[baseline][
        #         self.diversity_metric_name
        #     ] / len(val_data)
        # baseline_performance.loc["sentence length", baseline] += samples[baseline][
        #     "sentence length"
        # ].float().mean().item() / len(val_data)
        # baseline_performance = baseline_performance.reset_index(names="metric")
        # if self.logger is not None:
        #     self.logger.log_table("val/baseline performance", dataframe=baseline_performance)

        # Log baseline probes
        # if self.hparams.val_probes is not None and self.logger is not None:
        #     samples_table = self.sample_probes_baselines(self.hparams.val_probes)
        #     self.logger.log_table("samples/val_probes (baselines)", dataframe=samples_table)

    def sample_probes(self, probes, n_samples=4):
        samples = []
        for probe in probes:
            encoded_prompt = probe["encoded_prompt"]
            probe_str = self.tokenizer.decode(encoded_prompt[0])
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
                prompt_len=len(encoded_prompt[0]),
            )[1:3]

            log_ps *= self.reward.temperature
            log_ps_unpenalized *= self.reward.temperature

            generated_text = generated_text[:, len(encoded_prompt[0]) :]
            if self.constraint_config.processor_type == "number_only":
                generated_text = [
                    ",".join(
                        self.tokenizer.decode(token, skip_special_tokens=False) for token in text
                    )
                    for text in generated_text
                ]
            else:
                generated_text = [
                    self.tokenizer.decode(text, skip_special_tokens=False)
                    for text in generated_text
                ]

            if result_dict.get("full_tokens", None) is not None:
                generated_text = result_dict["full_tokens"]

            for i in range(len(generated_text)):
                samples.append(
                    {
                        "Sampled sentence": generated_text[i],
                        "logPf": log_pf[i].tolist(),
                        "logPf_ref": log_pf_ref[i].tolist() if log_pf_ref is not None else [],
                        "logPterm": log_pterm[i].tolist(),
                        "logR": log_r[i].tolist(),
                        "logR unpenalized": log_r_unpenalized[i].tolist(),
                    }
                )
        samples = pd.DataFrame(samples)
        return samples

    def sample_probes_baselines(self, probes, n_samples=4):
        assert isinstance(probes, list) and probes[0].ndim == 1
        samples = []
        for probe in probes:
            probe_str = self.tokenizer.decode(probe)
            probe_samples = self.sample_baselines(probe.to(self.device), n_samples=n_samples)
            for i in range(n_samples):
                sample = {"Prompt": probe_str}
                for baseline in probe_samples:
                    sample[f"Sampled sentence ({baseline})"] = probe_samples[baseline]["sample"][i]
                    sample[f"logP(s) ({baseline})"] = probe_samples[baseline]["logP(s)"][i].item()
                    sample[f"logP(s) unpenalized ({baseline})"] = probe_samples[baseline][
                        "logP(s) unpenalized"
                    ][i].item()
                samples.append(sample)

        samples = pd.DataFrame(samples)
        samples = samples.sort_values(by=["Prompt"], ascending=False)

        return samples

    def sample_baselines(self, prompt, n_samples=4):
        # https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationMixin.generate
        # https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationConfig
        assert prompt.ndim == 2
        # prompt = prompt.unsqueeze(0)

        def generate(prompt, **kwargs):
            with torch.no_grad():
                lora_to_base(self.net)

                try:
                    self.pre_grammar_processor.set_return_dict(False)
                except AttributeError:
                    self.pre_grammar_processor.return_dict = False

                generated_text = self.net.generate(
                    prompt,
                    min_new_tokens=self.constraint_config.min_sentence_len,
                    max_new_tokens=self.constraint_config.max_sentence_len + 1,
                    eos_token_id=self.end_of_sentence_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    forced_eos_token_id=self.end_of_sentence_token_id,
                    suppress_tokens=None,
                    logits_processor=[self.pre_grammar_processor],
                    **kwargs,
                )
                base_to_lora(self.net)

                # restore the return_dict setting
                try:
                    self.pre_grammar_processor.set_return_dict(True)
                except AttributeError:
                    self.pre_grammar_processor.return_dict = True

                # log_r, log_r_unpenalized = self.reward.score(
                #     generated_text,
                #     prompt_length=prompt.shape[1],
                #     model=self.net,
                #     tokenizer=self.tokenizer,
                # )
                # (
                #     _,
                #     last_log_r,
                #     last_log_r_unpenalized,
                #     sentence_len,
                # ) = get_termination_vals(
                #     generated_text=generated_text,
                #     log_pf=None,
                #     log_pterm=None,
                #     log_r=log_r,
                #     log_r_unpenalized=log_r_unpenalized,
                #     termination_token_id=self.end_of_sentence_token_id,
                #     prompt_len=prompt.shape[1],
                # )
                # log_ps = last_log_r * self.reward.temperature
                # log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature

            generated_text = generated_text[:, prompt.shape[1] :]
            generated_text = torch.where(
                generated_text == self.tokenizer.eos_token_id,
                self.end_of_sentence_token_id,
                generated_text,
            )
            generated_text = self.tokenizer.batch_decode(generated_text)
            generated_text = [text.replace(".", "") for text in generated_text]

            # if len(generated_text) > 1:
            #     diversity = self.diversity_metric(generated_text)
            # else:
            #     diversity = None

            # if len(generated_text) == 1:
            #     generated_text = generated_text * n_samples
            #     log_ps = log_ps.expand(n_samples, -1)
            #     log_ps_unpenalized = log_ps_unpenalized.expand(n_samples, -1)

            return {
                "sample": generated_text,
            }

        samples = {}

        # Beam search
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

        # Diverse beam search
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

        # LM greedy
        samples["LM"] = generate(
            prompt=prompt,
            do_sample=False,
            num_return_sequences=1,
            top_k=0,
        )

        # LM with temperature greedy
        samples["LM tempered"] = generate(
            prompt=prompt,
            do_sample=False,
            num_return_sequences=1,
            top_k=0,
            temperature=2.0,
        )

        # Greedy
        samples["greedy"] = generate(
            prompt=prompt,
            do_sample=False,
        )

        # Nucleaus sampling greedy
        samples["nucleus"] = generate(
            prompt=prompt,
            do_sample=False,
            num_return_sequences=1,
            top_k=0,
            top_p=0.95,
        )

        return samples

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
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
