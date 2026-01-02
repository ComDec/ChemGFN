### Expr24 + Llama3-1B Training Pipeline (GFN)

This document contains the full configuration and the key code paths inlined for the experiment:

- Config: `configs/experiment/baseline_expr24_zero_ds_mix_acc_tune_cfg_disable_ref.yaml`
- Overrides: `override /data: expr24`, `override /model: llama3_expr24`

---

## Entry Point (`chemgfn/train.py`)
```python
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    extras(cfg)
    if DEBUG_FLAG:
        cfg.logger.wandb.offline = True
        cfg.exp_name = "debug"
        cfg.trainer.devices = 1
    metric_dict, _ = train(cfg)
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

@task_wrapper
def train(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    logger = instantiate_loggers(cfg.get("logger"))
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    if cfg.get("train"):
        datamodule.setup("fit")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    return trainer.callback_metrics, {"cfg": cfg, "datamodule": datamodule, "model": model}
```

Run:
```bash
python chemgfn/train.py experiment=baseline_expr24_zero_ds_mix_acc_tune_cfg_disable_ref
```

## Experiment Config (overrides)
```yaml
# configs/experiment/baseline_expr24_zero_ds_mix_acc_tune_cfg_disable_ref.yaml
defaults:
  - override /data: expr24
  - override /model: llama3_expr24
trainer:
  max_steps: 60000
  gradient_clip_val: 0.5
  accumulate_grad_batches: 4
  precision: bf16-true
  devices: 4
model:
  optimizer:
    _target_: torch.optim.AdamW
    _partial_: true
    lr: 0.0001
  reward:
    _target_: chemgfn.models.reward.Target_Score_Positive
    invalid_start_ratio: 0.2
    invalid_end_ratio: 1.2
    illegal_vocab_penalty: -99
    grammar_disagree_penalty: -99
    sentence_validator:
      _target_: chemgfn.models.reward.Expr24Validator
      amortize_valid_state: false
  loss_fn:
    _target_: chemgfn.models.losses.ModifiedSubTBLoss
    subtb_lambda: 1.0
    eps: 1e-8
  reward_buffer:
    _target_: chemgfn.utils.gfn_utils.ReplayBuffer
    buffer_size: 200
    sim_tolerance: 0.25
    strict_mode: false
    buffer_aug_value: 0.0
  constraint_config:
    min_sentence_len: 1
    max_sentence_len: 7
    grammar_path: ${paths.assets_dir}/24_grammars/general.ebnf
    disable_grammar: false
    processor_type: "prefix"
    legal_tokens: ${paths.assets_dir}/token_list/24_points/general
    illegal_vocab_penalty: -50
    parse_mode: "limited"
  training_mixed_config:
    subtb_lambda: 1.0
    pf_temp_high: 2.0
    pf_temp_low: 0.5
    pf_temp_prob: 0.666
    n_samples: 10
    buffer_mixture_ratio: 1.0
    skip_baseline_sampling: true
    opt_task: false
  factor_schedulers:
    reward_temp:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 2
      end: 0.8
      horizon: 50000
    replay_buffer:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 0
      end: 0
      horizon: 50000
    dataset_buffer:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 1.0
      end: 1.0
      horizon: 25000
    reference_logits_scale:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 0
      end: 0
      horizon: 50000
data:
  num_workers: 8
```

## Data Pipeline (inline code)
```python
# configs/data/expr24.yaml
_target_: chemgfn.data.gfn_datamodule.BufferDataModule
tokenizer_name: ${model.tokenizer.pretrained_model_name_or_path}
data_path: ${paths.data_dir}/24_points/prompts.txt
train_size: 0.95
num_workers: 8
pin_memory: True
total_size: 10000
prompt_size: 1
n_samples: ${model.training_mixed_config.n_samples}  # 10
allowed_vocab_path: ${model.constraint_config.legal_tokens}
buffer_sample_path: ${paths.data_dir}/24_points/buffer_24.pt
```

```python
# chemgfn/data/gfn_datamodule.py (core)
class BufferDataModule(LightningDataModule):
    def __init__(..., prompt_size=1, total_size=10000, train_size=0.95, ...,
                 allowed_vocab_path=None, n_samples=8, buffer_tokenization=False):
        ...
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if allowed_vocab_path and os.path.exists(allowed_vocab_path):
            with open(allowed_vocab_path) as f:
                self.allowed_tokens = f.readlines()
        self.buffer_sample = torch.load(buffer_sample_path) if buffer_sample_path and os.path.exists(buffer_sample_path) else None

    def setup(self, stage):
        prompts = [p.strip() for p in open(self.data_path).readlines()][: self.prompt_size]
        num_train = int(self.total_size * self.train_size)
        self.train_data = BufferDataPipe(
            prompts, self.tokenizer, num_train, is_instruct=self.is_instruct,
            add_prompt=self.add_prompt, buffer_sample=self.buffer_sample,
            allowed_vocab=self.allowed_tokens, n_samples=self.n_samples,
            buffer_tokenization=self.buffer_tokenization,
        )
        self.val_data = BufferDataPipe(prompts, self.tokenizer, self.total_size - num_train,
                                       is_instruct=self.is_instruct, add_prompt=self.add_prompt)
```

## Model Construction (inline code)
```python
# configs/model/llama3_expr24.yaml (essentials)
_target_: chemgfn.models.gfn.ChemGFNModule
net_config:
  pretrained_model_name_or_path: "meta-llama/Llama-3.2-1B"
lora_config:
  target_modules: ["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"]
  r: 16
  lora_alpha: 16
  lora_dropout: 0.1
optimizer:
  _target_: torch.optim.AdamW
  _partial_: true
  lr: 0.0001
  weight_decay: 0.01
tokenizer:
  _target_: transformers.AutoTokenizer.from_pretrained
  pretrained_model_name_or_path: ${model.net_config.pretrained_model_name_or_path}
scheduler:
  _target_: torch.optim.lr_scheduler.PolynomialLR
  _partial_: true
  total_iters: 100000
  power: 1.0
compile: false
disable_peft: False
```

```python
# chemgfn/models/gfn.py (constructor highlights)
class ChemGFNModule(LightningModule):
    def __init__(..., net_config, lora_config, tokenizer, reward, loss_fn, reward_buffer,
                 training_mixed_config, constraint_config, optimizer, scheduler, factor_schedulers, compile, disable_peft=False):
        model = AutoModelForCausalLM.from_pretrained(net_config.pretrained_model_name_or_path)
        model_frozen = AutoModelForCausalLM.from_pretrained(net_config.pretrained_model_name_or_path)
        model_frozen.eval().requires_grad_(False)
        self.net = get_peft_model(model, lora_config) if not disable_peft else model
        self.net_frozen = model_frozen
        self.tokenizer = tokenizer
        self.factor_schedulers = factor_schedulers
        self.get_reward_temp_at_step = self.factor_schedulers["reward_temp"]
        self.get_dataset_buffer_at_step = self.factor_schedulers["dataset_buffer"]
        self.get_replay_buffer_at_step = self.factor_schedulers["replay_buffer"]
        self.reward = reward
        self.reward_buffer: ReplayBuffer = reward_buffer
        self.reward_buffer.set_termination_token_id(self.tokenizer.eos_token_id)
        self.loss_fn = loss_fn
        self.return_policy_logits = True
        if hasattr(self.loss_fn, "set_alpha_reference"):
            self.loss_fn.set_alpha_reference(self.get_alpha_reference_at_step(self.global_step))
```

## Training Step (inline code)
```python
# chemgfn/models/gfn.py (training_step)
def training_step(self, item, batch_idx):
    encoded_prompt = item["encoded_prompt"]
    prompt_len = encoded_prompt.shape[-1]
    buffer_sample = item["buffer_encoded_sample"]
    use_dataset_buffer = False

    if random.random() <= self.get_replay_buffer_at_step(self.global_step):
        replay_buffer_result = self.generate_from_replay_buffer(item, encoded_prompt)
        use_replay_buffer = replay_buffer_result is not None
    else:
        use_replay_buffer = False

    if use_replay_buffer:
        _, result_dict = replay_buffer_result
        pf_temp = 1.0
    else:
        pf_temp = self._sample_pf_temperature()
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

    log_pf, log_pterm = result_dict["log_pf"], result_dict["log_pterm"]
    log_r_reference = result_dict["log_r_reference"]
    log_r_target = result_dict["log_r_target"]
    log_pf_ref = result_dict["log_pf_ref"]
    log_pterm_ref = result_dict["log_pterm_ref"]
    model_log_r = result_dict["log_r"]
    log_r_unpenalized = result_dict["log_r_unpenalized"]
    generated_text = result_dict["state"]
    agree_list = result_dict["agree_list"]

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

    loss_output = self.loss_fn(
        log_pf=log_pf,
        log_r=log_r,
        log_r_reference=log_r_reference,
        log_r_target=log_r_target,
        log_pterm=log_pterm,
        generated_text=generated_text,
        termination_token_id=self.end_of_sentence_token_id,
        prompt_len=prompt_len,
        global_step=self.global_step,
        weight_overrides={
            name: sched(self.global_step)
            for name, sched in getattr(self, "loss_weight_schedulers", {}).items()
        } if getattr(self, "loss_weight_schedulers", None) else None,
    )
    loss = loss_output["loss"] if isinstance(loss_output, dict) else loss_output
    ...
    return loss
```

## Generation and Reward (inline code)
```python
# chemgfn/utils/gfn_utils.py (excerpt)
def generate_and_return_termination_logprob(model, encoded_data, termination_token_id, reward_fn,
    grammar_processor=None, vocab_nice_mask=None, vocab_invalid_mask=None,
    illegal_vocab_penalty=float("-inf"), max_len=10, min_len=0, temperature=1.0,
    reward_temperature=1.0, action_seq=None, skip_rewards=False,
    use_buffer_sample=False, buffer_sample=None, buffer_mixture_ratio=0.5, disable_grammar=False, **kwargs):
    encoded_prompt = encoded_data["encoded_prompt"]
    active_seqs = torch.ones(encoded_prompt.size(0), dtype=torch.bool, device=encoded_prompt.device)
    prompt_len = encoded_prompt.size(1)
    state = encoded_prompt.clone()
    log_pf, log_pterm, agree_entries = [], [], []
    ...
    if use_buffer_sample and buffer_sample is not None:
        nums_replace = max(1, int(encoded_prompt.size(0) * buffer_mixture_ratio))
    for step in range(max_len + 1):
        output = model(input_ids=token_ids, past_key_values=past_key_values)
        logits = output.logits[:, -1, :]
        if action_seq is None:
            scores = logits.clone().detach()
            if vocab_nice_mask is not None:
                scores[:, vocab_invalid_mask] = -torch.inf
            results = logits_processor(state, scores, disable_grammar=disable_grammar)
            modified_logits = results["masked_logits"] if isinstance(results, dict) else results
            agree_entries.append(results["acceptance"] if isinstance(results, dict) else torch.ones_like(modified_logits, dtype=torch.bool))
            if step < min_len:
                non_eos_only = torch.where(results["acceptance"].sum(dim=1) != 1)[0]
                modified_logits[non_eos_only, termination_token_id] = -torch.inf
            elif step >= max_len:
                mask = torch.ones_like(modified_logits, dtype=torch.bool)
                mask[:, termination_token_id] = False
                modified_logits[mask] = -torch.inf
                modified_logits[:, termination_token_id] = 0
            prob = (modified_logits / temperature).softmax(dim=-1)
            token_ids = torch.multinomial(prob, num_samples=1)
            if use_buffer_sample and buffer_sample is not None:
                if step < buffer_sample.size(-1):
                    if step >= max_len:
                        token_ids[:nums_replace, :] = termination_token_id
                    else:
                        token_ids[:nums_replace, :] = buffer_sample[:nums_replace, step].unsqueeze(-1)
                else:
                    token_ids[:nums_replace, :] = termination_token_id
        ...
    if not skip_rewards:
        reward_results = reward_fn(
            state[:, :-1], reward_temperature=reward_temperature,
            scaling_factor=kwargs.get("scaling_factor", 0.0),
            reference_logits_scale=kwargs.get("reference_logits_scale", 0.0),
            vocab_invalid_mask=vocab_invalid_mask,
            illegal_vocab_penalty=illegal_vocab_penalty,
            agree_list=_stack_if_not_empty(agree_entries),
            termination_token_id=termination_token_id,
            target_molecule=encoded_data.get("molecule"),
            action_seq=action_seq,
        )
    return {..., "log_r": reward_results["reward"], "log_r_unpenalized": reward_results["reward_unpenalized"], ...}
```

```python
# chemgfn/models/reward.py (Target_Score_Positive)
class Target_Score_Positive:
    def __init__(self, sentence_validator, invalid_start_ratio=0.2, invalid_end_ratio=1.2,
                 disable_peft=False, illegal_vocab_penalty=-99, grammar_disagree_penalty=-99, **kwargs):
        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = float(illegal_vocab_penalty)
        self.grammar_disagree_penalty = float(grammar_disagree_penalty)
        self.temperature = 1.0

    def score(self, input_batch, prompt_length, model, tokenizer, reward_temperature=1.0,
              vocab_invalid_mask=None, scaling_factor=0.5, reference_logits_scale=0.5,
              target_molecule=None, agree_list=None, action_seq=None, **kwargs):
        reference_results = score_fast(
            model=model, encoded_input=input_batch, termination_token_id=tokenizer.eos_token_id,
            skip_first=prompt_length, reward_temperature=reward_temperature,
            invalid_vocab_mask=vocab_invalid_mask, agree_list=agree_list,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
            grammar_disagree_penalty=self.grammar_disagree_penalty,
        )
        reward_unpenalized = reference_results["reward"]
        ref_log_pf = reference_results["ref_log_pf"]
        ref_log_pterm = reference_results["ref_log_pterm"]
        validator_dict = self.sentence_validator(input_batch[:, prompt_length:], tokenizer, target_molecule) if self.sentence_validator else None
        reward_penalized = validator_dict["local_score"] if validator_dict is not None else reward_unpenalized
        return {
            "reward": reward_penalized,
            "reward_unpenalized": reward_unpenalized,
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
        }
```

## Loss (inline code)
```python
# chemgfn/models/losses.py (ModifiedSubTBLoss)
class ModifiedSubTBLoss(GFNLoss):
    def forward(self, log_pf, log_r, log_pterm, generated_text, termination_token_id, prompt_len, **kwargs):
        delta = (
            log_r[:, :-1] + log_pf[:, :-1] + log_pterm[:, 1:]
            - log_r[:, 1:] - log_pterm[:, :-1]
        )
        delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1).cumsum(dim=1)
        mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(dim=-1) >= 1
        batch_loss = 0.0
        total_lambda = 0.0
        generated_len = generated_text.shape[1] - prompt_len
        for subtraj_len in range(1, generated_len):
            subtb_term = (delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]) ** 2
            subtb_term[mask[:, subtraj_len - 1 :]] = 0
            batch_loss += self.subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
            total_lambda += self.subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
        batch_loss /= total_lambda
        return {"loss": batch_loss}
```

## Training/Validation Hooks (inline code)
```python
# chemgfn/models/gfn.py (selected)
def on_train_batch_start(self, batch, batch_idx):
    reward_temp = self.get_reward_temp_at_step(self.global_step)
    scaling_factor = self.get_scaling_factor_at_step(self.global_step)
    lr = self.lr_schedulers().get_lr()[0]
    self.reward.temperature = reward_temp
    self.reward.scaling_factor = scaling_factor
    for pg in self.optimizers().param_groups:
        pg["lr"] = lr

def validation_step(self, batch, batch_idx):
    result_dict = self.forward(batch, reward_temperature=1.0, pf_temperature=1.0)
    loss_output = self.loss_fn(...)
    # logs validator metrics, logR/P, etc.

def on_validation_epoch_start(self):
    # probe sampling, Var(logR - logPf(s)), save CSV
    ...

def on_validation_epoch_end(self):
    diversity = calculate_diversity(torch.tensor(self.val_samples))
    self.log("val/diversity", diversity, sync_dist=True, on_epoch=True)
```

## Key Behaviors for This Experiment
- Data: single prompt resampled to 10k; optional buffer_24.pt mixed with probability 1.0, `buffer_mixture_ratio=1.0`, `n_samples=10`.
- Grammar: prefix incremental parser on `general.ebnf`, length 1–7, legal vocab list enforced.
- Rewards: frozen model log rewards, validator gives final-step 0/1; illegal/grammar penalties -99; constraint illegal vocab penalty -50.
- Loss: SubTB on generated trajectories.
- Replay buffer: written every step, but sampling prob scheduled 0 → 0 (unused for generation).
- Temps/schedules: reward_temp 2→0.8 (50k), pf_temp sampled with prob 0.666 in [0.5,2.0]; reference logits scale 0.

## How to run / tweak
- Train: `python chemgfn/train.py experiment=baseline_expr24_zero_ds_mix_acc_tune_cfg_disable_ref`
- To disable grammar disagreement penalty: set `model.reward.grammar_disagree_penalty: 0`.
- To stop dataset buffer mixing: set `model.factor_schedulers.dataset_buffer.start=end=0` or remove `buffer_sample_path`.
