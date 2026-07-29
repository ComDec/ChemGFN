"""Sampling utilities for GFlowNet fine-tuning of language models.

The core entry point is :func:`generate_and_return_termination_logprob`, which rolls out the
policy autoregressively under a grammar constraint, records the per-step forward and termination
log-probabilities that the trajectory-balance objectives consume, and scores the resulting
sequences with a reward module.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional

import torch
from transformers import PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessorList


def lora_to_base(model) -> None:
    """Disable the LoRA adapters and switch to eval mode, exposing the frozen base policy.

    Args:
        model: PEFT-wrapped causal language model.
    """
    model.base_model.disable_adapter_layers()
    model.eval()


def base_to_lora(model) -> None:
    """Re-enable the LoRA adapters and switch back to train mode.

    Args:
        model: PEFT-wrapped causal language model.
    """
    model.base_model.enable_adapter_layers()
    model.train()


def prepare_token_mask(
    tokenizer: PreTrainedTokenizer, vocab_path: str
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build boolean vocabulary masks from a file listing the task's legal tokens.

    Each line of ``vocab_path`` holds one legal token. EOS is always legal so that trajectories
    can terminate; BOS is always illegal so that it is never resampled mid-trajectory.

    Args:
        tokenizer: Tokenizer used for generation.
        vocab_path: Path to the legal-token file, one token per line.

    Returns:
        Tuple of ``(legal_token_mask, illegal_token_mask, legal_token_ids)``, where the masks are
        boolean tensors of shape ``[vocab_size]`` and ``legal_token_ids`` lists the legal ids.
    """
    with open(vocab_path) as f:
        legal_tokens = [line.rstrip("\n") for line in f.readlines()]

    legal_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in legal_tokens]

    legal_token_mask = torch.zeros(len(tokenizer), dtype=torch.bool)
    legal_token_mask[legal_token_ids] = True
    legal_token_mask[tokenizer.bos_token_id] = False
    legal_token_mask[tokenizer.eos_token_id] = True

    illegal_token_mask = ~legal_token_mask

    return legal_token_mask, illegal_token_mask, legal_token_ids


def prepare_token_mask_from_illegal(
    tokenizer: PreTrainedTokenizer, illegal_tokens: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """Build boolean vocabulary masks by excluding tokens from the full vocabulary.

    Open-vocabulary tasks such as CommonGen cannot enumerate their legal tokens, so they name the
    few tokens to suppress instead. Only tokens that encode to a single id can be masked; anything
    that tokenizes into multiple pieces has no single vocabulary entry to switch off.

    Args:
        tokenizer: Tokenizer used for generation.
        illegal_tokens: Token strings to forbid.

    Returns:
        Tuple of ``(legal_token_mask, illegal_token_mask, None)``. The third element is ``None``
        because this mode has no explicit legal-id list.

    Raises:
        ValueError: If none of ``illegal_tokens`` maps to a single vocabulary entry.
    """
    illegal_ids: list[int] = []
    for token in illegal_tokens:
        ids = tokenizer.encode(str(token), add_special_tokens=False)
        if len(ids) == 1:
            illegal_ids.append(int(ids[0]))

    if not illegal_ids:
        raise ValueError(
            f"None of the configured illegal tokens {list(illegal_tokens)!r} maps to a single "
            "token id for this tokenizer, so no vocabulary entry can be masked."
        )

    illegal_token_mask = torch.zeros(len(tokenizer), dtype=torch.bool)
    illegal_token_mask[illegal_ids] = True

    return ~illegal_token_mask, illegal_token_mask, None


def calculate_diversity(token_id_list: torch.Tensor) -> float:
    """Average per-position token entropy of a batch of samples.

    Args:
        token_id_list: Token ids of shape ``[num_samples, seq_len]``.

    Returns:
        Mean entropy over sequence positions; higher values indicate more diverse samples.
    """
    num_samples, seq_len = token_id_list.shape

    if num_samples == 1:
        return 0.0

    total_entropy = 0.0

    for pos in range(seq_len):
        tokens = token_id_list[:, pos]
        _, counts = torch.unique(tokens, return_counts=True)
        probs = counts.float() / num_samples
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        total_entropy += entropy.item()

    return total_entropy / seq_len


def calculate_diversity_by_length(token_id_list, eos_id: int) -> dict[int, float]:
    """Group samples by their pre-EOS length and report the diversity of each group.

    Args:
        token_id_list: Tensor of shape ``[num_samples, seq_len]`` or an iterable of id sequences.
        eos_id: Token id that terminates a trajectory.

    Returns:
        Mapping from sequence length to the average per-position entropy at that length.
    """
    if isinstance(token_id_list, torch.Tensor):
        seqs = token_id_list.tolist()
    else:
        seqs = [list(seq) for seq in token_id_list]

    groups: dict[int, list[list[int]]] = {}
    for seq in seqs:
        try:
            eos_pos = seq.index(eos_id)
        except ValueError:
            eos_pos = len(seq)
        length = int(eos_pos)
        groups.setdefault(length, []).append(seq[:length])

    out: dict[int, float] = {}
    for length, group in groups.items():
        if length <= 0 or len(group) <= 1:
            out[length] = 0.0
            continue
        tensor = torch.tensor(group, dtype=torch.long)
        out[length] = float(calculate_diversity(tensor))
    return out


def generate_and_return_termination_logprob(
    model,
    encoded_data: dict[str, Any],
    termination_token_id: int,
    reward_fn: Callable[..., dict[str, Any]],
    grammar_processor=None,
    vocab_nice_mask: Optional[torch.Tensor] = None,
    vocab_invalid_mask: Optional[torch.Tensor] = None,
    illegal_vocab_penalty: float = float("-inf"),
    max_len: int = 10,
    min_len: int = 0,
    temperature: float = 1.0,
    reward_temperature: float = 1.0,
    scaling_factor: float = 0.0,
    reference_logits_scale: float = 0.0,
    action_seq: Optional[torch.Tensor] = None,
    use_buffer_sample: bool = False,
    buffer_sample: Optional[torch.Tensor] = None,
    buffer_mixture_ratio: float = 0.5,
    disable_grammar: bool = False,
    grammar_disagree_penalty: float = -80,
) -> dict[str, Any]:
    """Roll out the policy and return the quantities required by the GFlowNet objectives.

    At every step the raw logits are masked by the task vocabulary and the grammar processor, a
    token is sampled (or read back from ``action_seq`` when replaying a stored trajectory), and
    the forward and termination log-probabilities are recorded. Tokens rejected by the grammar
    are penalised in the log-probabilities by ``grammar_disagree_penalty`` so that the policy is
    trained against the constrained distribution. Once every trajectory has terminated, the
    completed sequences are scored by ``reward_fn``.

    Args:
        model: Causal language model producing the forward policy.
        encoded_data: Mapping with the batched ``encoded_prompt`` tensor and an optional
            ``scaffold`` passed through to the reward.
        termination_token_id: Token id that terminates a trajectory (EOS).
        reward_fn: Callable returning the reward dictionary for a batch of finished sequences.
        grammar_processor: Logits processor enforcing the task grammar; ``None`` disables masking.
        vocab_nice_mask: Boolean mask marking the task's legal tokens.
        vocab_invalid_mask: Boolean mask marking tokens excluded from sampling.
        illegal_vocab_penalty: Log-space penalty applied by the reward to illegal tokens.
        max_len: Maximum number of generated tokens before termination is forced.
        min_len: Number of leading steps during which termination is suppressed.
        temperature: Sampling temperature for the forward policy.
        reward_temperature: Temperature applied to the reward.
        scaling_factor: Scale of the task score relative to the reference prior.
        reference_logits_scale: Scale of the reference-prior term in the reward.
        action_seq: Stored token ids to replay instead of sampling; ``None`` samples on-policy.
        use_buffer_sample: Whether to splice replay-buffer trajectories into the batch.
        buffer_sample: Replay-buffer token ids used when ``use_buffer_sample`` is set.
        buffer_mixture_ratio: Fraction of the batch replaced by replay-buffer trajectories.
        disable_grammar: Bypass grammar masking while still reporting acceptance.
        grammar_disagree_penalty: Log-space penalty added to grammar-rejected tokens.

    Returns:
        Dictionary with the sampled ``state`` and the per-step tensors ``log_pf``, ``log_pterm``,
        ``log_r`` and ``log_r_unpenalized``, the reference-policy terms ``log_pf_ref`` and
        ``log_pterm_ref``, the per-step grammar acceptance masks ``agree_list``, the decoded
        ``full_tokens``, and the ``validator_dict`` produced by the reward.
    """
    encoded_prompt = encoded_data["encoded_prompt"]
    scaffold = encoded_data.get("scaffold")
    device = encoded_prompt.device

    active_seqs = torch.ones(encoded_prompt.size(0), dtype=torch.bool, device=device)
    prompt_len = encoded_prompt.size(1)
    state = encoded_prompt.clone()
    log_pf: list[torch.Tensor] = []
    log_pterm: list[torch.Tensor] = []
    agree_entries: list[torch.Tensor] = []

    token_ids = state
    past_key_values = None

    if grammar_processor is not None:
        # Tolerate processors that do not expose the incremental-parsing hooks.
        try:
            grammar_processor.reset()
            grammar_processor.set_prompt_length(prompt_len)
        except Exception:
            pass
        logits_processor = grammar_processor
    else:
        logits_processor = LogitsProcessorList([])

    nums_replace = 0
    if use_buffer_sample and buffer_sample is not None:
        nums_replace = max(1, int(encoded_prompt.size(0) * buffer_mixture_ratio))

    for step in range(max_len + 1):
        output = model(input_ids=token_ids, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]

        if action_seq is None:
            scores = logits.clone().detach()

            if vocab_nice_mask is not None:
                scores[:, vocab_invalid_mask] = -torch.inf

            results = logits_processor(state, scores, disable_grammar=disable_grammar)

            if isinstance(results, dict):
                modified_logits = results["masked_logits"]
                acceptance = results["acceptance"]
            else:
                modified_logits = results
                acceptance = torch.ones_like(modified_logits, dtype=torch.bool)
            results = {"masked_logits": modified_logits, "acceptance": acceptance}
            agree_entries.append(acceptance)

            if step < min_len:
                non_eos_only = torch.where(results["acceptance"].sum(dim=1) != 1)[0]
                modified_logits[non_eos_only, termination_token_id] = -torch.inf
            elif step >= max_len:
                mask = torch.ones_like(modified_logits, dtype=torch.bool)
                mask[:, termination_token_id] = False
                modified_logits[mask] = -torch.inf
                modified_logits[:, termination_token_id] = 0

            if (~active_seqs).any():
                inactive = ~active_seqs
                modified_logits[inactive] = -torch.inf
                modified_logits[inactive, termination_token_id] = 0.0

            no_valid = ~torch.isfinite(modified_logits).any(dim=-1)
            if no_valid.any():
                modified_logits[no_valid] = -torch.inf
                modified_logits[no_valid, termination_token_id] = 0.0

            prob = (modified_logits / temperature).softmax(dim=-1)
            token_ids = torch.multinomial(prob, num_samples=1)
            if use_buffer_sample and buffer_sample is not None:
                if step < buffer_sample.size(-1):
                    if step >= max_len:
                        token_ids[:nums_replace, :] = termination_token_id
                    else:
                        token_ids[:nums_replace, :] = buffer_sample[:nums_replace, step].unsqueeze(
                            -1
                        )
                else:
                    token_ids[:nums_replace, :] = termination_token_id
        else:
            idx = prompt_len - 1 + step
            if idx >= action_seq.size(-1):
                token_ids = torch.full(
                    (action_seq.size(0), 1),
                    termination_token_id,
                    device=device,
                    dtype=action_seq.dtype,
                )
            else:
                token_ids = action_seq[:, idx].unsqueeze(-1).to(device)
                # Replaying a stored trajectory: only the grammar acceptance mask is recomputed,
                # since the tokens are fixed the length constraints need not be re-applied.
                scores = logits.clone().detach()
                results = logits_processor(state, scores, disable_grammar=disable_grammar)
                if isinstance(results, dict):
                    modified_logits = results["masked_logits"]
                    acceptance = results["acceptance"]
                else:
                    modified_logits = results
                    acceptance = torch.ones_like(modified_logits, dtype=torch.bool)
                results = {"masked_logits": modified_logits, "acceptance": acceptance}
                agree_entries.append(acceptance)

        inactive_tokens = token_ids.new_full(token_ids.shape, termination_token_id)
        token_ids = torch.where(active_seqs.unsqueeze(-1), token_ids, inactive_tokens)

        if grammar_disagree_penalty != 0 and results["acceptance"].ndim == 2:
            logits[~results["acceptance"]] += grammar_disagree_penalty
        logprob = logits.log_softmax(dim=-1)

        term_scores = logprob[:, termination_token_id]
        log_pterm.append(
            torch.where(active_seqs, term_scores, term_scores.new_zeros(term_scores.shape))
        )
        active_seqs = active_seqs & (token_ids.squeeze(-1) != termination_token_id)

        step_scores = logprob.gather(-1, token_ids).squeeze(-1)
        log_pf.append(
            torch.where(
                active_seqs,
                step_scores,
                step_scores.new_zeros(step_scores.shape),
            )
        )

        state = torch.cat([state, token_ids], dim=-1)

    log_pf = torch.stack(log_pf, dim=1)
    log_pterm = torch.stack(log_pterm, dim=1)

    reward_results = reward_fn(
        state[:, :-1],
        reward_temperature=reward_temperature,
        scaling_factor=scaling_factor,
        reference_logits_scale=reference_logits_scale,
        vocab_invalid_mask=vocab_invalid_mask,
        illegal_vocab_penalty=illegal_vocab_penalty,
        termination_token_id=termination_token_id,
        scaffold=scaffold,
        action_seq=action_seq,
    )

    return {
        "state": state,
        "log_pf": log_pf,
        "log_pterm": log_pterm,
        "log_r": reward_results["reward"],
        "log_r_unpenalized": reward_results["reward_unpenalized"],
        "log_pf_ref": reward_results.get("log_pf_ref"),
        "log_pterm_ref": reward_results.get("log_pterm_ref"),
        "agree_list": agree_entries,
        "full_tokens": reward_results.get("full_tokens"),
        "validator_dict": reward_results.get("validator_dict"),
    }


def get_termination_vals(
    generated_text: torch.Tensor,
    log_pf: Optional[torch.Tensor],
    log_pterm: Optional[torch.Tensor],
    log_r: torch.Tensor,
    log_r_unpenalized: torch.Tensor,
    termination_token_id: int,
    prompt_len: int,
) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read off the per-trajectory quantities at the terminating step.

    Args:
        generated_text: Prompt-and-completion token ids of shape ``[batch, prompt_len + steps]``.
        log_pf: Per-step forward log-probabilities, or ``None`` to skip the policy term.
        log_pterm: Per-step termination log-probabilities, or ``None``.
        log_r: Per-step log rewards.
        log_r_unpenalized: Per-step log rewards before the invalid-token penalty.
        termination_token_id: Token id that terminates a trajectory (EOS).
        prompt_len: Number of prompt tokens preceding the generated part.

    Returns:
        Tuple of ``(log_pfs, log_r, log_r_unpenalized, gen_len)`` evaluated at the EOS position,
        where ``log_pfs`` is the complete trajectory log-probability (``None`` when ``log_pf``
        and ``log_pterm`` are both ``None``) and ``gen_len`` is the generated length.
    """
    batch_idx = torch.arange(generated_text.size(0))
    gen_len = (generated_text[:, prompt_len:] == termination_token_id).byte().argmax(dim=-1)
    if log_pf is None and log_pterm is None:
        log_pfs = None
    else:
        log_pf = torch.cat([torch.zeros_like(log_pf[:, :1]), log_pf], dim=-1)[:, :-1]
        log_pfs = log_pf.cumsum(dim=-1) + log_pterm
        log_pfs = log_pfs[batch_idx, gen_len]
    log_r = log_r[batch_idx, gen_len]
    log_r_unpenalized = log_r_unpenalized[batch_idx, gen_len]
    return log_pfs, log_r, log_r_unpenalized, gen_len
