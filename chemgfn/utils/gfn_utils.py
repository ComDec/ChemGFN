import importlib
import warnings

import numpy as np
import torch

# slient the warning
from rdkit import Chem, RDLogger
from transformers import PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessorList

# Backwards compatibility for legacy ReplayBuffer import path used in old configs/checkpoints.
_LEGACY_REPLAY_BUFFER_ATTRS = {
    "ReplayBuffer",
    "ReplayBufferNative",
    "ReplayBufferSubmodular",
    "ReplayBufferSubmodularV1",
}


def __getattr__(name):
    if name in _LEGACY_REPLAY_BUFFER_ATTRS:
        warnings.warn(
            f"{name} has moved to chemgfn.utils.replay_buffer; please update imports/configs.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = importlib.import_module("chemgfn.utils.replay_buffer")
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


RDLogger.DisableLog("rdApp.*")


def lora_to_base(model):
    """Disable LoRA adapters and set model to eval mode for base model inference.

    Args:
        model: PEFT model with LoRA adapters.
    """
    model.base_model.disable_adapter_layers()
    model.eval()


def base_to_lora(model):
    """Enable LoRA adapters and set model to train mode.

    Args:
        model: PEFT model with LoRA adapters.
    """
    model.base_model.enable_adapter_layers()
    model.train()


def prepare_token_mask(tokenizer: PreTrainedTokenizer, vocab_path: str, reverse: bool = False):
    """Prepare token masks for legal and illegal vocabulary tokens.

    Args:
        tokenizer: Pre-trained tokenizer instance.
        vocab_path: Path to file containing legal tokens (one per line).
        reverse: If True, reverse the legal/illegal masks (default: False).

    Returns:
        Tuple of (legal_token_mask, illegal_token_mask, legal_token_ids_list):
            - legal_token_mask: Boolean tensor of shape [vocab_size] marking legal tokens
            - illegal_token_mask: Boolean tensor of shape [vocab_size] marking illegal tokens
            - legal_token_ids_list: List of legal token IDs
    """
    with open(vocab_path) as f:
        legal_tokens = f.readlines()

    legal_tokens = [line.rstrip("\n") for line in legal_tokens]
    legal_tokens = [tokenizer.encode(t, add_special_tokens=False)[0] for t in legal_tokens]

    legal_token_mask = torch.zeros(len(tokenizer), dtype=torch.bool)

    # tokenize legal tokens, leave numbers as they are
    legal_tokens = [
        [t] if isinstance(t, int) else tokenizer.encode(t, add_special_tokens=False)
        for t in legal_tokens
    ]
    assert all(len(t) == 1 for t in legal_tokens)

    # get inx of legal tokens
    legal_tokens = [t[0] for t in legal_tokens]
    legal_token_mask[legal_tokens] = True

    # add bos and eos as legal tokens
    legal_token_mask[tokenizer.bos_token_id] = False
    legal_token_mask[tokenizer.eos_token_id] = True

    illegal_token_mask = ~legal_token_mask

    return legal_token_mask, illegal_token_mask, legal_tokens


def calculate_diversity(token_id_list):
    """Calculate diversity of LLM sampling results using average per-position entropy.

    Diversity is measured as the average entropy across all sequence positions.
    Higher entropy indicates more diverse samples.

    Args:
        token_id_list: torch.Tensor of shape (num_samples, seq_len) containing token IDs

    Returns:
        float: Average entropy across all sequence positions (higher = more diverse)
    """
    # Convert to tensor and validate dimensions
    num_samples, seq_len = token_id_list.shape

    if num_samples == 1:
        return 0.0  # Only one sample = zero diversity

    total_entropy = 0.0

    for pos in range(seq_len):
        # Get token distribution at current position
        tokens = token_id_list[:, pos]
        unique_tokens, counts = torch.unique(tokens, return_counts=True)
        probs = counts.float() / num_samples

        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))  # Add epsilon to avoid log(0)
        total_entropy += entropy.item()

    return total_entropy / seq_len


def calculate_diversity_by_length(token_id_list, eos_id: int) -> dict[int, float]:
    """Calculate diversity grouped by length (before EOS).

    Args:
        token_id_list: Tensor or list of token ID sequences.
        eos_id: Token ID that marks termination.

    Returns:
        Dict mapping length -> diversity for that length bucket.
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


def _stack_if_not_empty(entries):
    tensors = [entry for entry in entries if entry is not None]
    if not tensors:
        return None
    return torch.stack(tensors, dim=0)


def generate_and_return_termination_logprob(
    model,
    encoded_data,
    termination_token_id,
    reward_fn,
    grammar_processor=None,
    vocab_nice_mask=None,
    vocab_invalid_mask=None,
    illegal_vocab_penalty=float("-inf"),
    max_len=10,
    min_len=0,
    temperature=1.0,
    reward_temperature=1.0,
    action_seq=None,
    skip_rewards=False,
    use_buffer_sample=False,
    buffer_sample=None,
    buffer_mixture_ratio=0.5,
    disable_grammar: bool = False,
    grammar_disagree_penalty=-80,
    **kwargs,
):
    """Generate sequences using the model and compute termination log probabilities.

    This function performs autoregressive generation with grammar constraints and computes
    forward policy log probabilities and termination probabilities at each step.

    Args:
        model: The language model to use for generation.
        encoded_data: Dictionary containing 'encoded_prompt' tensor and optionally 'molecule'.
        termination_token_id: Token ID that marks sequence termination (EOS).
        reward_fn: Function to compute rewards for generated sequences.
        grammar_processor: Optional grammar constraint processor for logits.
        vocab_nice_mask: Optional mask for allowed vocabulary tokens.
        vocab_invalid_mask: Optional mask for disallowed vocabulary tokens.
        illegal_vocab_penalty: Penalty value for illegal tokens (default: -inf).
        max_len: Maximum generation length (default: 10).
        min_len: Minimum generation length before allowing termination (default: 0).
        temperature: Sampling temperature for token selection (default: 1.0).
        reward_temperature: Temperature for reward computation (default: 1.0).
        action_seq: Optional pre-computed action sequence (for buffer sampling).
        skip_rewards: If True, skip reward computation (default: False).
        use_buffer_sample: Whether to use buffer samples for generation (default: False).
        buffer_sample: Optional buffer samples tensor.
        buffer_mixture_ratio: Ratio of samples to replace with buffer (default: 0.5).
        **kwargs: Additional arguments passed to reward_fn.

    Returns:
        Dictionary containing:
            - state: Generated sequences tensor [batch_size, seq_len]
            - log_pf: Forward policy log probabilities [batch_size, max_len]
            - log_pterm: Termination log probabilities [batch_size, max_len]
            - log_r: Reward log probabilities [batch_size, max_len]
            - log_r_unpenalized: Unpenalized reward log probabilities [batch_size, max_len]
            - agree_list: List of agreement tensors from grammar processor
            - log_pf_ref: Reference forward probabilities (if available)
            - full_tokens: Decoded token strings (if available)
            - validator_dict: Validator output dictionary (if available)
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
        try:
            grammar_processor.reset()
            grammar_processor.set_prompt_length(prompt_len)
        except Exception:
            pass
        logits_processor = grammar_processor
    else:
        logits_processor = LogitsProcessorList([])

    default_processor = LogitsProcessorList([])

    nums_replace = 0
    if use_buffer_sample and buffer_sample is not None:
        nums_replace = max(1, int(encoded_prompt.size(0) * buffer_mixture_ratio))

    for step in range(max_len + 1):
        output = model(input_ids=token_ids, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]

        if action_seq is None:
            scores = logits.clone().detach()
            # apply nice_vocab_mask if possible

            if vocab_nice_mask is not None:
                scores[:, vocab_invalid_mask] = -torch.inf

            scores = default_processor(state, scores)
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
                # TODO: simple mask, no-eos before max_len;
                scores = logits.clone().detach()
                scores = default_processor(state, scores)
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

    log_r = None
    log_r_unpenalized = None
    log_pf_ref = None
    full_tokens = None
    validator_dict = None
    phi_diag = None
    phi_state = None
    phi_tok = None
    pv = None

    if not skip_rewards:
        agree_tensor = _stack_if_not_empty(agree_entries)
        reward_results = reward_fn(
            state[:, :-1],
            reward_temperature=reward_temperature,
            scaling_factor=kwargs.get("scaling_factor", 0.0),
            reference_logits_scale=kwargs.get("reference_logits_scale", 0.0),
            vocab_invalid_mask=vocab_invalid_mask,
            illegal_vocab_penalty=illegal_vocab_penalty,
            agree_list=agree_tensor,
            termination_token_id=termination_token_id,
            scaffold=scaffold,
            action_seq=action_seq,
        )

        if isinstance(reward_results, dict):
            # tensors
            log_r = reward_results["reward"]
            log_r_unpenalized = reward_results["reward_unpenalized"]
            log_pf_ref = reward_results.get("log_pf_ref")
            log_pterm_ref = reward_results.get("log_pterm_ref")

            # auxiliary information
            full_tokens = reward_results.get("full_tokens")
            validator_dict = reward_results.get("validator_dict")

            # if you use split reward and loss
            log_r_reference = reward_results.get("reward_reference", None)
            log_r_target = reward_results.get("reward_target", None)

            # extra phi information
            phi_diag = reward_results.get("prefix_diag", None)
            phi_state = reward_results.get("phi_state", None)
            phi_tok = reward_results.get("phi_tok", None)
            phi_weight = reward_results.get("phi_weight", None)
            pv = reward_results.get("pv", None)

        else:
            log_r, log_r_unpenalized = reward_results

    return {
        "state": state,
        "log_pf": log_pf,
        "log_pterm": log_pterm,
        "log_r": log_r,
        "log_r_unpenalized": log_r_unpenalized,
        "log_r_reference": log_r_reference,
        "log_r_target": log_r_target,
        "log_pf_ref": log_pf_ref,
        "log_pterm_ref": log_pterm_ref,
        "agree_list": agree_entries,
        "full_tokens": full_tokens,
        "validator_dict": validator_dict,
        "phi_diag": phi_diag,
        "prefix_diag": phi_diag,
        "phi_state": phi_state,
        "phi_tok": phi_tok,
        "pv": pv,
        "phi_weight": phi_weight,
    }


def get_termination_vals(
    generated_text,
    log_pf,
    log_pterm,
    log_r,
    log_r_unpenalized,
    termination_token_id,
    prompt_len,
):
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
