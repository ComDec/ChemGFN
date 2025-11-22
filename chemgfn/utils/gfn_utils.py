import gzip
import heapq
import os
import pickle

import editdistance
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# slient the warning
from rdkit import Chem, RDLogger
from transformers import PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessorList

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


import numpy as np
import torch


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
    target_molecule = encoded_data.get("molecule")
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
            scores = default_processor(state, scores)
            results = logits_processor(state, scores)

            if isinstance(results, dict):
                modified_logits = results["masked_logits"]
                agree_entries.append(results["acceptance"])
            else:
                modified_logits = results
                agree_entries.append(torch.ones_like(modified_logits, dtype=torch.bool))

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
                if vocab_nice_mask is not None:
                    agree_entries.append(
                        vocab_nice_mask.unsqueeze(0).repeat(token_ids.size(0), 1).to(device)
                    )

        inactive_tokens = token_ids.new_full(token_ids.shape, termination_token_id)
        token_ids = torch.where(active_seqs.unsqueeze(-1), token_ids, inactive_tokens)

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
            target_molecule=target_molecule,
        )

        if isinstance(reward_results, dict):
            log_r = reward_results["reward"]
            log_r_unpenalized = reward_results["reward_unpenalized"]
            log_r_reference = reward_results.get("reward_reference", None)
            log_r_target = reward_results.get("reward_target", None)
            log_pf_ref = reward_results.get("log_pf_ref")
            full_tokens = reward_results.get("full_tokens")
            validator_dict = reward_results.get("validator_dict")
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
        "agree_list": agree_entries,
        "log_pf_ref": log_pf_ref,
        "full_tokens": full_tokens,
        "validator_dict": validator_dict,
    }


def generate_and_return_termination_logprob_for_sidechain_opt(
    model,
    encoded_data,
    termination_token_id,
    reward_fn,
    grammar_processor=None,
    vocab_nice_mask=None,  # Unused, kept for compatibility with generate_and_return_termination_logprob
    vocab_invalid_mask=None,
    illegal_vocab_penalty=float("-inf"),
    max_len: int = 10,
    min_len: int = 0,
    temperature: float = 1.0,
    reward_temperature: float = 1.0,
    scaling_factor: float = 50,
    action_seq=None,
    skip_rewards: bool = False,
    use_buffer_sample: bool = False,
    buffer_sample=None,
    buffer_mixture_ratio: float = 0.5,
):
    # generate and return the probability of terminating at every step
    encoded_prompt = encoded_data["encoded_prompt"]
    target_molecule = encoded_data["molecule"] if "molecule" in encoded_data else None
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    prompt_len = encoded_prompt.size(1)
    state = encoded_prompt.clone()
    log_pf = []
    log_pterm = []
    token_ids = state  # For caching hidden states during generation
    past_key_values = None  # For caching hidden states during generation
    if grammar_processor is not None:
        try:
            grammar_processor.reset()
            grammar_processor.set_prompt_length(prompt_len)
        except:
            pass
        logits_processor = grammar_processor
    else:
        logits_processor = LogitsProcessorList([])

    default_processor = LogitsProcessorList([])
    agree_list = []

    # according mixture_ratio, ramdom replace token_ids with buffer_sample
    nums_replace = max(1, int(token_ids.size(0) * buffer_mixture_ratio))

    # main loop
    for i in range(max_len + 1):
        output = model(input_ids=token_ids, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]

        if action_seq is None:
            with torch.no_grad():
                modified_logits = logits.clone().detach()
                if vocab_invalid_mask is not None:
                    modified_logits[:, vocab_invalid_mask] = -torch.inf

                if i >= max_len:
                    mask = torch.ones_like(modified_logits, dtype=torch.bool)
                    mask[:, termination_token_id] = False
                    modified_logits[mask] = -torch.inf
                    # if modified_logits[:, termination_token_id] == -torch.inf, we replace it with 0
                    modified_logits[:, termination_token_id] = 1
                    acceptance = torch.zeros_like(modified_logits, dtype=torch.bool)
                    acceptance[:, termination_token_id] = True
                    agree_list.append(acceptance)
                    token_ids = torch.ones_like(token_ids) * termination_token_id

                else:
                    modified_logits = default_processor(state, modified_logits)
                    # apply logits processor
                    results = logits_processor(state, modified_logits, min_len)
                    modified_logits = results["masked_logits"]
                    agree_list.append(results["acceptance"])

                    # TODO: EOS fetching problem, temperature
                    prob = (modified_logits / temperature).softmax(dim=-1)
                    if torch.any(torch.isnan(prob)):
                        # Replace NaN probabilities with uniform distribution
                        prob = torch.ones_like(prob) / prob.size(-1)
                    token_ids = torch.multinomial(prob, num_samples=1)

                    if use_buffer_sample and buffer_sample is not None:
                        if i < buffer_sample.size(-1):
                            if i >= max_len:
                                token_ids[:nums_replace, :] = termination_token_id
                            else:
                                token_ids[:nums_replace, :] = buffer_sample[
                                    :nums_replace, i
                                ].unsqueeze(-1)
                        else:
                            token_ids[:nums_replace, :] = termination_token_id

        else:
            if i >= max_len:
                token_ids = (torch.ones_like(action_seq[:, 0]) * termination_token_id).unsqueeze(
                    -1
                )
                agree_list.append(
                    torch.zeros_like(logits, dtype=torch.bool)
                    .to(token_ids.device)
                    .scatter_(1, token_ids, True)
                )
            else:
                token_ids = action_seq[:, prompt_len - 1 + i].unsqueeze(-1)
                acceptance = torch.zeros_like(logits, dtype=torch.bool).to(token_ids.device)
                # Set the acceptance for the current token
                acceptance.scatter_(1, token_ids, True)
                agree_list.append(acceptance)

        token_ids = torch.where(
            active_seqs.unsqueeze(-1),
            token_ids,
            termination_token_id,
        )

        logprob = logits.log_softmax(dim=-1)

        # prob list for termination token by steps
        log_pterm.append(
            torch.where(
                active_seqs,
                logprob[:, termination_token_id],
                0,
            )
        )
        active_seqs = active_seqs * (token_ids != termination_token_id).squeeze(-1)

        # prob list for the generated token by steps
        log_pf.append(
            torch.where(
                active_seqs,
                logprob.gather(-1, token_ids).squeeze(-1),
                0,
            )
        )
        # update the state, i.e., the sequence so far
        state = torch.cat([state, token_ids], dim=-1)

    log_pf = torch.stack(log_pf, dim=1)
    log_pterm = torch.stack(log_pterm, dim=1)

    if skip_rewards:
        log_r, log_r_unpenalized = None, None
    else:
        # Reward for all intermediate states (except the last one,
        # which is guaranteed to be the termination token)
        if agree_list is not None:
            agree_list = torch.stack(agree_list, dim=0)
        reward_results = reward_fn(
            state[:, :-1],
            reward_temperature=reward_temperature,
            scaling_factor=scaling_factor,
            vocab_invalid_mask=vocab_invalid_mask,
            illegal_vocab_penalty=illegal_vocab_penalty,
            agree_list=agree_list,
            termination_token_id=termination_token_id,
            target_molecule=target_molecule,
        )
        log_r = reward_results["reward"]
        log_r_unpenalized = reward_results["reward_unpenalized"]
        log_pf_ref = reward_results.get("log_pf_ref", None)
        full_tokens = reward_results.get("full_tokens", None)
        validator_dict = reward_results.get("validator_dict", None)

    return {
        "state": state,
        "log_pf": log_pf,
        "log_pterm": log_pterm,
        "log_r": log_r,
        "log_r_unpenalized": log_r_unpenalized,
        "agree_list": agree_list,
        "log_pf_ref": log_pf_ref,
        "full_tokens": full_tokens,
        "validator_dict": validator_dict,
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


class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """

    def __init__(
        self,
        buffer_size,
        sim_tolerance=0.25,
        strict_mode: bool = False,
        buffer_aug_value: float = 0,
    ):
        self.buffer_size = buffer_size
        self.sim_tolerance = sim_tolerance
        self.strict_mode = strict_mode
        self.buffer_aug_value = buffer_aug_value
        self.reset()

    def set_termination_token_id(self, termination_token_id):
        self.termination_token_id = termination_token_id

    def reset(self):
        self._buffer = {}

    def add(self, item, force_add=False, psudo_reward: float = 0):
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # if item is already in the buffer, skip it
        str_prompt = item["str_prompt"]

        # Hashable string for prompt+answer
        if item["str_sentence"] in self._buffer[str_prompt]["exists"]:
            return

        new_item = (
            psudo_reward,
            item["str_sentence"],
            item["tensor_sentence"],
            item["tensor_answer"],
            item["full_logrewards"],
            force_add,
        )
        buffer = self._buffer[str_prompt]["sentences"]

        for buffer_item in list(buffer):  # Iterate over a copy
            existing_answer = [
                x for x in buffer_item[3].tolist() if x != self.termination_token_id
            ]
            new_answer = [
                x for x in item["tensor_answer"].tolist() if x != self.termination_token_id
            ]
            if (
                editdistance.eval(new_answer, existing_answer)
                < (len(new_answer) + len(existing_answer)) * self.sim_tolerance
            ):
                if buffer_item[0] >= psudo_reward and (not force_add):
                    return
        # Critical fix: Only add to 'exists' AFTER successful heap insertion
        if len(buffer) >= self.buffer_size:
            # Push off the smallest item if buffer is full
            popped = heapq.heappop(buffer)
            # self._buffer[str_prompt]["exists"].remove(popped[1])
            self._buffer[str_prompt]["exists"].add(item["str_sentence"])
        else:
            if force_add:
                new_item = list(new_item)
                # ensure validated items are preferred
                new_item[-2] += self.buffer_aug_value
                new_item = tuple(new_item)
            if self.strict_mode:
                if force_add:
                    heapq.heappush(buffer, new_item)
                    self._buffer[str_prompt]["exists"].add(item["str_sentence"])
                else:
                    return
            else:
                heapq.heappush(buffer, new_item)
                self._buffer[str_prompt]["exists"].add(item["str_sentence"])

    def add_batch(self, prompt, sentences, logrewards, tokenizer, result_dict=None):
        """
        add a batch of items to the buffer
        """
        str_prompt = " ".join(
            [str(x) for x in tokenizer.batch_decode(prompt, skip_special_tokens=True)]
        )
        if str_prompt not in self._buffer:
            self._buffer[str_prompt] = {
                "tensor_prompt": prompt,
                "sentences": [],
                "exists": set(),
            }
        sentences[
            (sentences == self.termination_token_id).cumsum(dim=-1) >= 1
        ] = self.termination_token_id
        token_sentences = tokenizer.batch_decode(sentences)
        prompt_len = prompt.shape[1]

        for i in range(sentences.size(0)):
            # str_sentence = token_sentences[i].replace(".", "").strip()
            # there is no such termination token in the SMILES
            str_sentence = token_sentences[i].strip()
            valid_state = (result_dict["validator_dict"]["global_score"].bool())[i]
            self.add(
                {
                    "logreward": logrewards[
                        i, (sentences[i][prompt_len - 1 :] != self.termination_token_id).sum()
                    ].item(),
                    "str_prompt": str_prompt,
                    "str_sentence": str_sentence,
                    "tensor_answer": sentences[i][prompt_len - 1 :],
                    "tensor_sentence": sentences[i],
                    "full_logrewards": logrewards[i, :],
                },
                force_add=valid_state,
                psudo_reward=result_dict["validator_dict"]["global_score"][i].item(),
            )

    def sample(self, batch_size, prompt, tokenizer):
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        """
        str_prompt = " ".join(
            [str(x) for x in tokenizer.batch_decode(prompt, skip_special_tokens=True)]
        )
        if str_prompt not in self._buffer:
            return None, None
        if len(self._buffer[str_prompt]["sentences"]) < batch_size:
            return None, None
        prompt_buffer = self._buffer[str_prompt]["sentences"]
        idx = np.random.choice(
            len(prompt_buffer),
            batch_size,
            replace=True,
        )
        return torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][2] for i in idx],
            batch_first=True,
            padding_value=self.termination_token_id,
        ), torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][3] for i in idx],
            batch_first=True,
            padding_value=0,
        )

    def stat(self):
        """
        statistics of the buffer
        """
        stats = {}
        for idx, key in enumerate(self._buffer):
            total_buffer = len(self._buffer[key]["sentences"])
            avg_logR = sum([item[0] for item in self._buffer[key]["sentences"]])
            stats.update(
                {
                    f"prompt_{idx}_total_buffer": total_buffer,
                    f"prompt_{idx}_avg_logR": avg_logR / total_buffer if total_buffer > 0 else 0,
                }
            )
        return stats

    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["sentences"]:
                print(item[1])
            print("")

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)

    def save_csv(self, path, tokenizer):
        """
        Save the buffer to a CSV file.
        Each row contains: str_prompt, str_sentence, logreward, tensor_sentence, tensor_answer
        """
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("logr_sum,tensor_answer,answer_logR,answer,force_add\n")
            for key in self._buffer:
                for item in self._buffer[key]["sentences"]:
                    answer_tokens = item[2].tolist()
                    force_add = item[-1].item()
                    f.write(
                        f"{item[0]},{answer_tokens},{item[-2].tolist()},{item[1]},{force_add}\n"
                    )


class ReplayBufferNative(ReplayBuffer):
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """

    def __init__(self, buffer_size, sim_tolerance=0.25):
        self.buffer_size = buffer_size
        self.sim_tolerance = sim_tolerance
        self.reset()

    def set_termination_token_id(self, termination_token_id):
        self.termination_token_id = termination_token_id

    def reset(self):
        self._buffer = {}

    def add(self, item, force_add=False, psudo_reward: float = 0):
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # if item is already in the buffer, skip it
        str_prompt = item["str_prompt"]

        # Hashable string for prompt+answer
        if item["str_sentence"] in self._buffer[str_prompt]["exists"]:
            return

        new_item = (
            item["logreward"],
            item["str_sentence"],
            item["tensor_sentence"],
            item["tensor_answer"],
            item["full_logrewards"],
            force_add,
        )
        buffer = self._buffer[str_prompt]["sentences"]

        for buffer_item in list(buffer):  # Iterate over a copy
            existing_answer = [
                x for x in buffer_item[3].tolist() if x != self.termination_token_id
            ]
            new_answer = [
                x for x in item["tensor_answer"].tolist() if x != self.termination_token_id
            ]
            if (
                editdistance.eval(new_answer, existing_answer)
                < (len(new_answer) + len(existing_answer)) * self.sim_tolerance
            ):
                if buffer_item[0] >= item["logreward"] and not force_add:
                    return

        # Critical fix: Only add to 'exists' AFTER successful heap insertion
        if len(buffer) >= self.buffer_size:
            # Push off the smallest item if buffer is full
            popped = heapq.heappop(buffer)
            # self._buffer[str_prompt]["exists"].remove(popped[1])
            self._buffer[str_prompt]["exists"].add(item["str_sentence"])
        else:
            heapq.heappush(buffer, new_item)
            self._buffer[str_prompt]["exists"].add(item["str_sentence"])

    def add_batch(self, prompt, sentences, logrewards, tokenizer, result_dict=None):
        """
        add a batch of items to the buffer
        """
        str_prompt = " ".join(
            [str(x) for x in tokenizer.batch_decode(prompt, skip_special_tokens=True)]
        )
        if str_prompt not in self._buffer:
            self._buffer[str_prompt] = {
                "tensor_prompt": prompt,
                "sentences": [],
                "exists": set(),
            }
        sentences[
            (sentences == self.termination_token_id).cumsum(dim=-1) >= 1
        ] = self.termination_token_id
        token_sentences = tokenizer.batch_decode(sentences)
        prompt_len = prompt.shape[1]

        for i in range(sentences.size(0)):
            # str_sentence = token_sentences[i].replace(".", "").strip()
            # there is no such termination token in the SMILES
            str_sentence = token_sentences[i].strip()
            batch_invalid = result_dict["validator_dict"]["invalid"]
            valid_state = (~batch_invalid.bool())[i][-1]
            self.add(
                {
                    "logreward": logrewards[
                        i, (sentences[i][prompt_len - 1 :] != self.termination_token_id).sum()
                    ].item(),
                    "str_prompt": str_prompt,
                    "str_sentence": str_sentence,
                    "tensor_answer": sentences[i][prompt_len - 1 :],
                    "tensor_sentence": sentences[i],
                    "full_logrewards": logrewards[i, :],
                },
                force_add=valid_state,
                psudo_reward=result_dict["validator_dict"]["global_score"][i].item(),
            )

    def sample(self, batch_size, prompt, tokenizer):
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        """
        str_prompt = " ".join(
            [str(x) for x in tokenizer.batch_decode(prompt, skip_special_tokens=True)]
        )
        if str_prompt not in self._buffer:
            return None, None
        prompt_buffer = self._buffer[str_prompt]["sentences"]
        idx = np.random.choice(
            len(prompt_buffer),
            batch_size,
            replace=True,
        )
        return torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][2] for i in idx],
            batch_first=True,
            padding_value=self.termination_token_id,
        ), torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][3] for i in idx],
            batch_first=True,
            padding_value=0,
        )

    def stat(self):
        """
        statistics of the buffer
        """
        stats = {}
        for idx, key in enumerate(self._buffer):
            total_buffer = len(self._buffer[key]["sentences"])
            avg_logR = sum([item[0] for item in self._buffer[key]["sentences"]])
            stats.update(
                {
                    f"prompt_{idx}_total_buffer": total_buffer,
                    f"prompt_{idx}_avg_logR": avg_logR / total_buffer if total_buffer > 0 else 0,
                }
            )
        return stats

    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["sentences"]:
                print(item[1])
            print("")

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)

    def save_csv(self, path, tokenizer):
        """
        Save the buffer to a CSV file.
        Each row contains: str_prompt, str_sentence, logreward, tensor_sentence, tensor_answer
        """
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("logr_sum,tensor_answer,answer_logR,answer,force_add\n")
            for key in self._buffer:
                for item in self._buffer[key]["sentences"]:
                    answer_tokens = item[-3].tolist()
                    answer = "".join(
                        [
                            str(x)
                            for x in tokenizer.batch_decode(
                                answer_tokens, skip_special_tokens=False
                            )
                        ]
                    )
                    force_add = item[-1].item()
                    f.write(
                        f"{item[0]},{answer_tokens},{item[-2].tolist()},{answer},{force_add}\n"
                    )
