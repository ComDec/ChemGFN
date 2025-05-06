import gzip
import heapq
import pickle
import random

import editdistance
import numpy as np
import spacy
import torch

# slient the warning
from rdkit import Chem, RDLogger

# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim
from transformers import PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessorList

RDLogger.DisableLog("rdApp.*")


def lora_to_base(model):
    model.base_model.disable_adapter_layers()
    model.eval()


def base_to_lora(model):
    model.base_model.enable_adapter_layers()
    model.train()


def prepare_token_mask(tokenizer: PreTrainedTokenizer, vocab_path: str, reverse: bool = False):
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
    legal_token_mask[tokenizer.bos_token_id] = True
    legal_token_mask[tokenizer.eos_token_id] = True

    illegal_token_mask = ~legal_token_mask

    return legal_token_mask, illegal_token_mask, legal_tokens


def generate_and_return_termination_logprob(
    model,
    encoded_prompt,
    termination_token_id,
    reward_fn,
    grammar_processor=None,
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    naughty_vocab_alpha=float("-inf"),
    invalid_vocab_alpha=-50,
    max_len=10,
    min_len=0,
    temperature=1.0,
    reward_temperature=1.0,
    action_seq=None,
    skip_rewards=False,
    use_buffer_sample=False,
    buffer_sample=None,
    buffer_mixture_ratio=0.5,
):
    # generate and return the probability of terminating at every step
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
        except:
            pass
        logits_processor = LogitsProcessorList([grammar_processor])
    else:
        logits_processor = LogitsProcessorList([])

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
                # 应用 logits processor
                results = logits_processor(state, modified_logits, prompt_length=prompt_len)
                modified_logits = results["masked_logits"]
                agree_list.append(results["acceptance"])
                if i < min_len:
                    # if model generate eos normally but we don't reach the min_len
                    # then we get full nan probability
                    non_eos_only = torch.where(results["acceptance"].sum(dim=1) != 1)[0]
                    modified_logits[non_eos_only, termination_token_id] = -torch.inf

                elif i >= max_len:
                    mask = torch.ones_like(modified_logits, dtype=torch.bool)
                    mask[:, termination_token_id] = False  # EOS token保留
                    modified_logits[mask] = -torch.inf

                prob = (modified_logits / temperature).softmax(dim=-1)
                token_ids = torch.multinomial(prob, num_samples=1)

                # print(f"prob: {prob}\n state: {state}\n token_ids: {token_ids}")
                if use_buffer_sample:
                    # Use efficient sampling methods
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
            if i >= action_seq.size(-1):
                token_ids = (torch.ones_like(action_seq[:, 0]) * termination_token_id).unsqueeze(
                    -1
                )
            else:
                token_ids = action_seq[:, prompt_len - 1 + i].unsqueeze(-1)
                agree_list.append(
                    vocab_nice_mask.unsqueeze(0).repeat(token_ids.size(0), 1).to(token_ids.device)
                )

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

        # check if all sequences have terminated, apply when we need non-fixed length generation
        # if torch.all(~active_seqs):
        #     break

    log_pf = torch.stack(log_pf, dim=1)
    log_pterm = torch.stack(log_pterm, dim=1)

    if skip_rewards:
        log_r, log_r_unpenalized = None, None
    else:
        # Reward for all intermediate states (except the last one,
        # which is guaranteed to be the termination token)
        log_r, log_r_unpenalized = reward_fn(
            state[:, :-1],
            reward_temperature=reward_temperature,
            vocab_nice_mask=vocab_nice_mask,
            vocab_naughty_mask=vocab_naughty_mask,
            naughty_vocab_alpha=naughty_vocab_alpha,
            invalid_vocab_alpha=invalid_vocab_alpha,
            agree_list=torch.stack(agree_list, dim=0),
        )
    # add a termination token to the end of the sequence
    return state, log_pf, log_pterm, log_r, log_r_unpenalized


def modified_subtb_loss(
    log_pf,
    log_r,
    log_pterm,
    generated_text,
    termination_token_id,
    prompt_len,
    subtb_lambda=1.0,
):
    # Ensure the dimensions of log probabilities, rewards, and generated text match
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_text.shape[1] - prompt_len
    )
    # Ensure there is at least one transition before termination
    assert log_pf.shape[1] > 1

    # Calculate the change in expected reward and probability at each step
    delta = log_r[:, :-1] + log_pf[:, :-1] + log_pterm[:, 1:] - log_r[:, 1:] - log_pterm[:, :-1]
    # Compute cumulative sum of delta for subtrajectory balance calculation
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

    # Create a mask for tokens after the termination token
    mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(-1) >= 1
    batch_loss = 0.0
    total_lambda = 0.0
    generated_len = generated_text.shape[1] - prompt_len
    for subtraj_len in range(1, generated_len):
        # Calculate the subtrajectory balance term
        subtb_term = (delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]) ** 2
        # Apply mask to ignore invalid parts of the sequence
        subtb_term[mask[:, subtraj_len - 1 :]] = 0
        # Accumulate weighted subtrajectory balance term
        batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
        # Accumulate total weight for normalization
        total_lambda += subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
    # Normalize the loss by the total weight
    batch_loss /= total_lambda
    return batch_loss


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

    def __init__(self, buffer_size, sim_tolerance=0.25):
        self.buffer_size = buffer_size
        self.sim_tolerance = sim_tolerance
        self.reset()

    def set_termination_token_id(self, termination_token_id):
        self.termination_token_id = termination_token_id

    def reset(self):
        self._buffer = {}

    def add(self, item):
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # if item is already in the buffer, skip it
        str_prompt = item["str_prompt"]
        if item["str_sentence"] in self._buffer[str_prompt]["exists"]:
            return
        # if the edit distance between item and any item in the buffer is small, skip it
        tokenized_sentence = [
            x for x in item["tensor_sentence"].tolist() if x != self.termination_token_id
        ]
        for buffer_item in self._buffer[str_prompt]["sentences"]:
            tokenized_existing_sentence = [
                x for x in buffer_item[2].tolist() if x != self.termination_token_id
            ]
            if (
                editdistance.eval(tokenized_sentence, tokenized_existing_sentence)
                < (len(tokenized_sentence) + len(tokenized_existing_sentence)) * self.sim_tolerance
            ):
                if buffer_item[0] >= item["logreward"]:
                    return
                else:
                    self._buffer[str_prompt]["exists"].remove(buffer_item[1])
                    self._buffer[str_prompt]["sentences"].remove(buffer_item)
                    heapq.heapify(self._buffer[str_prompt]["sentences"])
                    self._buffer[str_prompt]["exists"].add(item["str_sentence"])
                    heapq.heappush(
                        self._buffer[str_prompt]["sentences"],
                        (
                            item["logreward"],
                            item["str_sentence"],
                            item["tensor_sentence"],
                            item["full_logrewards"],
                        ),
                    )
                    return
        self._buffer[str_prompt]["exists"].add(item["str_sentence"])
        if len(self._buffer[str_prompt]["sentences"]) >= self.buffer_size:
            popped = heapq.heappushpop(
                self._buffer[str_prompt]["sentences"],
                (
                    item["logreward"],
                    item["str_sentence"],
                    item["tensor_sentence"],
                    item["full_logrewards"],
                ),
            )
            self._buffer[str_prompt]["exists"].remove(popped[1])
        else:
            heapq.heappush(
                self._buffer[str_prompt]["sentences"],
                (
                    item["logreward"],
                    item["str_sentence"],
                    item["tensor_sentence"],
                    item["full_logrewards"],
                ),
            )

    def add_batch(self, prompt, sentences, logrewards, tokenizer):
        """
        add a batch of items to the buffer
        """
        str_prompt = " ".join([str(x) for x in prompt.tolist()])
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
            self.add(
                {
                    "logreward": logrewards[
                        i, (sentences[i][prompt_len - 1 :] != self.termination_token_id).sum()
                    ].item(),
                    "str_prompt": str_prompt,
                    "str_sentence": str_sentence,
                    "tensor_sentence": sentences[i],
                    "full_logrewards": logrewards[i, :],
                }
            )

    def sample(self, batch_size, prompt):
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        """
        str_prompt = " ".join([str(x) for x in prompt.tolist()])
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

    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["sentences"]:
                print(item[1])
            print("")

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)
