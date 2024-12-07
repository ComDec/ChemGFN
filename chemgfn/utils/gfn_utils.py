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
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import PreTrainedTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

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

    illegal_token_mask = torch.ones(len(tokenizer), dtype=torch.bool)

    # tokenize illegal tokens, leave numbers as they are
    illegal_tokens = [
        [t] if isinstance(t, int) else tokenizer.encode(t, add_special_tokens=False)
        for t in illegal_tokens
    ]
    assert all(len(t) == 1 for t in illegal_tokens)

    # get inx of illegal tokens
    illegal_tokens = [t[0] for t in illegal_tokens]
    illegal_token_mask[illegal_tokens] = True
    illegal_token_mask = illegal_token_mask.numpy()
    illegal_token_mask[legal_tokens] = False

    # add bos and eos as legal tokens
    illegal_token_mask[tokenizer.bos_token_id] = False
    illegal_token_mask[tokenizer.eos_token_id] = False

    return illegal_token_mask


def generate_and_return_termination_logprob(
    model,
    encoded_prompt,
    termination_token_id,
    reward_fn,
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    vocab_alpha=-99,
    max_len=10,
    min_len=0,
    temperature=1.0,
    top_k=999999,
    top_p=1.0,
    action_seq=None,
    skip_rewards=False,
):
    # generate and return the probability of terminating at every step
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    state = encoded_prompt.clone()
    log_pf = []
    log_pterm = []
    token_ids = state  # For caching hidden states during generation
    past_key_values = None  # For caching hidden states during generation

    logits_processor = LogitsProcessorList(
        [
            TopKLogitsWarper(top_k=top_k),
            TopPLogitsWarper(top_p=top_p),
        ]
    )
    # use only when apply grammar constraints
    # logits_processor[0].reset()

    temperature_processor = TemperatureLogitsWarper(temperature=temperature)

    last_token_ids = token_ids[0, -1]
    i = 0

    while last_token_ids.item() != termination_token_id:
        if i >= max_len:
            break

        output = model(input_ids=token_ids, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]

        if action_seq is None:
            with torch.no_grad():
                # set the probability of illegal tokens to 0, except for the termination token and bos token
                # prob[:, np.where(vocab_naughty_mask == True)] = 0
                # modified logits is used for fullfilling top-k, top-p, and length constraints

                # repalce native top-k and top-p with the LogitsProcessorList
                modified_logits = logits.clone().detach()
                modified_logits = logits_processor(state, modified_logits)

                # modified_logits = logits_wrapper(token_ids, modified_logits)

                # implement top-k by getting the top-k largest values and setting the rest to 0
                # if top_k < 999999:
                #     modified_logits[prob >= prob.topk(top_k)] = -torch.inf
                # # implement top-p by getting indices in the top-p prob mass and setting the rest to 0
                # if top_p < 1.0:
                #     prob[vocab_naughty_mask] = -torch.inf
                #     sorted_probs, _ = torch.sort(prob, dim=-1, descending=True)
                #     cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
                #     nucleus = cumsum_prob < top_p
                #     nucleus = torch.cat(
                #         [
                #             nucleus.new_ones(nucleus.shape[:-1] + (1,)),
                #             nucleus[..., :-1],
                #         ],
                #         dim=-1,
                #     )
                #     modified_logits[~nucleus] = -torch.inf
                # TODO: actually you can make this process smoother, create a func which increase the prob of the termination token gradually, according to the length of the sequence
                # TODO: I don't think this is necessary, this way to control seq_len is not appropriate for the SMILES generation task
                # if i < min_len:
                #     # if we haven't reach the minimum length, set the probability of terminating to 0
                #     modified_logits[:, termination_token_id] = -torch.inf
                # elif i >= max_len:
                #     # if we've reached the maximum length, set the probability of terminating to 1
                #     mask = [True] * modified_logits.shape[1]
                #     mask[termination_token_id] = False
                #     modified_logits[:, mask] = -torch.inf

                # temp remove the nice/naughty token mask

                # if vocab_nice_mask is not None:
                #     # add vocab_alpha to the logits of the unmasked vocab items
                #     modified_logits[:, ~vocab_nice_mask] += vocab_alpha

                if vocab_naughty_mask is not None:
                    # add vocab_alpha to the logits of the masked vocab items
                    modified_logits[:, vocab_naughty_mask] += vocab_alpha

                prob = temperature_processor(state, modified_logits)

                # replace with HF temperature processor
                # prob = (modified_logits / temperature).softmax(dim=-1)

                token_ids = torch.multinomial(prob, num_samples=1)
                last_token_ids = token_ids[0]

        else:
            if i >= action_seq.size(-1):
                token_ids = (torch.ones_like(action_seq[:, 0]) * termination_token_id).unsqueeze(
                    -1
                )
            else:
                token_ids = action_seq[:, i].unsqueeze(-1)
            last_token_ids = token_ids[:, -1]

        i += 1

        token_ids = torch.where(
            active_seqs.unsqueeze(-1),
            token_ids,
            termination_token_id,
        )
        if vocab_nice_mask is not None:
            logits[:, ~vocab_nice_mask] += vocab_alpha
        if vocab_naughty_mask is not None:
            logits[:, vocab_naughty_mask] += vocab_alpha

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
        # check if all sequences have terminated
        if torch.all(~active_seqs):
            break

    log_pf = torch.stack(log_pf, dim=1)
    log_pterm = torch.stack(log_pterm, dim=1)
    if skip_rewards:
        log_r, log_r_unpenalized = None, None
    else:
        # Reward for all intermediate states (except the last one,
        # which is guaranteed to be the termination token)
        log_r, log_r_unpenalized = reward_fn(state[:, :-1])
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
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_text.shape[1] - prompt_len
    )
    assert (
        log_pf.shape[1] > 1
    )  # With modified-style losses, we need at least one transition before terminating

    delta = log_r[:, :-1] + log_pf[:, :-1] + log_pterm[:, 1:] - log_r[:, 1:] - log_pterm[:, :-1]
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

    # Get a mask for tokens after the termination token in the generated_text
    mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(-1) >= 1
    batch_loss = 0.0
    total_lambda = 0.0
    generated_len = generated_text.shape[1] - prompt_len
    for subtraj_len in range(1, generated_len):
        subtb_term = (delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]) ** 2
        subtb_term[mask[:, subtraj_len - 1 :]] = 0
        batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
        total_lambda += subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
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


class SequenceDiversity:
    def __init__(self, method, **kwargs):
        self.method = method
        if method is None:
            pass
        elif method == "sequence_embedding":
            model_name = kwargs.get("model_name", "sentence-transformers/all-mpnet-base-v2")
            self.model = SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unknown sequence diversity method: {method}")

    @torch.no_grad()
    def __call__(self, sequences):
        if self.method is None:
            return None
        elif self.method == "sequence_embedding":
            embeddings = self.model.encode(sequences, show_progress_bar=False)
            sim = cos_sim(embeddings, embeddings)
            indices = torch.triu_indices(len(sequences), len(sequences), offset=1)
            diversity = 1 - sim[indices[0], indices[1]].mean().item()
        else:
            raise ValueError(f"Unknown sequence diversity method: {self.method}")
        return diversity


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

        for i in range(sentences.size(0)):
            # str_sentence = token_sentences[i].replace(".", "").strip()
            # there is no such termination token in the SMILES
            str_sentence = token_sentences[i].strip()
            self.add(
                {
                    "logreward": logrewards[
                        i, (sentences[i] != self.termination_token_id).sum()
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
