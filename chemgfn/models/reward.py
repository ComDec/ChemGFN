import random
from typing import Literal

import numpy as np
import torch

# slient the warning
from rdkit import Chem, RDLogger
from transformers import PreTrainedTokenizer

from chemgfn.utils.gfn_utils import base_to_lora, lora_to_base
from chemgfn.utils.register import ClassRegistry

RDLogger.DisableLog("rdApp.*")


@torch.no_grad()
def score_fast(
    model,
    encoded_input,
    termination_token_id,
    min_len,
    skip_first,
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    vocab_alpha=-99,
    prompt_cache=None,
):
    if prompt_cache is None:
        logits = model(encoded_input).logits
    else:
        # prompt_cache[1] contains past_key_values which need to be reshaped to the right batch size from encoded_input
        batched_prompt_cache = tuple(
            tuple(
                [
                    prompt_cache[1][i][j].repeat(encoded_input.shape[0], 1, 1, 1)
                    for j in range(len(prompt_cache[1][i]))
                ]
            )
            for i in range(len(prompt_cache[1]))
        )
        logits = model(encoded_input, past_key_values=batched_prompt_cache).logits
    # get rid of the first few tokens
    # I didn't see the necessity, maybe to ensure the initial state have a penalty
    logits = logits[:, skip_first - 1 :]

    # score the log probability of the input sequence while ignoring termination and padding tokens
    if vocab_nice_mask is not None:
        # add vocab_alpha to the logits of the unmasked vocab items
        logits[:, :, ~vocab_nice_mask] += vocab_alpha
    elif vocab_naughty_mask is not None:
        # add vocab_alpha to the logits of the masked vocab items
        logits[:, :, vocab_naughty_mask] += vocab_alpha

    logprob = logits.log_softmax(-1)
    token_ids = encoded_input[:, skip_first:].unsqueeze(-1)
    # catch the log probability of each token in the input sequence
    logPF = logprob[:, :-1].gather(-1, token_ids).squeeze(-1)

    logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)

    reward = logprob[
        :, :, termination_token_id
    ]  # logP(generated[i+1]=term | prompt + generated[:i+1])
    reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)
    non_term_mask = (encoded_input != termination_token_id)[:, skip_first:]
    # repeat for sampled size
    non_term_mask = torch.cat(
        (
            non_term_mask.new_ones(non_term_mask.shape[0], 1),
            non_term_mask,
        ),
        dim=-1,
    )  # Start (i.e., empty) state has never terminated
    reward[~non_term_mask] = 0.0
    reward_unpenalized = reward.clone()
    reward = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_len, -99, reward)
    return reward, reward_unpenalized


class SentenceValidator:
    def __init__(self, termination_token_id: int = -1) -> None:
        self.termination_token_id = termination_token_id

    def __call__(self, sentences, tokenizer):
        pass


class RDKitValidator(SentenceValidator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, sentences, tokenizer: PreTrainedTokenizer):
        termination_token_id = tokenizer.eos_token_id

        valid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            device=sentences.device,
        )
        valid[:, 0] = 1  # Empty sentence is never valid

        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                if sentences[i, j] == termination_token_id:
                    break  # Only unterminated sentences get a reward
                tokens = "".join(tokenizer.decode(sentences[i, : j + 1]).split())
                mol = Chem.MolFromSmiles(tokens)
                if mol:
                    # if recover a SMILES from invalid way, give a high reward
                    if valid[i, j] < 10:
                        # TODO: this is a hard-coded value, should be changed
                        valid[i, j + 1] = 10
                    else:
                        valid[i, j + 1] = 5
                else:
                    valid[i, j + 1] = 1 + random.random()
        return valid


class FrozenModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_validator: SentenceValidator,
        temperature=1.0,
        min_len=1,
        vocab_alpha=-50.0,
        vocab_nice_mask=None,
        vocab_naughty_mask=None,
        valid_sentence_alpha=None,
    ):
        assert (
            sentence_validator is None
            and valid_sentence_alpha is None
            or sentence_validator is not None
            and valid_sentence_alpha is not None
        )

        self.temperature = temperature
        self.vocab_nice_mask = vocab_nice_mask
        self.vocab_naughty_mask = vocab_naughty_mask
        self.vocab_alpha = vocab_alpha
        self.min_len = min_len
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha

    def score(self, input_batch, prompt_length, model, tokenizer: PreTrainedTokenizer):
        lora_to_base(model)
        training = model.training
        model.eval()
        reward, reward_unpenalized = score_fast(
            model=model,
            encoded_input=input_batch,
            termination_token_id=tokenizer.eos_token_id,
            skip_first=prompt_length,
            vocab_nice_mask=self.vocab_nice_mask,
            vocab_naughty_mask=self.vocab_naughty_mask,
            vocab_alpha=self.vocab_alpha,
            min_len=self.min_len,
        )
        reward /= self.temperature
        reward_unpenalized /= self.temperature
        base_to_lora(model)
        if training:
            model.train()

        if self.sentence_validator is not None:
            # use valid list instead of invalid list
            valid = self.sentence_validator(input_batch[:, prompt_length:], tokenizer)
            valid = valid * self.valid_sentence_alpha
            # TODO: max or mean
            reward = torch.max(reward, valid)

        return reward, reward_unpenalized
