import random
from typing import Literal

import numpy as np
import torch

# slient the warning
from rdkit import Chem, RDLogger
from transformers import PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessorList

from chemgfn.utils.gfn_utils import base_to_lora, lora_to_base

RDLogger.DisableLog("rdApp.*")


def score_fast(
    model,
    encoded_input,
    termination_token_id,
    skip_first,
    reward_temperature=1.0,
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    naughty_vocab_alpha=-99,
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

    # predict the logits of next tokens
    # the input tokens remove the last token: state[:-1], so the logits is for state[1:]
    logits = logits.detach()
    logits = logits[:, skip_first - 1 :]

    # score the log probability of the input sequence while ignoring termination and padding tokens
    if vocab_nice_mask is not None:
        # add vocab_alpha to the logits of the unmasked vocab items
        logits[:, :, ~vocab_nice_mask] += naughty_vocab_alpha
    elif vocab_naughty_mask is not None:
        # add vocab_alpha to the logits of the masked vocab items
        logits[:, :, vocab_naughty_mask] += naughty_vocab_alpha

    # add reward temperature
    logits /= reward_temperature
    logprob = logits.log_softmax(-1)
    token_ids = encoded_input[:, skip_first:].unsqueeze(-1)

    # # catch the log probability of each token in the input sequence
    # logPF = logprob[:, :-1].gather(-1, token_ids).squeeze(-1)

    # strategy 0: replace -inf with P(eos), random sample token_is'prob
    # batch_size, seq_length, vocab_size = logprob.size()
    # updated_logprob = logprob.clone()

    # for i in range(batch_size):
    #     for j in range(seq_length - 1):
    #         current_logprob = logprob[i, j]
    #         if torch.isinf(current_logprob).any():
    #             valid_indices = torch.where(~torch.isinf(current_logprob))[0]
    #             sampled_index = valid_indices[torch.multinomial(torch.ones(len(valid_indices)), 1)]
    #             updated_logprob[i, j, token_ids[i, j]] = current_logprob[sampled_index]

    # # 计算logPF
    # logPF = updated_logprob[:, :-1].gather(-1, token_ids).squeeze(-1)

    # strategy1: repalce -inf with P(eos), add temperature control
    # updated_logprob = logprob.clone()

    # # Replace -inf values with termination token probability
    # inf_mask = torch.isinf(updated_logprob)
    # if inf_mask.any():
    #     termination_probs = logprob[:, :, termination_token_id].unsqueeze(-1)  # Get termination token probabilities
    #     updated_logprob[inf_mask] = termination_probs.expand_as(updated_logprob)[inf_mask]

    # # Calculate logPF using updated probabilities
    # logPF = updated_logprob[:, :-1].gather(-1, token_ids).squeeze(-1)

    logPF = logprob[:, :-1].gather(-1, token_ids).squeeze(-1)
    logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)

    reward = logprob[
        :, :, termination_token_id
    ]  # logP(generated[i+1]=term | prompt + generated[:i+1])

    reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)

    ### all the shift above is wired

    non_term_mask = (encoded_input != termination_token_id)[:, skip_first:]

    non_term_mask = torch.cat(
        (
            non_term_mask.new_ones(non_term_mask.shape[0], 1),
            non_term_mask,
        ),
        dim=-1,
    )  # Start (i.e., empty) state has never terminated

    reward[~non_term_mask] = 0.0
    reward_unpenalized = reward.clone()

    # reward = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_len, -99, reward)
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


def number_reward(sampled_numbers):
    # check if the list of numbers follow the rule: even number followed by odd number and odd number followed by even number and between 0 and 20.
    # if any number larger than 20, return 0, else return 1.
    # if number follow the rule, return 2, else return 1.
    numbers = [int(number) for number in sampled_numbers]
    if not all((numbers[i] % 2 != numbers[i + 1] % 2) for i in range(len(numbers) - 1)):
        return 0
    else:
        return 1


def validate_brackets(s: torch.Tensor, tokenizer: PreTrainedTokenizer) -> bool:
    stack = []
    pairs = {")": "(", "]": "[", ">": "<"}

    decoded_list = []
    # remove the eos token
    for t in s:
        if t.item() != tokenizer.eos_token_id:
            decoded_list.append(tokenizer.decode(t))
        else:
            break

    # think of "<<" due to bpe, we need to re-concatenate the tokens
    decoded_string = "".join(decoded_list)
    for char in list(decoded_string):
        if char in pairs.values():
            stack.append(char)
        elif char in pairs.keys():
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    return not stack


def number_accuracy(generated_tokens, tokenizer):
    numbers_list = []

    for batch in range(generated_tokens.shape[0]):
        number = []
        for t in generated_tokens[batch]:
            if t.item() != tokenizer.eos_token_id:
                number.append(int(tokenizer.decode(t)))
            else:
                break
        numbers_list.append(number)

    correct = 0

    for batch in range(generated_tokens.shape[0]):
        correct += number_reward(numbers_list[batch])

    return correct / generated_tokens.shape[0]


def parentheses_accuracy(generated_tokens, tokenizer):
    correct = 0
    for batch in range(generated_tokens.shape[0]):
        correct += float(validate_brackets(generated_tokens[batch], tokenizer))
    return correct / generated_tokens.shape[0]


class NumberValidator(SentenceValidator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def accuracy(self, sentences, tokenizer: PreTrainedTokenizer):
        return number_accuracy(sentences, tokenizer)

    def __call__(self, sentences, tokenizer: PreTrainedTokenizer):
        termination_token_id = tokenizer.eos_token_id

        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            device=sentences.device,
        )
        invalid[:, 0] = 1  # Empty sentence is never valid

        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                if sentences[i, j] == termination_token_id:
                    break  # Only unterminated sentences get a reward
                tokens = [tokenizer.decode(t) for t in sentences[i, : j + 1]]
                numbers = [int(number) for number in tokens]
                if number_reward(numbers):
                    invalid[i, j + 1] = False
                else:
                    invalid[i, j + 1] = True

        return invalid


class ParenthesesValidator(SentenceValidator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def accuracy(self, sentences, tokenizer: PreTrainedTokenizer):
        return parentheses_accuracy(sentences, tokenizer)

    def __call__(self, sentences, tokenizer: PreTrainedTokenizer):
        termination_token_id = tokenizer.eos_token_id

        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            device=sentences.device,
        )
        invalid[:, 0] = 1  # Empty sentence is never valid

        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                if sentences[i, j] == termination_token_id:
                    break  # Only unterminated sentences get a reward
                if validate_brackets(sentences[i, : j + 1], tokenizer):
                    invalid[i, j + 1] = False
                else:
                    invalid[i, j + 1] = True
            # if the final sentence is valid, the whole sentence traj. should be valid
            if validate_brackets(sentences[i, :], tokenizer):
                invalid[i, :] = False

        return invalid


class FrozenModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_validator: SentenceValidator,
        temperature=1.0,
        valid_sentence_alpha=None,
    ):
        self.temperature = temperature
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha

    def score(
        self,
        input_batch,
        prompt_length,
        model,
        tokenizer: PreTrainedTokenizer,
        reward_temperature=1.0,
        vocab_nice_mask=None,
        vocab_naughty_mask=None,
        naughty_vocab_alpha=-99,
        invalid_vocab_alpha=-99,
    ):
        # why lora_to_base?
        lora_to_base(model)

        reward, reward_unpenalized = score_fast(
            model=model,
            encoded_input=input_batch,
            termination_token_id=tokenizer.eos_token_id,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            vocab_nice_mask=vocab_nice_mask,
            vocab_naughty_mask=vocab_naughty_mask,
            naughty_vocab_alpha=naughty_vocab_alpha,
        )

        # reward /= self.temperature
        # reward_unpenalized /= self.temperature

        base_to_lora(model)

        if self.sentence_validator is not None:
            # use valid list instead of invalid list
            invalid = self.sentence_validator(input_batch[:, prompt_length:], tokenizer)
            invalid = invalid * invalid_vocab_alpha
            # TODO: max or mean
            reward = torch.min(reward, invalid)

        return reward, reward_unpenalized
