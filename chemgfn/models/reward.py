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


@torch.no_grad()
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
    logits = logits.detach()[:, skip_first - 1 :]

    # score the log probability of the input sequence while ignoring termination and padding tokens
    # if vocab_naughty_mask is not None:
    #     logits[:, :, vocab_naughty_mask] += naughty_vocab_alpha

    # Add reward temperature
    logits /= reward_temperature
    logprob = logits.log_softmax(-1)
    token_ids = encoded_input[:, skip_first:].unsqueeze(-1)

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


class BracketValidator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        # 定义括号映射关系（右括号 -> 左括号）
        self.bracket_map = {")": "(", "]": "[", ">": "<"}
        self.left_brackets = set(self.bracket_map.values())
        self.right_brackets = set(self.bracket_map.keys())
        self.tokenizer = tokenizer

    def preprocess(self, s: str) -> tuple[str, bool]:
        decoded_list = []
        # remove the eos token
        total_length = len(s)
        for idx, t in enumerate(s):
            if t.item() != self.tokenizer.eos_token_id:
                decoded_list.append(self.tokenizer.decode(t))
            else:
                break

        # think of "<<" due to bpe, we need to re-concatenate the tokens
        decoded_string = "".join(decoded_list)

        # if total_length > idx, it means the sentence is termiated
        return decoded_string, total_length > idx

    def is_valid(self, s: str) -> bool:
        """验证完整括号匹配"""
        s, early_terminated = self.preprocess(s)
        stack = []
        for char in s:
            if char in self.left_brackets:
                stack.append(char)
            elif char in self.right_brackets:
                if not stack or stack[-1] != self.bracket_map[char]:
                    return False
                stack.pop()
            else:
                return False  # 包含非括号字符
        valid = not stack
        return valid

    def is_valid_prefix(self, s: str) -> bool:
        """验证有效前缀"""
        s, early_terminated = self.preprocess(s)
        stack = []
        for char in s:
            if char in self.left_brackets:
                stack.append(char)
            elif char in self.right_brackets:
                if not stack or stack[-1] != self.bracket_map[char]:
                    return False
                stack.pop()
            else:
                return False  # 包含非括号字符
        return True


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
        validator = BracketValidator(tokenizer)
        correct = 0
        for batch in range(sentences.shape[0]):
            valid = validator.is_valid(sentences[batch])
            correct += valid
        return correct / sentences.shape[0]

    def __call__(self, sentences, tokenizer: PreTrainedTokenizer):
        termination_token_id = tokenizer.eos_token_id
        validator = BracketValidator(tokenizer)

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

                valid = validator.is_valid(sentences[i, : j + 1])
                if valid:
                    # current step is valid and yield eos now, so the model should terminate now.
                    invalid[i, j + 1] = 0
                else:
                    invalid[i, j + 1] = 1

        return invalid


class FrozenModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_validator: SentenceValidator,
        temperature=1.0,
        valid_sentence_alpha=None,
        start_ratio: float = 0.2,
        end_ratio: float = 1.2,
    ):
        self.temperature = temperature
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

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

            # TODO: max or mean
            # reward = reward + invalid

            # v0
            # invalid_value = invalid * invalid_vocab_alpha
            # reward = torch.min(reward, invalid_value)

            # v1
            # reward_min_value = torch.min(reward, dim=-1).values
            # invalid_min_value = reward_min_value.unsqueeze(-1) * invalid * 1.05
            # reward = torch.min(reward, invalid_min_value)

            # v3
            # reward_min = torch.min(reward, dim=-1).values  # 形状 [B]

            # start_values = reward_min * 0.2
            # end_values = reward_min * 1.2

            # invalid_list = []
            # for i in range(reward.shape[0]):
            #     seq = torch.linspace(
            #         start=start_values[i],
            #         end=end_values[i],
            #         steps=reward.shape[1],  # 直接使用L作为步数
            #         device=reward.device
            #     )
            #     invalid_list.append(seq)

            # invalid_value = torch.stack(invalid_list, dim=0)

            # reward = torch.min(reward, invalid_value * invalid)

            # v4
            # 确保已经定义了prompt_len，例如：prompt_len = ...（某个整数）
            reward_min = torch.min(reward, dim=-1).values  # 形状 [B]
            start_values = reward_min * self.start_ratio
            end_values = reward_min * self.end_ratio

            invalid_list = []
            for i in range(reward.shape[0]):
                # 前prompt_len个元素固定为start_values[i]
                prefix = torch.full((prompt_length,), start_values[i], device=reward.device)
                # 生成后面的线性增长部分
                if reward.shape[1] > prompt_length:
                    suffix = torch.linspace(
                        start=start_values[i],
                        end=end_values[i],
                        steps=reward.shape[1] - prompt_length,
                        device=reward.device,
                    )
                    seq = torch.cat([prefix, suffix])
                else:
                    # 如果prompt_len >= L，直接取prefix的前L个元素
                    seq = prefix[: reward.shape[1]]
                invalid_list.append(seq)

            invalid_value = torch.stack(invalid_list, dim=0)  # 形状 [B, L]

            # 假设invalid是一个与reward形状相同的掩码张量（值为0或1）
            reward = torch.min(reward, invalid_value * invalid)

        return reward, reward_unpenalized
