import random
from typing import Literal

import numpy as np
import torch

# slient the warning
from rdkit import Chem, RDLogger
from transformers import PreTrainedTokenizer

from chemgfn.utils.gfn_utils import base_to_lora, lora_to_base
from chemgfn.utils.rdkit_utils import sa_scorer, verify_smiles

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


@torch.no_grad()
def uniform_score(
    agree_list: list[int],
    encoded_input,
    termination_token_id,
    skip_first,
    reward_temperature=1.0,
    prompt_cache=None,
):
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

    @staticmethod
    def smiles_accuracy(generated_tokens, tokenizer):
        smiles_list = []

        for batch in range(generated_tokens.shape[0]):
            string = []
            for t in generated_tokens[batch]:
                if t.item() != tokenizer.eos_token_id:
                    string.append(tokenizer.decode(t))
                else:
                    break
            smiles_list.append(string)

        correct = 0
        strict_correct = 0
        avg_sa_score = 0

        for batch in range(generated_tokens.shape[0]):
            try:
                tokens = "".join(smiles_list[batch])
                mol = Chem.MolFromSmiles(tokens)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol)
                        strict_correct += 1
                    except:
                        strict_correct += 0
            except:
                mol = None

            avg_sa_score += sa_scorer(mol) if mol else 0
            correct += 1 if mol else 0

        return {
            "acc": correct / generated_tokens.shape[0],
            "strict_acc": strict_correct / generated_tokens.shape[0],
            "avg_sa": avg_sa_score / generated_tokens.shape[0],
        }

    def accuracy(self, sentences, tokenizer: PreTrainedTokenizer):
        return self.smiles_accuracy(sentences, tokenizer)

    def __call__(self, sentences, tokenizer: PreTrainedTokenizer):
        termination_token_id = tokenizer.eos_token_id

        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            device=sentences.device,
        )
        invalid[:, 0] = 1  # Empty sentence is never valid

        global_invalid = torch.zeros(
            sentences.shape[0],
            device=sentences.device,
        )

        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                if sentences[i, j] == termination_token_id:
                    tokens = "".join(
                        tokenizer.decode(sentences[i, :j], skip_special_tokens=True).split()
                    )
                    global_invalid[i] = sa_scorer(Chem.MolFromSmiles(tokens))
                    break  # Only unterminated sentences get a reward
                tokens = "".join(tokenizer.decode(sentences[i, : j + 1]).split())
                valid = verify_smiles(tokens)
                if valid:
                    invalid[i, j + 1] = 0
                else:
                    invalid[i, j + 1] = 1

        return {"invalid": invalid, "sa": global_invalid}


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

    def has_multiple_nesting(self, s: str) -> bool:
        """检查有效括号字符串是否存在至少两层的嵌套"""
        s, early_terminated = self.preprocess(s)
        stack = []
        has_nesting = False
        for char in s:
            if char in self.left_brackets:
                stack.append(char)
            elif char in self.right_brackets:
                if not stack or stack[-1] != self.bracket_map[char]:
                    return False  # 括号不匹配，结构无效
                stack.pop()
                # 检查闭合后栈的深度是否仍有未闭合的左括号
                if len(stack) >= 1:
                    has_nesting = True
            else:
                return False  # 包含非括号字符，结构无效
        if stack:  # 检查是否所有括号已闭合
            return False
        return has_nesting


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
        return {"acc": number_accuracy(sentences, tokenizer)}

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
        nest = 0
        for batch in range(sentences.shape[0]):
            valid = validator.is_valid(sentences[batch])
            nested = validator.has_multiple_nesting(sentences[batch])
            correct += valid
            nest += int(nested)

        return {"acc": correct / sentences.shape[0], "nest": nest / sentences.shape[0]}

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
        invalid_start_ratio: float = 0.2,
        invalid_end_ratio: float = 1.2,
    ):
        self.temperature = temperature
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha
        self.invalid_start_ratio = invalid_start_ratio
        self.invalid_end_ratio = invalid_end_ratio

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
        **kwargs,
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
            invalid = self.sentence_validator(input_batch[:, prompt_length:], tokenizer)["invalid"]

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

            start_values = reward_min * self.invalid_start_ratio
            end_values = reward_min * self.invalid_end_ratio

            invalid_list = []
            for i in range(reward.shape[0]):
                # prompt_len: large enough value
                # prefix = torch.full((prompt_length,), start_values[i], device=reward.device)

                # generate linear incresement part for the rest of the sequence
                seq = torch.linspace(
                    start=start_values[i],
                    end=end_values[i],
                    steps=reward.shape[1],
                    device=reward.device,
                )
                invalid_list.append(seq)

            invalid_value = torch.stack(invalid_list, dim=0)  # shape: [B, L]

            reward = torch.min(reward, invalid_value * invalid)

        return reward, reward_unpenalized


class SMILESFrozenModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_validator: SentenceValidator,
        temperature=1.0,
        valid_sentence_alpha=None,
        invalid_start_ratio: float = 0.2,
        invalid_end_ratio: float = 1.2,
        sa_threshold: float = 5.0,
    ):
        self.temperature = temperature
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha
        self.invalid_start_ratio = invalid_start_ratio
        self.invalid_end_ratio = invalid_end_ratio
        self.sa_threshold = sa_threshold

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
        **kwargs,
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

        base_to_lora(model)

        if self.sentence_validator is not None:
            # use valid list instead of invalid list
            validator_dict = self.sentence_validator(input_batch[:, prompt_length:], tokenizer)
            invalid = validator_dict["invalid"]
            sa = validator_dict["sa"]

            # v4
            # 确保已经定义了prompt_len，例如：prompt_len = ...（某个整数）
            reward_min = torch.min(reward, dim=-1).values  # 形状 [B]

            start_values = reward_min * self.invalid_start_ratio
            end_values = reward_min * self.invalid_end_ratio

            invalid_list = []
            for i in range(reward.shape[0]):
                # prompt_len: large enough value
                # prefix = torch.full((prompt_length,), start_values[i], device=reward.device)

                # generate linear incresement part for the rest of the sequence
                # rectify using SA Scores
                seq = torch.linspace(
                    start=start_values[i] * ((10 - sa[i]) / self.sa_threshold),
                    end=end_values[i],
                    steps=reward.shape[1],
                    device=reward.device,
                )
                invalid_list.append(seq)

            invalid_value = torch.stack(invalid_list, dim=0)  # shape: [B, L]

            reward = torch.min(reward, invalid_value * invalid)

        return reward, reward_unpenalized


class UniformModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_validator: SentenceValidator,
        temperature=1.0,
        valid_sentence_alpha=None,
        invalid_start_ratio: float = 0.2,
        invalid_end_ratio: float = 1.2,
    ):
        self.temperature = temperature
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha
        self.invalid_start_ratio = invalid_start_ratio
        self.invalid_end_ratio = invalid_end_ratio

    def score(
        self,
        input_batch,
        prompt_length,
        tokenizer: PreTrainedTokenizer,
        reward_temperature=1.0,
        termination_token_id: int = -1,
        agree_list: list[int] = None,
        termination_logits: torch.Tensor = None,
        **kwargs,
    ):
        sum_along_D = agree_list.sum(dim=-1, keepdim=True)  # B*L*1

        psudo_logits = torch.where(agree_list, 1.0 / sum_along_D.clamp(min=1e-6), -torch.inf)
        logprob = torch.log(psudo_logits.softmax(dim=-1) + 1e-6).permute(1, 0, 2)  # B*L*V
        token_ids = input_batch[:, prompt_length:].unsqueeze(-1)
        logPF = logprob[:, :-1].gather(-1, token_ids).squeeze(-1)
        logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)

        reward = logprob[
            :, :, termination_token_id
        ]  # logP(generated[i+1]=term | prompt + generated[:i+1])

        reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)

        non_term_mask = (input_batch != termination_token_id)[:, prompt_length:]

        non_term_mask = torch.cat(
            (
                non_term_mask.new_ones(non_term_mask.shape[0], 1),
                non_term_mask,
            ),
            dim=-1,
        )  # Start (i.e., empty) state has never terminated

        reward[~non_term_mask] = 0.0
        reward_unpenalized = reward.clone()

        if self.sentence_validator is not None:
            # use valid list instead of invalid list
            invalid = self.sentence_validator(input_batch[:, prompt_length:], tokenizer)

            # v4
            # 确保已经定义了prompt_len，例如：prompt_len = ...（某个整数）
            reward_min = torch.min(reward, dim=-1).values  # 形状 [B]

            start_values = reward_min * self.invalid_start_ratio
            end_values = reward_min * self.invalid_end_ratio

            invalid_list = []
            for i in range(reward.shape[0]):
                # prompt_len: large enough value
                # prefix = torch.full((prompt_length,), start_values[i], device=reward.device)

                # generate linear incresement part for the rest of the sequence
                seq = torch.linspace(
                    start=start_values[i],
                    end=end_values[i],
                    steps=reward.shape[1],
                    device=reward.device,
                )
                invalid_list.append(seq)

            invalid_value = torch.stack(invalid_list, dim=0)  # shape: [B, L]

            reward = torch.min(reward, invalid_value * invalid)

        return reward, reward_unpenalized
