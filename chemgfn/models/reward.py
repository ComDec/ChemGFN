import random
from typing import Literal, Optional

import numpy as np
import partialsmiles as ps
import torch

# slient the warning
from rdkit import Chem, RDLogger
from transformers import PreTrainedTokenizer

from chemgfn.utils.gfn_utils import base_to_lora, lora_to_base
from chemgfn.utils.rdkit_utils import FUNCTION_MAPPING, verify_smiles, verify_smiles_pa

RDLogger.DisableLog("rdApp.*")


@torch.no_grad()
def score_fast(
    model,
    encoded_input,
    termination_token_id,
    skip_first,
    reward_temperature=1.0,
    agree_list=None,
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
    if agree_list is not None:
        logits[~agree_list.permute(1, 0, 2)] = -torch.inf
    # Add reward temperature
    logits /= reward_temperature
    logprob = logits.log_softmax(-1)
    token_ids = encoded_input[:, skip_first:].unsqueeze(-1)

    # the last pos for logprob must be the termination token
    logPF = logprob[:, :-1].gather(-1, token_ids).squeeze(-1)
    logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)

    reward = logprob[
        :, :, termination_token_id
    ]  # logP(generated[i+1]=term | prompt + generated[:i+1])

    reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)

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
    def __init__(self, scorer: str = "sa", backend: Literal["rdkit", "pa"] = "rdkit") -> None:
        super().__init__(scorer)
        self.score_function = FUNCTION_MAPPING[scorer]
        self.scorer_name = scorer
        self.backend = backend

    @staticmethod
    def rdkit_validate(smiles: str) -> bool:
        """Validate SMILES using RDKit."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol)
            return True
        except Exception:
            return False

    @staticmethod
    def pa_validate(smiles: str) -> bool:
        """Validate SMILES using PartialSMILES."""
        try:
            ps.ParseSmiles(smiles)
            return True
        except Exception:
            return False

    def smiles_accuracy(self, generated_tokens, tokenizer, target_molecule: Optional[str] = None):
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
        avg_sa_score = 0

        for batch in range(generated_tokens.shape[0]):
            tokens = "".join(smiles_list[batch])
            if target_molecule is not None:
                tokens = target_molecule.replace("*", tokens)
            if self.backend == "rdkit":
                valid = self.rdkit_validate(tokens)
            elif self.backend == "pa":
                valid = self.pa_validate(tokens)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

            if not valid:
                mol = None
            else:
                mol = Chem.MolFromSmiles(tokens)

            avg_sa_score += self.score_function(mol) if mol else 0
            correct += 1 if mol else 0

        return {
            "acc": correct / generated_tokens.shape[0],
            f"{self.scorer_name}": avg_sa_score / generated_tokens.shape[0],
            f"{self.scorer_name}_filter": avg_sa_score / correct if correct > 0 else 0.0,
        }

    def accuracy(
        self, sentences, tokenizer: PreTrainedTokenizer, target_molecule: Optional[str] = None
    ):
        return self.smiles_accuracy(sentences, tokenizer, target_molecule)

    def __call__(
        self, sentences, tokenizer: PreTrainedTokenizer, target_molecule: Optional[str] = None
    ):
        termination_token_id = tokenizer.eos_token_id

        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            device=sentences.device,
        )
        invalid[:, 0] = 1  # Empty sentence is never valid

        valid_score = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            device=sentences.device,
        )

        valid_score[:, 0] = 0

        global_score = torch.zeros(
            sentences.shape[0],
            device=sentences.device,
        )

        full_tokens_list = []
        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                if sentences[i, j] == termination_token_id:
                    tokens = "".join(
                        tokenizer.decode(sentences[i, :j], skip_special_tokens=True).split()
                    )
                    if target_molecule is not None:
                        full_tokens = target_molecule.replace("*", tokens)
                    try:
                        global_score[i] = self.score_function(Chem.MolFromSmiles(full_tokens))
                    except:
                        global_score[i] = 0.0
                    break  # Only unterminated sentences get a reward
                tokens = "".join(tokenizer.decode(sentences[i, : j + 1]).split())
                if target_molecule is not None:
                    full_tokens = target_molecule.replace("*", tokens)

                valid = (
                    verify_smiles(full_tokens)
                    if self.backend == "rdkit"
                    else verify_smiles_pa(full_tokens)
                )
                valid_score[i, j + 1] = (
                    self.score_function(Chem.MolFromSmiles(full_tokens)) if valid else -1
                )

                if valid:
                    invalid[i, j + 1] = 0
                else:
                    invalid[i, j + 1] = 1

            full_tokens_list.append(full_tokens)

        return {
            "invalid": invalid,
            "global_score": global_score,
            "valid_score": valid_score,
            "full_tokens": full_tokens_list,
        }


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


class Reference_Target_Score_Positive_Mixed:
    def __init__(
        self,
        sentence_validator: SentenceValidator,
        valid_sentence_alpha=None,
        invalid_start_ratio: float = 0.2,
        invalid_end_ratio: float = 1.2,
        target_score_alpha: float = 0.1,
        target_score_threshold: float = 5.0,
    ):
        # reward = logP + alpha * reward_norm
        self.target_score_alpha = target_score_alpha
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha
        self.invalid_start_ratio = invalid_start_ratio
        self.invalid_end_ratio = invalid_end_ratio
        self.target_score_threshold = target_score_threshold

        # temperature for the target score
        self.temperature = 1.0

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
        target_molecule: Optional[str] = None,
        **kwargs,
    ):
        lora_to_base(model)

        reference_logits, _ = score_fast(
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
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, target_molecule
            )
            invalid = validator_dict["invalid"]
            valid_score = validator_dict["valid_score"]

            reference_logits_norm = torch.nn.functional.log_softmax(
                reference_logits / reward_temperature, dim=-1
            )
            reward_norm = (valid_score - valid_score.mean()) / (valid_score.std() + 1e-8)
            reward_mixed = reference_logits_norm + self.target_score_alpha * reward_norm

        return reward_mixed, reward_mixed


class Reference_Invalid_Mask:
    def __init__(
        self,
        sentence_validator: SentenceValidator,
        valid_sentence_alpha=None,
        invalid_start_ratio: float = 0.2,
        invalid_end_ratio: float = 1.2,
        target_score_alpha: float = 0.1,
        **kwargs,
    ):
        # reward = logP + alpha * reward_norm
        self.target_score_alpha = target_score_alpha
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha
        self.invalid_start_ratio = invalid_start_ratio
        self.invalid_end_ratio = invalid_end_ratio

        # temperature for the target score
        self.temperature = 1.0

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
        advantage_alpha=0.5,
        target_molecule: Optional[str] = None,
        agree_list=None,
        **kwargs,
    ):
        lora_to_base(model)
        reference_logits, _ = score_fast(
            model=model,
            encoded_input=input_batch,
            termination_token_id=tokenizer.eos_token_id,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            agree_list=agree_list,
        )

        base_to_lora(model)

        if self.sentence_validator is not None:
            # use valid list instead of invalid list
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, target_molecule
            )
            invalid = validator_dict["invalid"]
            reference_logits_norm = reference_logits / reward_temperature
            reward_mixed = reference_logits_norm

            # apply invalid mask
            invalid_list = []
            for i in range(reward_mixed.shape[0]):
                # apply group advantage
                min_value = torch.min(reward_mixed[i, :])
                start_values = min_value * self.invalid_start_ratio
                end_values = min_value * self.invalid_end_ratio
                seq = torch.linspace(
                    start=start_values,
                    end=end_values,
                    steps=reward_mixed.shape[1],
                    device=reward_mixed.device,
                )
                invalid_list.append(seq)

            invalid_value = torch.stack(invalid_list, dim=0) * invalid  # shape: [B, L]

        return {
            "reward": torch.min(reward_mixed, invalid_value),
            "reward_unpenalized": reward_mixed,
            "full_tokens": validator_dict.get("full_tokens", None),
            "log_pf_ref": reference_logits,
        }


class Reference_Invalid_Mask_Z_score:
    def __init__(
        self,
        sentence_validator: SentenceValidator,
        valid_sentence_alpha=None,
        invalid_start_ratio: float = 0.2,
        invalid_end_ratio: float = 1.2,
        target_score_alpha: float = 0.1,
        **kwargs,
    ):
        # reward = logP + alpha * reward_norm
        self.target_score_alpha = target_score_alpha
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha
        self.invalid_start_ratio = invalid_start_ratio
        self.invalid_end_ratio = invalid_end_ratio

        # temperature for the target score
        self.temperature = 1.0

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
        advantage_alpha=0.5,
        target_molecule: Optional[str] = None,
        agree_list=None,
        **kwargs,
    ):
        lora_to_base(model)
        reference_logits, _ = score_fast(
            model=model,
            encoded_input=input_batch,
            termination_token_id=tokenizer.eos_token_id,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            agree_list=agree_list,
        )

        base_to_lora(model)

        if self.sentence_validator is not None:
            # use valid list instead of invalid list
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, target_molecule
            )
            invalid = validator_dict["invalid"]
            reference_logits_norm = reference_logits_norm = (
                reference_logits - reference_logits.mean()
            ) / (reference_logits.std() + 1e-8)
            reward_mixed = reference_logits_norm

            # apply invalid mask
            invalid_list = []
            for i in range(reward_mixed.shape[0]):
                # apply group advantage
                min_value = torch.min(reward_mixed[i, :])
                start_values = min_value * self.invalid_start_ratio
                end_values = min_value * self.invalid_end_ratio
                seq = torch.linspace(
                    start=start_values,
                    end=end_values,
                    steps=reward_mixed.shape[1],
                    device=reward_mixed.device,
                )
                invalid_list.append(seq)

            invalid_value = torch.stack(invalid_list, dim=0) * invalid  # shape: [B, L]

        return {
            "reward": torch.min(reward_mixed, invalid_value),
            "reward_unpenalized": reward_mixed,
            "full_tokens": validator_dict.get("full_tokens", None),
            "log_pf_ref": reference_logits,
            "validator_dict": validator_dict,
        }


class Reference_Target_Score_Positive_Mixed_Invalid_Mask:
    def __init__(
        self,
        sentence_validator: SentenceValidator,
        valid_sentence_alpha=None,
        invalid_start_ratio: float = 0.2,
        invalid_end_ratio: float = 1.2,
        target_score_alpha: float = 0.1,
        **kwargs,
    ):
        # reward = logP + alpha * reward_norm
        self.target_score_alpha = target_score_alpha
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha
        self.invalid_start_ratio = invalid_start_ratio
        self.invalid_end_ratio = invalid_end_ratio

        # temperature for the target score
        self.temperature = 1.0

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
        advantage_alpha=0.5,
        target_molecule: Optional[str] = None,
        agree_list=None,
        **kwargs,
    ):
        lora_to_base(model)
        reference_logits, _ = score_fast(
            model=model,
            encoded_input=input_batch,
            termination_token_id=tokenizer.eos_token_id,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            agree_list=agree_list,
        )

        base_to_lora(model)

        if self.sentence_validator is not None:
            # use valid list instead of invalid list
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, target_molecule
            )
            invalid = validator_dict["invalid"]
            valid_score = validator_dict["valid_score"]
            reference_logits_norm = (reference_logits - reference_logits.mean()) / (
                reference_logits.std() + 1e-8
            )  # normalize logits

            # TODO: print reward/ vars
            reward_norm = (valid_score - valid_score.mean()) / (valid_score.std() + 1e-8)
            reward_mixed = reference_logits_norm + advantage_alpha * reward_norm

            # apply invalid mask
            invalid_list = []

            for i in range(reward_mixed.shape[0]):
                min_value = -1
                start_values = min_value * self.invalid_start_ratio
                end_values = min_value * self.invalid_end_ratio
                seq = torch.linspace(
                    start=start_values,
                    end=end_values,
                    steps=reward_mixed.shape[1],
                    device=reward_mixed.device,
                )
                invalid_list.append(seq)

            invalid_value = torch.stack(invalid_list, dim=0) * invalid  # shape: [B, L]

        return {
            "reward": torch.min(reward_mixed, invalid_value),
            "reward_unpenalized": reward_mixed,
            "full_tokens": validator_dict.get("full_tokens", None),
            "log_pf_ref": reference_logits,
            "validator_dict": validator_dict,
        }


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
        min_len: int = 10,
        **kwargs,
    ):
        # P-terminal
        # 1. base prob
        sum_along_D = agree_list.sum(dim=-1, keepdim=True)  # B*L*1
        valid_mask = sum_along_D > 0
        uniform_base = torch.where(agree_list, 1.0 / sum_along_D.clamp(min=1e-6), 0).detach()

        # 2. length exp increase for eos
        first_position_counts = sum_along_D[0, :, :].squeeze(-1)  # [B]

        # log_start_for_first_pos.
        log_start = torch.log((1 / first_position_counts).clamp(min=1e-6))  # [B]
        seq_len = input_batch.size(1) - prompt_length
        steps = torch.linspace(0, 1, seq_len + 1, device=input_batch.device)  # [L+1]
        log_space = log_start[None, :] * (1 - steps[:, None])  # [L+1, B]
        length_increase = torch.exp(log_space).unsqueeze(-1)  # [L+1, B, 1]

        eos_mask = torch.zeros_like(agree_list)
        eos_mask[..., termination_token_id] = agree_list[..., termination_token_id]

        # P(seq) + P(eos) * length_increase
        uniform_probs = uniform_base * (~eos_mask)
        eos_probs = torch.log_softmax(uniform_base * eos_mask * length_increase, dim=-1)
        logprob = torch.log_softmax(uniform_probs, dim=-1)

        # global compensation

        # if quality_scorer is not None:
        #     # 获取完整序列质量分数 [B]
        #     full_sequences = input_batch[:, prompt_length:]
        #     quality_scores = quality_scorer(full_sequences)  # [B]

        #     # 在终止位置加入质量补偿
        #     quality_bonus = quality_scores.unsqueeze(-1) * torch.linspace(
        #         0, 1, seq_len, device=input_batch.device
        #     )  # [B, L]

        #     # 将质量补偿映射到EOS位置
        #     eos_positions = (input_batch[:, prompt_length:] == termination_token_id)
        #     psudo_logits[..., termination_token_id] += (
        #         quality_bonus.masked_fill(~eos_positions, 0)
        #     )

        token_ids = input_batch[:, prompt_length:].unsqueeze(-1)  # [B, L, 1]
        logPF = logprob[1:].gather(-1, token_ids.transpose(0, 1)).squeeze(-1)  # [L-1, B]
        logP = logPF.cumsum(dim=0)  # [L-1, B]
        reward = eos_probs[
            ..., termination_token_id
        ]  # logP(generated[i+1]=term | prompt + generated[:i+1])
        reward[1:,] += logP  # logP(generated[:i] + term | prompt)

        non_term_mask = (input_batch != termination_token_id)[:, prompt_length:]

        non_term_mask = torch.cat(
            (
                non_term_mask.new_ones(non_term_mask.shape[0], 1),
                non_term_mask,
            ),
            dim=-1,
        )

        # before reach the min_len, the reward is -999
        # reward = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_len, -999, reward)

        reward_unpenalized = reward.clone()

        # valid_eos_mask = (
        #     input_batch[:, prompt_length:] == termination_token_id
        # ) & valid_mask.squeeze(-1)

        # # Eos based reward
        # termination_logits = psudo_logits[..., termination_token_id]  # [B, L]
        # reward = termination_logits * valid_eos_mask.float()
        # length_penalty = -0.1 * torch.arange(1, seq_len + 1, device=reward.device).float().log()
        # reward += length_penalty.unsqueeze(0)
        # reward_unpenalized = reward.clone()

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
                # generate linear incresement part for the rest of the sequence
                seq = torch.linspace(
                    start=start_values[i],
                    end=end_values[i],
                    steps=reward.shape[1],
                    device=reward.device,
                )
                invalid_list.append(seq)

            invalid_value = torch.stack(invalid_list, dim=0)  # shape: [B, L]

            reward = torch.min(reward.permute(1, 0), invalid_value.permute(1, 0) * invalid)

        return reward, reward_unpenalized.permute(1, 0)
