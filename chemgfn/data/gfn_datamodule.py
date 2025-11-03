import json
import os
import random
import warnings
from typing import Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .components.prompts import *

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def merge_chars_fast(chars, vocab_list):
    merged = []
    i = 0
    n = len(chars)

    while i < n:
        if i + 1 < n and (chars[i] + chars[i + 1]) in vocab_list:
            merged.append(chars[i] + chars[i + 1])
            i += 2
        else:
            merged.append(chars[i])
            i += 1

    return merged


class NumberDataSet(Dataset):
    def __init__(self, numbers_list, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.numbers_list = numbers_list

    def __len__(self):
        return len(self.numbers_list)

    def __getitem__(self, index):
        prompts = "Generate a list of numbers that follow the rule: even number followed by odd number and odd number followed by even number and between 0 and 20: "
        encoded_prompt = self.tokenizer(
            prompts,
            return_tensors="pt",
        )
        return {
            "encoded_prompt": encoded_prompt["input_ids"],
            "numbers_list": torch.tensor(self.numbers_list[index]),
            "buffer_encoded_sample": None,
        }


class NumberDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer_name: str = "openai-community/gpt2",
        train_size: float = 0.95,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_path = data_path
        self.train_size = train_size
        self.train_data = None
        self.val_data = None

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage):
        numbers_list = np.load(self.data_path)

        num_train = int(len(numbers_list) * self.train_size)
        self.train_data = NumberDataSet(numbers_list[:num_train], self.tokenizer)
        self.val_data = NumberDataSet(numbers_list[num_train:], self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=4 if self.num_workers > 0 else 2,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=4 if self.num_workers > 0 else 2,
            persistent_workers=True if self.num_workers > 0 else False,
        )


# Alias for backward compatibility - will be defined after BufferDataPipe
# PromptDataSet = BufferDataPipe


class BufferDataPipe(Dataset):
    def __init__(
        self,
        prompts,
        tokenizer: AutoTokenizer,
        total_size: int = 10000,
        is_instruct: bool = False,
        add_prompt: bool = False,
        buffer_sample: list = None,
        allowed_vocab: list = None,
        n_samples: int = 4,
        buffer_tokenization: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.total_size = total_size

        self.is_instruct = is_instruct
        self.add_prompt = add_prompt
        self.buffer_tokenization = buffer_tokenization

        # Generate prompts by randomly sampling with replacement
        self.prompts = random.choices(prompts, k=self.total_size)
        self.buffer_sample = buffer_sample
        self.allowed_vocab = allowed_vocab
        self.n_samples = n_samples

    def __len__(self):
        return len(self.prompts)

    @staticmethod
    def merge_chars_fast(chars, vocab_list):
        merged = []
        i = 0
        n = len(chars)

        while i < n:
            if i + 1 < n and (chars[i] + chars[i + 1]) in vocab_list:
                merged.append(chars[i] + chars[i + 1])
                i += 2
            else:
                merged.append(chars[i])
                i += 1

        return merged

    def generate_message(self, question):
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant. You are an expert on cheminformatics and helpful assistant. You are here to help the user generate valid molecules representation.",
            },
            {"role": "user", "content": f"{question}"},
        ]

    def __getitem__(self, index):
        if self.add_prompt:
            _prompt = (
                SMILES_PROMPTS_AHEAD
                + self.prompts[index]
                + BASE_PROMPTS_EXAMPLES
                + SMILES_PROMPTS_BEHIND
            )
        else:
            _prompt = self.prompts[index]

        if False:
            message = self.generate_message(_prompt)
            encoded_prompt = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            encoded_prompt = self.tokenizer(_prompt, return_tensors="pt")["input_ids"]

        buffer_encoded_samples = None

        # Process buffer samples if available
        if self.buffer_sample is not None:
            # Sample from buffer based on its type
            if isinstance(self.buffer_sample, torch.Tensor):
                # For 2D tensor: randomly select n_samples rows
                if self.buffer_sample.dim() == 2:
                    num_samples = min(self.n_samples, len(self.buffer_sample))
                    indices = torch.randperm(len(self.buffer_sample))[:num_samples]
                    sampled_buffer = self.buffer_sample[indices]
                    # Already tokenized tensor, return directly after potential padding
                    buffer_encoded_samples = sampled_buffer
                elif self.buffer_sample.dim() == 1:
                    # 1D tensor, treat as a single sample
                    buffer_encoded_samples = self.buffer_sample.unsqueeze(0)
                else:
                    # Unexpected tensor shape, skip
                    buffer_encoded_samples = None
            elif isinstance(self.buffer_sample, (list, tuple)):
                # For list/tuple: use random.sample
                num_samples = min(self.n_samples, len(self.buffer_sample))
                sampled_buffer = random.sample(list(self.buffer_sample), num_samples)
                buffer_encoded_samples = []

                for sample in sampled_buffer:
                    # Check if sample is already a tensor (token ids)
                    if isinstance(sample, torch.Tensor):
                        # Already tokenized, use directly
                        buffer_encoded_sample = sample.reshape(-1)
                    elif (
                        isinstance(sample, (list, tuple))
                        and len(sample) > 0
                        and isinstance(sample[0], (int, torch.Tensor))
                    ):
                        # List of token ids
                        buffer_encoded_sample = torch.tensor(sample).reshape(-1)
                    elif isinstance(sample, str):
                        # Pure text, need to process with allowed_vocab and merge_chars_fast
                        if self.allowed_vocab is not None:
                            if self.buffer_tokenization:
                                # Use standard tokenization
                                buffer_encoded_sample = torch.tensor(
                                    self.tokenizer.encode(sample, add_special_tokens=False)
                                ).reshape(-1)
                            else:
                                # Use merge_chars_fast with allowed_vocab
                                merged_chars = self.merge_chars_fast(sample, self.allowed_vocab)
                                buffer_encoded_sample = torch.tensor(
                                    [
                                        self.tokenizer.encode(x, add_special_tokens=False)
                                        for x in merged_chars
                                    ]
                                ).reshape(-1)
                        else:
                            # No allowed_vocab, use standard tokenization
                            buffer_encoded_sample = torch.tensor(
                                self.tokenizer.encode(sample, add_special_tokens=False)
                            ).reshape(-1)
                    else:
                        # Unknown type, skip this sample
                        continue

                    buffer_encoded_samples.append(buffer_encoded_sample)

                # Pad sequences if we have any valid samples
                if buffer_encoded_samples:
                    buffer_encoded_samples = pad_sequence(
                        buffer_encoded_samples,
                        batch_first=True,
                        padding_value=self.tokenizer.eos_token_id,
                    )
                else:
                    buffer_encoded_samples = None
            else:
                # Unknown buffer type, skip
                buffer_encoded_samples = None

        return {
            "encoded_prompt": encoded_prompt,
            "buffer_encoded_sample": buffer_encoded_samples,
        }


class BufferDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        buffer_sample_path: Optional[str] = None,
        tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        prompt_size: int = 1,
        total_size: int = 10000,
        train_size: float = 0.95,
        num_workers: int = 8,
        pin_memory: bool = True,
        add_prompt: bool = False,
        allowed_vocab_path: Optional[str] = None,
        n_samples: int = 8,
        buffer_tokenization: bool = False,
    ):
        """
        Data module for handling buffer data, which includes prompts and optional buffer samples.
        Args:
            data_path: Path to the file containing prompts.
            buffer_sample_path: Path to the buffer sample file.
            buffer_tokenization: If True, tokenizes the buffer samples. Else, construct tokens from vocabulary list
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_path = data_path
        self.train_size = train_size
        self.train_data = None
        self.val_data = None

        self.prompt_size = prompt_size
        self.total_size = total_size
        self.n_samples = n_samples

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.add_prompt = add_prompt
        self.buffer_tokenization = buffer_tokenization
        self.is_instruct = True if "Instruct" in tokenizer_name else False
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Check if allowed_vocab_path exists and load it
        self.allowed_tokens = None
        if allowed_vocab_path and os.path.exists(allowed_vocab_path):
            with open(allowed_vocab_path) as f:
                self.allowed_tokens = f.readlines()

        # Check if buffer_sample_path exists and load it
        if buffer_sample_path and os.path.exists(buffer_sample_path):
            self.buffer_sample = torch.load(buffer_sample_path)
        else:
            self.buffer_sample = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage):
        with open(self.data_path) as f:
            prompts = f.readlines()

        # strip all the \n
        prompts = [prompt.strip() for prompt in prompts][: self.prompt_size]
        num_train = int(self.total_size * self.train_size)

        self.train_data = BufferDataPipe(
            prompts,
            self.tokenizer,
            num_train,
            is_instruct=self.is_instruct,
            add_prompt=self.add_prompt,
            buffer_sample=self.buffer_sample,
            allowed_vocab=self.allowed_tokens,
            n_samples=self.n_samples,
            buffer_tokenization=self.buffer_tokenization,
        )
        self.val_data = BufferDataPipe(
            prompts,
            self.tokenizer,
            self.total_size - num_train,
            is_instruct=self.is_instruct,
            add_prompt=self.add_prompt,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=4 if self.num_workers > 0 else 2,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=4 if self.num_workers > 0 else 2,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class MolOptDataPipe(Dataset):
    def __init__(
        self,
        prompts: list[dict],
        tokenizer: AutoTokenizer,
        total_size: int = 10000,
        is_instruct: bool = False,
        add_prompt: bool = False,
        buffer_sample: list = None,
        allowed_vocab: list = None,
        n_samples: int = 4,
        buffer_tokenization: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.total_size = total_size

        self.is_instruct = is_instruct
        self.add_prompt = add_prompt
        self.buffer_tokenization = buffer_tokenization

        # Generate prompts by randomly sampling with replacement
        self.prompts = random.choices(prompts, k=self.total_size)
        self.buffer_sample = buffer_sample
        self.allowed_vocab = allowed_vocab
        self.n_samples = n_samples

    def __len__(self):
        return len(self.prompts)

    def generate_message(self, question):
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant. You are an expert on cheminformatics and helpful assistant. You are here to help the user generate valid molecules representation.",
            },
            {"role": "user", "content": f"{question}"},
        ]

    def __getitem__(self, index):
        if self.add_prompt:
            _prompt = (
                SMILES_PROMPTS_AHEAD
                + self.prompts[index]["prompt"]
                + BASE_PROMPTS_EXAMPLES
                + SMILES_PROMPTS_BEHIND
            )
        else:
            _prompt = self.prompts[index]["prompt"]

        if False:
            message = self.generate_message(_prompt)
            encoded_prompt = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            encoded_prompt = self.tokenizer(_prompt, return_tensors="pt")["input_ids"]

        sampled_buffers = []
        buffer_encoded_samples = []
        if self.buffer_sample is not None:
            sampled_buffer = random.sample(self.buffer_sample, self.n_samples)
            if self.allowed_vocab is not None:
                for i in range(len(sampled_buffer)):
                    if self.buffer_tokenization:
                        buffer_encoded_sample = torch.tensor(
                            self.tokenizer.encode(sampled_buffer[i], add_special_tokens=False)
                        ).reshape(-1)
                    else:
                        sampled_buffers.append(
                            merge_chars_fast(sampled_buffer[i], self.allowed_vocab)
                        )
                        buffer_encoded_sample = torch.tensor(
                            [
                                self.tokenizer.encode(x, add_special_tokens=False)
                                for x in sampled_buffers[i]
                            ]
                        ).reshape(-1)

                    buffer_encoded_samples.append(buffer_encoded_sample)

                buffer_encoded_samples = pad_sequence(
                    buffer_encoded_samples,
                    batch_first=True,
                    padding_value=self.tokenizer.eos_token_id,
                )
            else:
                sampled_buffers = list(sampled_buffer)

        else:
            buffer_encoded_samples = None

        return {
            "encoded_prompt": encoded_prompt,
            "buffer_encoded_sample": buffer_encoded_samples,
            "molecule": self.prompts[index]["molecule"],
        }


class MolOptDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        buffer_sample_path: Optional[str] = None,
        tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        prompt_size: int = 1,
        total_size: int = 10000,
        train_size: float = 0.95,
        num_workers: int = 8,
        pin_memory: bool = True,
        add_prompt: bool = False,
        allowed_vocab_path: Optional[str] = None,
        n_samples: int = 4,
        buffer_tokenization: bool = False,
    ):
        """
        Data module for handling buffer data, which includes prompts and optional buffer samples.
        Args:
            data_path: Path to the file containing prompts.
            buffer_sample_path: Path to the buffer sample file.
            buffer_tokenization: If True, tokenizes the buffer samples. Else, construct tokens from vocabulary list
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_path = data_path
        self.train_size = train_size
        self.train_data = None
        self.val_data = None

        self.prompt_size = prompt_size
        self.total_size = total_size
        self.n_samples = n_samples

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.add_prompt = add_prompt
        self.buffer_tokenization = buffer_tokenization
        self.is_instruct = True if "Instruct" in tokenizer_name else False
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        with open(allowed_vocab_path) as f:
            self.allowed_tokens = f.readlines() if allowed_vocab_path else None

        self.buffer_sample = torch.load(buffer_sample_path) if buffer_sample_path else None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage):
        with open(self.data_path) as f:
            data_base = json.load(f)

        # strip all the \n
        for i, _ in enumerate(data_base):
            data_base[i]["prompt"] = data_base[i]["prompt"].strip()

        prompts = data_base[: self.prompt_size]
        num_train = int(self.total_size * self.train_size)

        self.train_data = MolOptDataPipe(
            prompts,
            self.tokenizer,
            num_train,
            is_instruct=self.is_instruct,
            add_prompt=self.add_prompt,
            buffer_sample=self.buffer_sample,
            allowed_vocab=self.allowed_tokens,
            n_samples=self.n_samples,
            buffer_tokenization=self.buffer_tokenization,
        )
        self.val_data = MolOptDataPipe(
            prompts,
            self.tokenizer,
            self.total_size - num_train,
            is_instruct=self.is_instruct,
            add_prompt=self.add_prompt,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=4 if self.num_workers > 0 else 2,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=4 if self.num_workers > 0 else 2,
            persistent_workers=True if self.num_workers > 0 else False,
        )
