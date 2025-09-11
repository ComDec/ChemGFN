import json
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

        # Pre-tokenize the constant prompt once to avoid repeated work
        prompt = (
            "Generate a list of numbers that follow the rule: even number followed by "
            "odd number and odd number followed by even number and between 0 and 20: "
        )
        self.encoded_prompt = self.tokenizer(prompt, return_tensors="pt")["input_ids"]

    def __len__(self):
        return len(self.numbers_list)

    def __getitem__(self, index):
        return {
            "encoded_prompt": self.encoded_prompt.clone(),
            "numbers_list": torch.tensor(self.numbers_list[index]),
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
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2,
        )


class ParenthesesDataSet(Dataset):
    def __init__(self, prompts, tokenizer, total_size: int = 10000) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.total_size = total_size

        # Generate prompts by randomly sampling with replacement
        sampled_prompts = random.choices(prompts, k=self.total_size)
        # Pre-tokenize all prompts to avoid doing it in __getitem__
        self.encoded_prompts = [
            self.tokenizer(p, return_tensors="pt")["input_ids"] for p in sampled_prompts
        ]

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return {"encoded_prompt": self.encoded_prompts[index].clone()}


class ParenthesesDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer_name: str = "openai-community/gpt2",
        prompt_size: int = 1,
        total_size: int = 10000,
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

        self.prompt_size = prompt_size
        self.total_size = total_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage):
        with open(self.data_path) as f:
            prompts = f.readlines()

        # strip all the \n
        prompts = [prompt.strip() for prompt in prompts][: self.prompt_size]

        num_train = int(self.total_size * self.train_size)

        self.train_data = ParenthesesDataSet(prompts, self.tokenizer, num_train)
        self.val_data = ParenthesesDataSet(prompts, self.tokenizer, self.total_size - num_train)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2,
        )


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
        self.total_size = total_size

        self.is_instruct = is_instruct
        self.add_prompt = add_prompt
        self.buffer_tokenization = buffer_tokenization

        # Generate prompts by randomly sampling with replacement and pre-tokenize
        sampled_prompts = random.choices(prompts, k=self.total_size)
        if add_prompt:
            sampled_prompts = [
                SMILES_PROMPTS_AHEAD + p + BASE_PROMPTS_EXAMPLES + SMILES_PROMPTS_BEHIND
                for p in sampled_prompts
            ]
        self.encoded_prompts = [
            self.tokenizer(p, return_tensors="pt")["input_ids"] for p in sampled_prompts
        ]
        self.buffer_sample = buffer_sample
        self.allowed_vocab = allowed_vocab
        self.n_samples = n_samples

        # Pre-tokenize buffer samples if needed
        if self.buffer_sample is not None and self.buffer_tokenization:
            self.buffer_sample = [
                torch.tensor(
                    self.tokenizer.encode(x, add_special_tokens=False)
                ).reshape(-1)
                for x in self.buffer_sample
            ]

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
        encoded_prompt = self.encoded_prompts[index]

        sampled_buffers = []
        buffer_encoded_samples = []
        if self.buffer_sample is not None:
            sampled_buffer = random.sample(self.buffer_sample, self.n_samples)
            if self.allowed_vocab is not None:
                for i in range(len(sampled_buffer)):
                    if self.buffer_tokenization:
                        buffer_encoded_sample = sampled_buffer[i]
                    else:
                        sampled_buffers.append(
                            self.merge_chars_fast(sampled_buffer[i], self.allowed_vocab)
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
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2,
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
        self.total_size = total_size

        self.is_instruct = is_instruct
        self.add_prompt = add_prompt
        self.buffer_tokenization = buffer_tokenization

        # Generate prompts by randomly sampling with replacement and pre-tokenize
        sampled_prompts = random.choices(prompts, k=self.total_size)
        if add_prompt:
            final_prompts = [
                SMILES_PROMPTS_AHEAD
                + p["prompt"]
                + BASE_PROMPTS_EXAMPLES
                + SMILES_PROMPTS_BEHIND
                for p in sampled_prompts
            ]
        else:
            final_prompts = [p["prompt"] for p in sampled_prompts]
        self.encoded_prompts = [
            self.tokenizer(p, return_tensors="pt")["input_ids"] for p in final_prompts
        ]
        self.molecules = [p.get("molecule") for p in sampled_prompts]
        self.buffer_sample = buffer_sample
        self.allowed_vocab = allowed_vocab
        self.n_samples = n_samples

        if self.buffer_sample is not None and self.buffer_tokenization:
            self.buffer_sample = [
                torch.tensor(
                    self.tokenizer.encode(x, add_special_tokens=False)
                ).reshape(-1)
                for x in self.buffer_sample
            ]

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
        encoded_prompt = self.encoded_prompts[index]

        sampled_buffers = []
        buffer_encoded_samples = []
        if self.buffer_sample is not None:
            sampled_buffer = random.sample(self.buffer_sample, self.n_samples)
            if self.allowed_vocab is not None:
                for i in range(len(sampled_buffer)):
                    if self.buffer_tokenization:
                        buffer_encoded_sample = sampled_buffer[i]
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
            "molecule": self.molecules[index],
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
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2,
        )
