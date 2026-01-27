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
        scaffolds=None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.total_size = total_size

        self.is_instruct = is_instruct
        self.add_prompt = add_prompt
        self.buffer_tokenization = buffer_tokenization

        # Generate prompts by randomly sampling with replacement
        if scaffolds is not None:
            if len(scaffolds) != len(prompts):
                raise ValueError("scaffolds length must match prompts length")
            indices = random.choices(range(len(prompts)), k=self.total_size)
            self.prompts = [prompts[i] for i in indices]
            self.scaffolds = [scaffolds[i] for i in indices]
        else:
            self.prompts = random.choices(prompts, k=self.total_size)
            self.scaffolds = None
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
                if len(self.buffer_sample) > 0:
                    num_samples = min(self.n_samples, len(self.buffer_sample))
                    if num_samples == 0:
                        buffer_encoded_samples = None
                    else:
                        sampled_buffer = random.sample(list(self.buffer_sample), num_samples)
                        first_sample = sampled_buffer[0]

                        # Tokenized tensor list: use directly
                        if isinstance(first_sample, torch.Tensor):
                            buffer_encoded_samples = pad_sequence(
                                [sample.reshape(-1) for sample in sampled_buffer],
                                batch_first=True,
                                padding_value=self.tokenizer.eos_token_id,
                            )
                        # Tokenized ids list
                        elif (
                            isinstance(first_sample, (list, tuple))
                            and len(first_sample) > 0
                            and isinstance(first_sample[0], (int, torch.Tensor))
                        ):
                            buffer_encoded_samples = pad_sequence(
                                [torch.tensor(sample).reshape(-1) for sample in sampled_buffer],
                                batch_first=True,
                                padding_value=self.tokenizer.eos_token_id,
                            )
                        else:
                            # Pure text list, use existing tokenization logic
                            buffer_encoded_samples = []
                            for sample in sampled_buffer:
                                if not isinstance(sample, str):
                                    continue
                                if self.allowed_vocab is not None:
                                    if self.buffer_tokenization:
                                        buffer_encoded_sample = torch.tensor(
                                            self.tokenizer.encode(sample, add_special_tokens=False)
                                        ).reshape(-1)
                                    else:
                                        merged_chars = self.merge_chars_fast(
                                            sample, self.allowed_vocab
                                        )
                                        buffer_encoded_sample = torch.tensor(
                                            [
                                                self.tokenizer.encode(x, add_special_tokens=False)
                                                for x in merged_chars
                                            ]
                                        ).reshape(-1)
                                else:
                                    buffer_encoded_sample = torch.tensor(
                                        self.tokenizer.encode(sample, add_special_tokens=False)
                                    ).reshape(-1)

                                buffer_encoded_samples.append(buffer_encoded_sample)

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

        result = {
            "encoded_prompt": encoded_prompt,
            "buffer_encoded_sample": buffer_encoded_samples,
        }
        if self.scaffolds is not None:
            result["scaffold"] = self.scaffolds[index]
        return result


class BufferDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        buffer_sample_path: Optional[str] = None,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.2-1B",
        prompt_size: int = 1,
        total_size: int = 10000,
        train_size: float = 0.95,
        num_workers: int = 0,
        pin_memory: bool = True,
        add_prompt: bool = False,
        allowed_vocab_path: Optional[str] = None,
        n_samples: int = 8,
        buffer_tokenization: bool = False,
        scaffold_path: Optional[str] = None,
        scaffold_list: Optional[list] = None,
    ):
        """
        Data module for handling buffer data, which includes prompts and optional buffer samples.
        Args:
            data_path: Path to the file containing prompts.
            buffer_sample_path: Path to the buffer sample file.
            scaffold_path: Path to the scaffold list file (ignored if JSON data provides scaffold).
            scaffold_list: Optional scaffold list aligned with prompts.
            buffer_tokenization: If True, tokenizes the buffer samples. Else, construct tokens from vocabulary list
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_path = data_path
        self.train_size = train_size
        self.train_data = None
        self.val_data = None
        self.scaffold_path = scaffold_path
        self.scaffold_list = scaffold_list

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
        self.dataset_buffer = self.buffer_sample

    def prepare_data(self) -> None:
        pass

    def _load_prompts_and_scaffolds(self):
        data = None
        try:
            with open(self.data_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, TypeError):
            data = None

        if isinstance(data, list):
            if len(data) == 0:
                return [], None
            first = data[0]
            if not isinstance(first, dict):
                raise ValueError("JSON prompt file must contain a list of dict entries.")
            prompts = []
            scaffolds = []
            for item in data:
                if "prompt" not in item or "scaffold" not in item:
                    raise ValueError("JSON prompt entries must contain 'prompt' and 'scaffold'.")
                prompts.append(item["prompt"])
                scaffolds.append(item["scaffold"])
            return prompts, scaffolds

        with open(self.data_path) as f:
            prompts = [p.strip() for p in f.readlines()]
        return prompts, None

    def setup(self, stage):
        prompts, json_scaffolds = self._load_prompts_and_scaffolds()
        prompts = prompts[: self.prompt_size]

        if json_scaffolds is not None:
            scaffolds = json_scaffolds[: self.prompt_size]
        else:
            scaffolds = None
            if self.scaffold_list is not None:
                scaffolds = list(self.scaffold_list)
            elif self.scaffold_path and os.path.exists(self.scaffold_path):
                with open(self.scaffold_path) as f:
                    scaffolds = [line.strip() for line in f]
            if scaffolds is not None:
                scaffolds = scaffolds[: self.prompt_size]
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
            scaffolds=scaffolds,
        )
        self.val_data = BufferDataPipe(
            prompts,
            self.tokenizer,
            self.total_size - num_train,
            is_instruct=self.is_instruct,
            add_prompt=self.add_prompt,
            scaffolds=scaffolds,
        )
        self.test_data = BufferDataPipe(
            prompts,
            self.tokenizer,
            self.total_size - num_train,
            is_instruct=self.is_instruct,
            add_prompt=self.add_prompt,
            scaffolds=scaffolds,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )
