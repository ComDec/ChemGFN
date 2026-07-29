"""Prompt and dataset-buffer datamodule shared by the SMILES, Expr24 and AMP tasks."""

from __future__ import annotations

import json
import os
import random
import warnings
from typing import Any, Optional, Sequence

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class BufferDataPipe(Dataset):
    """Map-style dataset yielding one tokenized prompt (plus dataset-buffer samples) per item.

    Prompts are drawn with replacement so that a single conditioning prompt can back an
    arbitrary number of training steps. When a dataset buffer is supplied, each item also
    carries a random slice of it; the training loop mixes those pre-tokenized trajectories
    into the on-policy batch.

    Args:
        prompts: Prompt strings to sample from.
        tokenizer: Tokenizer used to encode the prompt.
        total_size: Number of items in the dataset.
        buffer_sample: Optional ``(N, T)`` int64 tensor of pre-tokenized dataset trajectories.
        n_samples: Number of buffer rows drawn per item.
        scaffolds: Optional per-prompt scaffold, aligned with ``prompts``.
    """

    def __init__(
        self,
        prompts: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        total_size: int = 10000,
        buffer_sample: Optional[torch.Tensor] = None,
        n_samples: int = 4,
        scaffolds: Optional[Sequence[Any]] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.total_size = total_size

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
        self.n_samples = n_samples

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded_prompt = self.tokenizer(self.prompts[index], return_tensors="pt")["input_ids"]

        buffer_encoded_sample = None
        if self.buffer_sample is not None:
            num_samples = min(self.n_samples, len(self.buffer_sample))
            indices = torch.randperm(len(self.buffer_sample))[:num_samples]
            buffer_encoded_sample = self.buffer_sample[indices]

        item: dict[str, Any] = {
            "encoded_prompt": encoded_prompt,
            "buffer_encoded_sample": buffer_encoded_sample,
        }
        if self.scaffolds is not None:
            item["scaffold"] = self.scaffolds[index]
        return item


class BufferDataModule(LightningDataModule):
    """Lightning datamodule that serves prompts and an optional dataset buffer.

    ``data_path`` is either a plain text file with one prompt per line, or a JSON file holding
    a list of ``{"prompt": ..., "scaffold": ...}`` entries. A scaffold is the template the
    generated fragment is substituted into (the ``*`` placeholder in the SMILES task); tasks
    without a template leave it unset.

    Args:
        data_path: Path to the prompt file (``.txt`` or ``.json``).
        buffer_sample_path: Optional path to a ``(N, T)`` int64 tensor of pre-tokenized
            trajectories used as the dataset buffer.
        tokenizer_name: Hugging Face tokenizer name or path.
        prompt_size: Number of prompts kept from the file.
        total_size: Combined number of train and validation items.
        train_size: Fraction of ``total_size`` allocated to training.
        num_workers: Dataloader worker processes.
        pin_memory: Whether dataloaders pin host memory.
        n_samples: Number of dataset-buffer rows drawn per item.
        scaffold_list: Optional scaffolds aligned with the prompts, used when the prompt file
            is plain text and therefore carries no scaffold of its own.
    """

    def __init__(
        self,
        data_path: str,
        buffer_sample_path: Optional[str] = None,
        tokenizer_name: str = "meta-llama/Llama-3.2-1B",
        prompt_size: int = 1,
        total_size: int = 10000,
        train_size: float = 0.95,
        num_workers: int = 0,
        pin_memory: bool = True,
        n_samples: int = 8,
        scaffold_list: Optional[Sequence[Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_path = data_path
        self.train_size = train_size
        self.scaffold_list = scaffold_list

        self.prompt_size = prompt_size
        self.total_size = total_size
        self.n_samples = n_samples

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.train_data: Optional[BufferDataPipe] = None
        self.val_data: Optional[BufferDataPipe] = None
        self.test_data: Optional[BufferDataPipe] = None

        self.buffer_sample: Optional[torch.Tensor] = None
        if buffer_sample_path and os.path.exists(buffer_sample_path):
            buffer_sample = torch.load(buffer_sample_path)
            if not isinstance(buffer_sample, torch.Tensor) or buffer_sample.dim() != 2:
                raise ValueError(
                    f"Dataset buffer {buffer_sample_path} must be a 2-D tensor of token ids."
                )
            self.buffer_sample = buffer_sample

    def prepare_data(self) -> None:
        """No download step is required; prompts and buffers ship with the repository."""

    def _load_prompts_and_scaffolds(self) -> tuple[list[str], Optional[list[Any]]]:
        """Read the prompt file, returning prompts and per-prompt scaffolds when present."""
        try:
            with open(self.data_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, TypeError):
            data = None

        if isinstance(data, list):
            if len(data) == 0:
                return [], None
            if not isinstance(data[0], dict):
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

    def setup(self, stage: Optional[str] = None) -> None:
        """Build the train, validation and test datasets."""
        prompts, json_scaffolds = self._load_prompts_and_scaffolds()
        prompts = prompts[: self.prompt_size]

        if json_scaffolds is not None:
            scaffolds = json_scaffolds[: self.prompt_size]
        elif self.scaffold_list is not None:
            scaffolds = list(self.scaffold_list)[: self.prompt_size]
        else:
            scaffolds = None

        num_train = int(self.total_size * self.train_size)

        self.train_data = BufferDataPipe(
            prompts,
            self.tokenizer,
            num_train,
            buffer_sample=self.buffer_sample,
            n_samples=self.n_samples,
            scaffolds=scaffolds,
        )
        self.val_data = BufferDataPipe(
            prompts,
            self.tokenizer,
            self.total_size - num_train,
            scaffolds=scaffolds,
        )
        self.test_data = BufferDataPipe(
            prompts,
            self.tokenizer,
            self.total_size - num_train,
            scaffolds=scaffolds,
        )

    def _dataloader(self, dataset: BufferDataPipe, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the shuffled training dataloader."""
        return self._dataloader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return self._dataloader(self.val_data)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return self._dataloader(self.test_data)
