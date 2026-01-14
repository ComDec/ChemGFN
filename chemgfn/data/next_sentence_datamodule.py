import os
from typing import Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class PromptContinuationDataset(Dataset):
    """Minimal prompt-only dataset for next-sentence continuation.

    Each item is a single prompt tensor shaped (1, L). We keep the output schema
    consistent with existing dataloaders used by ``ChemGFNModule`` by returning a
    dictionary that contains ``encoded_prompt`` and ``buffer_encoded_sample`` keys.
    """

    def __init__(self, prompts: list[str], tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self.prompts = prompts
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | None]:
        encoded_prompt = self.tokenizer(self.prompts[index], return_tensors="pt")["input_ids"]
        return {
            "encoded_prompt": encoded_prompt,
            "buffer_encoded_sample": None,
        }


class NextSentenceDataModule(LightningDataModule):
    """DataModule for next-sentence continuation prompts.

    This mirrors the original ``next_sentence`` experiment: a flat text file with one
    prompt per line, a configurable train/val split, and no dataset-side buffer.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        train_size: float = 0.95,
        limit_prompts: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.train_size = float(train_size)
        self.limit_prompts = limit_prompts
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_bos_token=False)

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    def _load_prompts(self) -> list[str]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Prompt file not found: {self.data_path}")
        with open(self.data_path) as f:
            prompts = [line.rstrip("\n") for line in f]
        if self.limit_prompts is not None:
            prompts = prompts[: self.limit_prompts]
        return prompts

    def setup(self, stage: str) -> None:
        prompts = self._load_prompts()
        num_train = int(len(prompts) * self.train_size)
        train_prompts = prompts[:num_train]
        val_prompts = prompts[num_train:]

        self.train_data = PromptContinuationDataset(train_prompts, self.tokenizer)
        self.val_data = PromptContinuationDataset(val_prompts, self.tokenizer)
        self.test_data = PromptContinuationDataset(val_prompts, self.tokenizer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
