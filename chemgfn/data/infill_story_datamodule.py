import os
from typing import Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class StoryInfillDataset(Dataset):
    """Prompt-only dataset with reference completions for infilling."""

    def __init__(self, examples: list[dict[str, str]], tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | None | str]:
        item = self.examples[index]
        encoded_prompt = self.tokenizer(item["prompt"], return_tensors="pt")["input_ids"]
        return {
            "encoded_prompt": encoded_prompt,
            "buffer_encoded_sample": None,
            "scaffold": item["reference"],
        }


class StoryInfillDataModule(LightningDataModule):
    """DataModule for ROCStories-style infilling prompts."""

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        train_start: int = 100,
        train_end: Optional[int] = 1000,
        eval_start: int = 0,
        eval_end: Optional[int] = 100,
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.train_start = int(train_start)
        self.train_end = train_end
        self.eval_start = int(eval_start)
        self.eval_end = eval_end
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_bos_token=False)

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    def _load_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"ROCStories CSV not found at {self.data_path}. "
                "Download and place `stories.csv` at this path."
            )
        return pd.read_csv(self.data_path)

    def _build_examples(
        self, df: pd.DataFrame, start: int, end: Optional[int]
    ) -> list[dict[str, str]]:
        end_idx = len(df) if end is None else min(int(end), len(df))
        start_idx = max(0, int(start))
        examples = []
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            beginning = f"{row['sentence1']} {row['sentence2']} {row['sentence3']}"
            ending = row["sentence5"]
            middle = f" {str(row['sentence4'])[:-1]}"
            prompt = f"Beginning: {beginning}\nEnding: {ending}\nMiddle:"
            examples.append({"prompt": prompt, "reference": middle})
        return examples

    def setup(self, stage: str) -> None:
        df = self._load_csv()
        train_examples = self._build_examples(df, self.train_start, self.train_end)
        eval_examples = self._build_examples(df, self.eval_start, self.eval_end)

        if self.max_train_samples is not None:
            train_examples = train_examples[: int(self.max_train_samples)]
        if self.max_eval_samples is not None:
            eval_examples = eval_examples[: int(self.max_eval_samples)]

        self.train_data = StoryInfillDataset(train_examples, self.tokenizer)
        self.val_data = StoryInfillDataset(eval_examples, self.tokenizer)
        self.test_data = StoryInfillDataset(eval_examples, self.tokenizer)

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
