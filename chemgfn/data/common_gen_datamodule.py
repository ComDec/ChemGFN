"""Data module for the CommonGen concept-to-sentence task.

Each example is a set of concepts that a generated sentence must cover. Concepts are grouped by
``concept_set_idx`` so that every prompt carries all human reference sentences for that concept
set; the references are passed downstream through the ``scaffold`` field and used by
:class:`~chemgfn.models.validators.CommonGenValidator` for the quality term and for BLEU.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

DEFAULT_PROMPT_TEMPLATE = (
    "Write one short, fluent sentence that uses ALL of these concepts: {concepts}.\nSentence:"
)


@dataclass
class CommonGenItem:
    """One CommonGen concept set together with its reference sentences."""

    concept_set_idx: int
    concepts: list[str]
    targets: list[str]


def group_by_concept_set(dataset: HFDataset) -> list[CommonGenItem]:
    """Collapse the flat ``(concepts, target)`` rows into one item per concept set."""
    grouped: dict[int, dict[str, Any]] = defaultdict(lambda: {"concepts": None, "targets": []})
    for example in dataset:
        idx = int(example["concept_set_idx"])
        slot = grouped[idx]
        if slot["concepts"] is None:
            slot["concepts"] = [str(c) for c in example.get("concepts", [])]
        target = str(example.get("target", "") or "")
        if target:
            slot["targets"].append(target)

    return [
        CommonGenItem(
            concept_set_idx=idx,
            concepts=grouped[idx]["concepts"] or [],
            targets=grouped[idx]["targets"] or [],
        )
        for idx in sorted(grouped)
    ]


class CommonGenPromptDataset(Dataset):
    """Renders each concept set into a tokenized prompt plus its reference sentences."""

    def __init__(
        self,
        items: list[CommonGenItem],
        tokenizer: AutoTokenizer,
        prompt_template: str,
    ) -> None:
        super().__init__()
        self.items = items
        self.tokenizer = tokenizer
        self.prompt_template = str(prompt_template)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        concepts = [c.strip() for c in item.concepts if str(c).strip()]
        prompt = self.prompt_template.format(concepts=", ".join(concepts))
        return {
            "encoded_prompt": self.tokenizer(prompt, return_tensors="pt")["input_ids"],
            "buffer_encoded_sample": None,
            "scaffold": {
                "concept_set_idx": item.concept_set_idx,
                "concepts": concepts,
                "references": [t for t in item.targets if t],
            },
        }


class CommonGenDataModule(LightningDataModule):
    """Loads CommonGen from the Hub and serves one concept set per step.

    The diagnostic setting used in the paper restricts training to a fixed handful of concept
    sets (``limit_train``) and repeats them to fill an epoch (``train_repeat_to``), so that
    termination behaviour can be tracked on a stable prompt set.

    Args:
        tokenizer_name: Hub id of the tokenizer, normally the policy's own tokenizer.
        dataset_name: Hub id of the CommonGen dataset.
        train_split: Split to draw training concept sets from.
        val_split: Split used for validation.
        test_split: Split used for testing. The Hub test split has no reference targets, so
            validation is the sensible default.
        prompt_template: Format string with a single ``{concepts}`` field.
        limit_train: Keep at most this many training concept sets.
        limit_val: Keep at most this many validation concept sets.
        limit_test: Keep at most this many test concept sets.
        train_repeat_to: Repeat the (possibly limited) training subset up to this many items.
        shuffle_train: Shuffle the training concept sets.
        seed: Seed for the shuffles above.
        num_workers: Dataloader worker processes.
        pin_memory: Pin dataloader memory.
    """

    def __init__(
        self,
        tokenizer_name: str,
        dataset_name: str = "allenai/common_gen",
        train_split: str = "train",
        val_split: str = "validation",
        test_split: str = "validation",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        limit_train: int | None = None,
        limit_val: int | None = None,
        limit_test: int | None = None,
        train_repeat_to: int | None = None,
        shuffle_train: bool = True,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = str(dataset_name)
        self.train_split = str(train_split)
        self.val_split = str(val_split)
        self.test_split = str(test_split)
        self.prompt_template = str(prompt_template)
        self.limit_train = None if limit_train is None else int(limit_train)
        self.limit_val = None if limit_val is None else int(limit_val)
        self.limit_test = None if limit_test is None else int(limit_test)
        self.train_repeat_to = None if train_repeat_to is None else int(train_repeat_to)
        self.shuffle_train = bool(shuffle_train)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_bos_token=False)

        self.train_data: Dataset | None = None
        self.val_data: Dataset | None = None
        self.test_data: Dataset | None = None

    def prepare_data(self) -> None:
        """Populate the Hugging Face cache before any worker starts."""
        load_dataset(self.dataset_name)

    def _take(
        self, items: list[CommonGenItem], limit: int | None, shuffle: bool
    ) -> list[CommonGenItem]:
        if shuffle:
            items = list(items)
            random.Random(self.seed).shuffle(items)
        if limit is not None:
            items = items[: max(0, limit)]
        return items

    def _repeat_to(
        self, items: list[CommonGenItem], target_size: int | None, shuffle: bool
    ) -> list[CommonGenItem]:
        if target_size is None or not items:
            return items
        if target_size <= 0:
            return []
        if len(items) >= target_size:
            return list(items[:target_size])

        reps, rem = divmod(target_size, len(items))
        out = list(items) * reps + list(items[:rem])
        if shuffle:
            random.Random(self.seed).shuffle(out)
        return out

    def setup(self, stage: str) -> None:
        """Build the per-split prompt datasets."""
        dataset = load_dataset(self.dataset_name)

        train_items = self._take(
            group_by_concept_set(dataset[self.train_split]), self.limit_train, self.shuffle_train
        )
        train_items = self._repeat_to(train_items, self.train_repeat_to, self.shuffle_train)
        val_items = self._take(
            group_by_concept_set(dataset[self.val_split]), self.limit_val, False
        )
        test_items = self._take(
            group_by_concept_set(dataset[self.test_split]), self.limit_test, False
        )

        self.train_data = CommonGenPromptDataset(train_items, self.tokenizer, self.prompt_template)
        self.val_data = CommonGenPromptDataset(val_items, self.tokenizer, self.prompt_template)
        self.test_data = CommonGenPromptDataset(test_items, self.tokenizer, self.prompt_template)

    def _loader(self, data: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            data,
            shuffle=shuffle,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_data, shuffle=self.shuffle_train)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_data)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_data)
