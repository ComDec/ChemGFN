import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


@dataclass
class CommonGenItem:
    concept_set_idx: int
    concepts: list[str]
    targets: list[str]


def _group_commongen(ds: HFDataset) -> list[CommonGenItem]:
    grouped: dict[int, dict[str, Any]] = defaultdict(lambda: {"concepts": None, "targets": []})
    for ex in ds:
        idx = int(ex["concept_set_idx"])
        concepts = [str(x) for x in ex.get("concepts", [])]
        target = str(ex.get("target", "") or "")

        slot = grouped[idx]
        if slot["concepts"] is None:
            slot["concepts"] = concepts
        slot_targets: list[str] = slot["targets"]
        if target:
            slot_targets.append(target)

    items: list[CommonGenItem] = []
    for idx in sorted(grouped.keys()):
        concepts = grouped[idx]["concepts"] or []
        targets = grouped[idx]["targets"] or []
        items.append(CommonGenItem(concept_set_idx=idx, concepts=concepts, targets=targets))
    return items


class CommonGenPromptDataset(Dataset):
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
        concept_str = ", ".join(concepts)
        prompt = self.prompt_template.format(concepts=concept_str)
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        return {
            "encoded_prompt": encoded_prompt,
            "buffer_encoded_sample": None,
            "scaffold": {
                "concept_set_idx": item.concept_set_idx,
                "concepts": concepts,
                # validation has references; HF test split has empty targets
                "references": [t for t in item.targets if t],
            },
        }


class CommonGenDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer_name: str,
        dataset_name: str = "allenai/common_gen",
        train_split: str = "train",
        val_split: str = "validation",
        test_split: str = "validation",
        prompt_template: str = (
            "Write one short, fluent sentence that uses ALL of these concepts: {concepts}.\n"
            "Sentence:"
        ),
        limit_train: Optional[int] = None,
        limit_val: Optional[int] = None,
        limit_test: Optional[int] = None,
        # When set, repeats the (possibly limited) train subset to reach this size.
        # Useful for small-debug subsets where per-epoch validation overhead is high.
        train_repeat_to: Optional[int] = None,
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

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    def prepare_data(self) -> None:
        # Ensure dataset is present in HF cache.
        load_dataset(self.dataset_name)

    def _limit_and_maybe_shuffle(
        self, items: list[CommonGenItem], limit: Optional[int], shuffle: bool
    ) -> list[CommonGenItem]:
        if shuffle:
            rng = random.Random(self.seed)
            items = list(items)
            rng.shuffle(items)
        if limit is not None:
            items = items[: max(0, int(limit))]
        return items

    def _repeat_to_size(
        self, items: list[CommonGenItem], target_size: Optional[int], shuffle: bool
    ) -> list[CommonGenItem]:
        if target_size is None:
            return items
        target_size = int(target_size)
        if target_size <= 0:
            return []
        if not items:
            return items
        if len(items) >= target_size:
            return list(items[:target_size])

        reps = target_size // len(items)
        rem = target_size % len(items)
        out = list(items) * reps + list(items[:rem])
        if shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(out)
        return out

    def setup(self, stage: str) -> None:
        ds = load_dataset(self.dataset_name)

        train_items = _group_commongen(ds[self.train_split])
        val_items = _group_commongen(ds[self.val_split])
        test_items = _group_commongen(ds[self.test_split])

        train_items = self._limit_and_maybe_shuffle(
            train_items, self.limit_train, self.shuffle_train
        )
        val_items = self._limit_and_maybe_shuffle(val_items, self.limit_val, False)
        test_items = self._limit_and_maybe_shuffle(test_items, self.limit_test, False)

        train_items = self._repeat_to_size(train_items, self.train_repeat_to, self.shuffle_train)

        self.train_data = CommonGenPromptDataset(train_items, self.tokenizer, self.prompt_template)
        self.val_data = CommonGenPromptDataset(val_items, self.tokenizer, self.prompt_template)
        self.test_data = CommonGenPromptDataset(test_items, self.tokenizer, self.prompt_template)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            shuffle=self.shuffle_train,
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
