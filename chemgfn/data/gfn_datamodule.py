import json
import random
import warnings

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .components.prompts import *

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class SMILESDataPipe(Dataset):
    def __init__(self, prompts, tokenizer, total_size: int = 10000) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.total_size = total_size

        # Generate prompts by randomly sampling with replacement
        self.prompts = random.choices(prompts, k=self.total_size)

    def __len__(self):
        return len(self.prompts)

    def generate_message(self, question):
        return [
            {
                "role": "system",
                "content": "You are an expert on cheminformatics and helpful assistant. You are here to help the user generate valid molecules representation.",
            },
            {"role": "user", "content": f"{question}"},
        ]

    def __getitem__(self, index):
        # _prompt = (
        #     SMILES_PROMPTS_AHEAD
        #     + self.prompts[index]
        #     + BASE_PROMPTS_EXAMPLES
        #     + SMILES_PROMPTS_BEHIND
        # )

        # _prompt = (self.prompts[index] + SMILES_PROMPTS_BEHIND)

        # message = self.generate_message(_prompt)
        # encoded_prompt = self.tokenizer.apply_chat_template(
        #     message,
        #     add_generation_prompt=True,
        #     return_tensors="pt",
        # )

        encoded_prompt = self.tokenizer(self.prompts[index], return_tensors="pt")

        return {
            "encoded_prompt": encoded_prompt["input_ids"],
        }


class SMILESDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
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

        self.train_data = SMILESDataPipe(prompts, self.tokenizer, num_train)
        self.val_data = SMILESDataPipe(prompts, self.tokenizer, self.total_size - num_train)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class ParenthesesDataSet(Dataset):
    def __init__(self, prompts, tokenizer, total_size: int = 10000) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.total_size = total_size

        # Generate prompts by randomly sampling with replacement
        self.prompts = random.choices(prompts, k=self.total_size)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        prompts = self.prompts[index]
        encoded_prompt = self.tokenizer(
            prompts,
            return_tensors="pt",
        )
        return {
            "encoded_prompt": encoded_prompt["input_ids"],
        }


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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
