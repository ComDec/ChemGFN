import json
import warnings

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from transformers import AutoTokenizer

from .components.prompts import *

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class SMILESDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
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
        with open(self.data_path) as f:
            data = json.load(f)
        prompts = [x["instruction"] for x in data]

        num_train = int(len(prompts) * self.train_size)
        self.train_data = SMILESDataPipe(prompts[:num_train], self.tokenizer)
        self.val_data = SMILESDataPipe(prompts[num_train:], self.tokenizer)

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


class SMILESDataPipe(MapDataPipe):
    def __init__(self, prompts, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts

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
        _prompt = (
            SMILES_PROMPTS_AHEAD
            + self.prompts[index]
            + BASE_PROMPTS_EXAMPLES
            + SMILES_PROMPTS_BEHIND
        )
        message = self.generate_message(_prompt)
        input_ids = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return input_ids
