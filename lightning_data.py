import json
import warnings

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class PromptDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer,
        train_size=0.95,
        limit_prompts=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        with open(self.hparams.data_path, "r") as f:
            prompts = f.readlines()
        prompts = [line.rstrip("\n") for line in prompts]
        if self.hparams.limit_prompts is not None:
            prompts = prompts[: self.hparams.limit_prompts]
        num_train = int(len(prompts) * self.hparams.train_size)
        self.train_data = PromptDataPipe(prompts[:num_train], self.tokenizer)
        self.val_data = PromptDataPipe(prompts[num_train:], self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=None, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=None, num_workers=8)


class PromptDataPipe(MapDataPipe):
    def __init__(self, prompts, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompt = self.tokenizer(
            self.prompts[index],
            return_tensors="pt",
        )["input_ids"]
        return prompt


class SMILESDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer,
        train_size=0.95,
        limit_prompts=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        with open(self.hparams.data_path, "r") as f:
            data = json.load(f)
        prompts = [x["instruction"] for x in data]
        if self.hparams.limit_prompts is not None:
            prompts = prompts[: self.hparams.limit_prompts]
        num_train = int(len(prompts) * self.hparams.train_size)
        self.train_data = SMILESDataPipe(prompts[:num_train], self.tokenizer)
        self.val_data = SMILESDataPipe(prompts[num_train:], self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=None, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=None, num_workers=8)


PROMPTS_AHEAD = "You must follow the following rules to generate SMILES:\
1. **Basic Structure:**\
   - SMILES is a line notation using printable characters without spaces.\
   - Represents molecules and reactions.\
2. **Atoms Representation:**\
   - Atoms are represented by their atomic symbols.\
   - Non-hydrogen atoms are enclosed in square brackets, e.g., [C], [O].\
   - Elements in the organic subset (B, C, N, O, P, S, F, Cl, Br, I) can be written without brackets if they conform to normal valences.\
3. **Hydrogens and Charges:**\
   - Attached hydrogens are shown by H followed by a digit (optional).\
   - Formal charges are shown by + or -, followed by a digit (optional).\
   - Example: [Fe+++] is the same as [Fe+3].\
4. **Bonds Representation:**\
   - Single: -\
   - Double: =\
   - Triple: #\
   - Aromatic: :\
   - Adjacent atoms are assumed to be connected by a single or aromatic bond if no bond symbol is present.\
5. **Branches and Cyclic Structures:**\
   - Branches are enclosed in parentheses and can be nested.\
   - Cyclic structures are represented by breaking one bond in each ring and using digits to indicate ring closure.\
6. **Disconnected Compounds:**\
   - Written as individual structures separated by a period (.)\
7. **Isomer and Chirality Specifications:**\
   - Chirality is indicated by @ or @@ following the atomic symbol.\
   - @ indicates anticlockwise; @@ indicates clockwise.\
   - Absence of chirality specification means chirality is not specified.\
8. **Isotopic Specifications:**\
   - Indicated by preceding the atomic symbol with the atomic mass number inside brackets.\
9. **Double Bond Configuration:**\
   - Directional bonds are shown by / and \ to indicate relative directionality.\
10. **General Rules:**\
    - Any valid order of SMILES notation is acceptable.\
    - Implicit hydrogens are assumed unless explicitly stated.\
    - Matching pairs of digits indicate bonded atoms, and adjacent atoms separated by a period (.) are not bonded.\
By following these simplified rules, you can effectively use SMILES notation to represent molecular structures."

PROMPTS_EXAMPLES = "There are several examples to convert IUPAC names into SMILES notation.\
                    The IUPAC name 6-methyl-5-propan-2-yl-3,4-dihydro-2H-1,4-thiazine have its SMILES is CC1=C(NCCS1)C(C)C \
                    The IUPAC chemical name 4-tert-butyl-6-pyrrolidin-3-ylmorpholin-3-one into its SMILES form is CC(C)(C)N1CC(OCC1=O)C2CCNC2 \
                    The SMILES version of the IUPAC name 2-(4-propan-2-ylphenyl)-1,3-thiazole is CC(C)C1=CC=C(C=C1)C2=NC=CS2"

PROMPTS_BEHIND = "You must give only one SMILES string as output without any additional information, explanation, context, and characters."


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
                "content": "You are an expert on cheminformatics. You are here to help the user generate valid molecules.",
            },
            {"role": "user", "content": f"{question}"},
        ]

    def __getitem__(self, index):
        _prompt = PROMPTS_AHEAD + self.prompts[index] + PROMPTS_EXAMPLES + PROMPTS_BEHIND
        message = self.generate_message(_prompt)
        input_ids = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return input_ids
