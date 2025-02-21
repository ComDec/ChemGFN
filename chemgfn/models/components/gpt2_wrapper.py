import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)


class GPT2Model(nn.Module):
    def __init__(
        self,
        model_name: str,
        lora_config: LoraConfig = None,
    ):
        super().__init__()

        self.model_name = model_name

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )

        # Wrap using Lora
        self.model = get_peft_model(self.model, lora_config)

        # Remove dropout
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def forward(self, input_ids, past_key_values=None, **kwargs):
        return self.model(input_ids, past_key_values=past_key_values, **kwargs)
