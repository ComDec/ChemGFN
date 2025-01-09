import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)


class Llama3Model(nn.Module):
    def __init__(
        self,
        model_name: str,
        use_bnb: bool = False,
        bnb_config: BitsAndBytesConfig = None,
        lora_config: LoraConfig = None,
    ):
        super().__init__()

        if not use_bnb:
            print("You set use_bnb to False. bnb_config will be overwritten.")
            bnb_config = None

        self.model_name = model_name

        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )

        # Prepare model for k-bit training
        if use_bnb:
            self.model = prepare_model_for_kbit_training(
                _model,
                use_gradient_checkpointing=False,  # Doesn't save memory when generating autoregressively compared to caching
            )

        # Wrap using Lora
        self.model = get_peft_model(_model, lora_config)

        # Remove dropout
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def forward(self, input_ids, past_key_values=None, **kwargs):
        return self.model(input_ids, past_key_values=past_key_values, **kwargs)
