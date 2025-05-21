# export_lorahub.py
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file

BASE_MODEL = "allenai/OLMo-1B"
MERGED_ADAPTER_PATH = "../lorahub/lorahub_merged_adapter.safetensors"
ADAPTER_CONFIG_PATH = "../lorahub/lorahub_adapters/converted/client_2_model_epoch_0"

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)
peft_config = PeftConfig.from_pretrained(ADAPTER_CONFIG_PATH)
model = PeftModel(base_model, peft_config)

adapter_state = load_file(MERGED_ADAPTER_PATH, device="cpu")
set_peft_model_state_dict(model, adapter_state)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("exported_lorahub_model")
print(" LoRAHub model exported to: exported_lorahub_model/")
