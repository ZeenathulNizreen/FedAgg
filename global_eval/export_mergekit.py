import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file

MODEL_PATH = "../hf_compatible/hf_compatible_model/model.safetensors"
BASE_MODEL = "allenai/OLMo-1B"

# Load base model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Load MergeKit weights using safetensors
state_dict = load_file(MODEL_PATH)
model.load_state_dict(state_dict, strict=False)

# Save in HuggingFace format
model.save_pretrained("exported_mergekit_model")
print("MergeKit model exported to: exported_mergekit_model/")
