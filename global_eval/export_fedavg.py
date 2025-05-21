import torch
from transformers import AutoModelForCausalLM

# Path to your merged FedAvg model
MODEL_PATH = "../qlora-FedAggregation/10/final_fedavg_model.safetensors"
BASE_MODEL = "allenai/OLMo-1B"

# Load base model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Load FedAvg weights
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)

# Save to HuggingFace format
model.save_pretrained("exported_fedavg_model")
print(" FedAvg model exported to: exported_fedavg_model/")
