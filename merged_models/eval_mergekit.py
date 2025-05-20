import os
import torch
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths
base_model = "allenai/OLMo-1B-hf"  #  Use this HF-compatible version
weights_path = "hf_compatible_model/model.safetensors"  #   federated merged model
output_dir = "./hf_compatible_model"

# Load model architecture
model = AutoModelForCausalLM.from_pretrained(base_model)

# Load federated weights
state_dict = safetensors.torch.load_file(weights_path)
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print(" Missing keys:", missing)
print("Unexpected keys:", unexpected)

# Save for future use
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir, safe_serialization=False)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(output_dir)

print(" Merged model saved for reuse at:", output_dir)

inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
