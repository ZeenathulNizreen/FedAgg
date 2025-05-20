from lorahub import load_base_model_and_lora_modules, get_final_weights
from safetensors.torch import save_file

# === CONFIG ===
import glob

adapter_paths = sorted(glob.glob("lorahub_adapters/converted/*"))


output_path = "lorahub_merged_adapter.safetensors"

# === Load adapter states directly ===
print("Loading LoRA adapters...")
model, tokenizer, cache = load_base_model_and_lora_modules(adapter_paths)
lora_module_list = list(cache.keys())  # Now this is correctly the list of adapter names


# === Merge
print(" Merging using LoRAHub...")
weights = [1.0] * len(lora_module_list)
merged_weights = get_final_weights(weights, lora_module_list, cache)


# === Save result
save_file(merged_weights, output_path)
print(f" Merged adapter saved at: {output_path}")
