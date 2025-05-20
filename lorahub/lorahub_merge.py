from lorahub import load_base_model_and_lora_modules, get_final_weights
from safetensors.torch import save_file

# === CONFIG ===
import glob

adapter_paths = sorted(glob.glob("lorahub_adapters/converted/*"))


output_path = "lorahub_merged_adapter.safetensors"

# === Load adapter states directly ===
print("✅ Loading LoRA adapters...")
model, lora_module_list, cache = load_base_model_and_lora_modules(adapter_paths)

# === Merge
print("🔄 Merging using LoRAHub...")
merged_weights = get_final_weights(list(cache.keys()), lora_module_list, cache)

# === Save result
save_file(merged_weights, output_path)
print(f"✅ Merged adapter saved at: {output_path}")
