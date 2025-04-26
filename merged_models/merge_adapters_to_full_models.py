from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil
import os

# Your LoRA adapter checkpoint directories
adapter_dirs = [
    "../qlora-FedAggregation/10/trainer_saved/local_output_2/checkpoint-7400/",
    "../qlora-FedAggregation/10/trainer_saved/local_output_3/checkpoint-7400/",
    "../qlora-FedAggregation/10/trainer_saved/local_output_4/checkpoint-7400/",
    "../qlora-FedAggregation/10/trainer_saved/local_output_5/checkpoint-7400/",
    "../qlora-FedAggregation/10/trainer_saved/local_output_8/checkpoint-7400/",
    "../qlora-FedAggregation/10/trainer_saved/local_output_9/checkpoint-7400/",
]

# Base model from Huggingface
base_model_name = "allenai/OLMo-1B"

# Output folder
output_root = "./merged_full_models/"
os.makedirs(output_root, exist_ok=True)

for idx, adapter_dir in enumerate(adapter_dirs, start=2):
    print(f"\n Merging adapter: {adapter_dir}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)

    # Load LoRA adapter
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_dir)

    # Merge LoRA into base model
    merged_model = model_with_adapter.merge_and_unload()

    # Save the fully merged model
    save_path = os.path.join(output_root, f"local_output_{idx}_merged")
    merged_model.save_pretrained(save_path)
    print(f" Saved merged model at: {save_path}")

print("\n All LoRA adapters merged into full models successfully!")