import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from peft import PeftModel, PeftConfig
from peft.utils.save_and_load import set_peft_model_state_dict

# === CONFIG ===
BASE_MODEL_NAME = "allenai/OLMo-1B"
MERGED_ADAPTER_PATH = "lorahub_merged_adapter.safetensors"
ADAPTER_CONFIG_PATH = "lorahub_adapters/converted/client_2_model_epoch_0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "eval-logs/lorahub_prompt_loss_plot.png"

# === PROMPTS ===
eval_prompts = [
    "Explain federated learning in simple terms.",
    "What is QLoRA and why is it useful?",
    "Describe the use of LoRA in low-resource settings.",
    "How does parameter-efficient tuning work?",
    "What are the privacy benefits of federated AI?",
    "List applications of LLMs in education.",
    "What makes instruction tuning effective?",
    "Summarize how MergeKit works with adapters.",
    "Define model merging in federated learning.",
    "Give an example of prompt-tuned adaptation."
]

# === LOAD MODEL ===
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True).to(DEVICE)

# === Load adapter config and wrap base model ===
peft_config = PeftConfig.from_pretrained(ADAPTER_CONFIG_PATH)  # directory only
model = PeftModel(base_model, peft_config)
model.eval()

# === Load merged weights ===
print("Loading merged LoRAHub adapter...")
adapter_state_dict = load_file(MERGED_ADAPTER_PATH, device="cpu")

set_peft_model_state_dict(model, adapter_state_dict)

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

# === Evaluate ===
print(" Evaluating prompts...")
losses = []
model.eval()

for prompt in tqdm(eval_prompts):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())

avg_loss = sum(losses) / len(losses)
print(f"\n LoRAHub Average Loss: {avg_loss:.4f}")

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(losses, marker='o', label="LoRAHub Prompt Loss")
plt.xticks(ticks=range(len(eval_prompts)), labels=eval_prompts, rotation=45, ha="right", fontsize=9)
plt.ylabel("Loss")
plt.title("LoRAHub Adapter - Prompt-wise Evaluation Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
os.makedirs("eval-logs", exist_ok=True)
plt.savefig(SAVE_PATH)
plt.show()

print(f" Plot saved at: {SAVE_PATH}")
