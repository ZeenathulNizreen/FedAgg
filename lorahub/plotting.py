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
SAVE_PATH = "eval-logs/lorahub_prompt_loss_bar_chart.png"

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

# === Wrap with PEFT ===
peft_config = PeftConfig.from_pretrained(ADAPTER_CONFIG_PATH)
model = PeftModel(base_model, peft_config)
model.eval()

# === Load adapter weights ===
print("Loading merged LoRAHub adapter...")
adapter_state_dict = load_file(MERGED_ADAPTER_PATH, device="cpu")
set_peft_model_state_dict(model, adapter_state_dict)

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

# === Evaluation ===
print("Evaluating prompts...")
losses = []

for prompt in tqdm(eval_prompts):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())

avg_loss = sum(losses) / len(losses)
print(f"\nLoRAHub Average Loss: {avg_loss:.4f}")

# === BAR PLOT ===
plt.figure(figsize=(14, 6))
bars = plt.bar(range(len(losses)), losses)
plt.xticks(range(len(eval_prompts)), eval_prompts, rotation=45, ha="right", fontsize=9)
plt.ylabel("Loss")
plt.title("LoRAHub Adapter - Prompt-wise Evaluation Loss (Bar Chart)")
plt.grid(axis='y')

# Annotate bars
for bar, loss in zip(bars, losses):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{loss:.2f}", ha='center', va='bottom')

os.makedirs("eval-logs", exist_ok=True)
plt.tight_layout()
plt.savefig(SAVE_PATH)
plt.show()

print(f" Bar chart saved at: {SAVE_PATH}")
