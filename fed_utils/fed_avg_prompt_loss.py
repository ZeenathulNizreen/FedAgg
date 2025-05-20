import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss

# === CONFIG ===
ADAPTER_PATH = "../qlora-FedAggregation/10/final_fedavg_model.safetensors"
BASE_MODEL_NAME = "allenai/OLMo-1B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "eval-logs/fedavg_prompt_loss_plot.png"

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
print(" Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True).to(DEVICE)

print(" Loading FedAvg weights...")
state_dict = torch.load(ADAPTER_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

# === EVALUATE ===
print(" Evaluating prompts...")
loss_fn = CrossEntropyLoss()
losses = []

for prompt in tqdm(eval_prompts):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        losses.append(loss)

avg_loss = sum(losses) / len(losses)
print(f"\n Average Loss: {avg_loss:.4f}")

# === PLOT ===
plt.figure(figsize=(14, 6))
bars = plt.bar(range(len(eval_prompts)), losses, tick_label=eval_prompts)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.ylabel("Loss")
plt.title("FedAvg Model Prompt-wise Evaluation Loss")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bars with loss values
for bar, loss in zip(bars, losses):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{loss:.2f}", ha="center", va="bottom")

# Save
os.makedirs("eval-logs", exist_ok=True)
plt.tight_layout()
plt.savefig(SAVE_PATH)
plt.show()

print(f" Plot saved to: {SAVE_PATH}")
