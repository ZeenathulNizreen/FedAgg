import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss

# === Config ===
fedavg_model_dir = "/root/FedAgg/qlora-FedAggregation/10/6"  # Replace with whichever dir has pytorch_model.bin
base_model = "allenai/OLMo-1B-hf"
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

# === Load model and tokenizer ===
print("✅ Loading full FedAvg model from pytorch_model.bin...")
model = AutoModelForCausalLM.from_pretrained(
    fedavg_model_dir,
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda()

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model.eval()

# === Evaluate ===
loss_fn = CrossEntropyLoss()
losses = []

print("🔍 Evaluating FedAvg model...")

for prompt in tqdm(eval_prompts):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        losses.append(loss)

# === Results ===
avg_loss = sum(losses) / len(losses)
print("\n📊 FedAvg Evaluation Results")
print(f"Average Loss: {avg_loss:.4f}")
for i, prompt in enumerate(eval_prompts):
    print(f"[{i+1}] Loss: {losses[i]:.4f} | Prompt: {prompt[:60]}...")

# === Save to file ===
os.makedirs("eval-logs", exist_ok=True)
with open("eval-logs/fedavg_loss_results.txt", "w") as f:
    f.write(f"Average Loss: {avg_loss:.4f}\n")
    for i, prompt in enumerate(eval_prompts):
        f.write(f"[{i+1}] Loss: {losses[i]:.4f} | Prompt: {prompt}\n")

print("✅ FedAvg loss results saved to eval-logs/fedavg_loss_results.txt")
