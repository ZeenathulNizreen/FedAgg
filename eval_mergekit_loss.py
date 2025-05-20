import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.nn import CrossEntropyLoss

# === Config ===
base_model = "allenai/OLMo-1B-hf"
adapter_dir = "hf_compatible/hf_compatible_model"  # where merged adapter is
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
    # Add more prompts from your 80 held-out set
]

# === Load model and tokenizer ===
print(" Loading model and adapter...")
model = AutoModelForCausalLM.from_pretrained(adapter_dir, torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(base_model)
model.eval()


# === Loss function ===
loss_fn = CrossEntropyLoss()
losses = []

print(" Evaluating model on held-out prompts...")

for prompt in tqdm(eval_prompts):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        losses.append(loss)

# === Summary ===
avg_loss = sum(losses) / len(losses)
print("\n Evaluation Results")
print(f"Average Loss: {avg_loss:.4f}")
for i, prompt in enumerate(eval_prompts):
    print(f"[{i+1}] Loss: {losses[i]:.4f} | Prompt: {prompt[:60]}...")

# === Save results ===
os.makedirs("eval-logs", exist_ok=True)
with open("eval-logs/merged_loss_results.txt", "w") as f:
    f.write(f"Average Loss: {avg_loss:.4f}\n")
    for i, prompt in enumerate(eval_prompts):
        f.write(f"[{i+1}] Loss: {losses[i]:.4f} | Prompt: {prompt}\n")

print("Results saved to eval-logs/merged_loss_results.txt")
