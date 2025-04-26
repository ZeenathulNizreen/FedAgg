import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
import os

# === CONFIG ===
ADAPTER_PATH = "../qlora-FedAggregation/10/final_fedavg_model.safetensors"
DATASET_PATH = "../data/10/global_test.json"
BASE_MODEL_NAME = "allenai/OLMo-1B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CHECK FILES EXIST ===
if not os.path.exists(ADAPTER_PATH):
    raise FileNotFoundError(f"Adapter file not found at {ADAPTER_PATH}")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")

# === LOAD BASE MODEL ===
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True).to(DEVICE)

# === LOAD FINAL FedAvg Weights ===
print("Loading final FedAvg model weights...")
state_dict = torch.load(ADAPTER_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.eval()

# === LOAD TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

# === LOAD DATA ===
print("Loading evaluation data...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

losses = []
with torch.no_grad():
    for i, data in enumerate(dataset):
        inputs = tokenizer(data["response"], return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != "token_type_ids"}
        labels = inputs["input_ids"].clone()

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()
        losses.append(loss)
        print(f"Sample {i+1} Loss: {loss:.4f}")

# === PLOT ===
average_loss = sum(losses) / len(losses)
print(f"\n Final Average Loss: {average_loss}")

plt.plot(losses, label="Per-sample loss")
plt.xlabel("Sample")
plt.ylabel("Loss")
plt.title("Final FedAvg Model Evaluation Loss")
plt.legend()
plt.grid(True)
plt.savefig("fedavg_loss_curve.png")
plt.show()