import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation import evaluate_model  # Make sure evaluation.py exists in fed_utils
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

# === LOAD FINAL FEDAVG WEIGHTS ===
print("Loading final FedAvg adapter...")
state_dict = torch.load(ADAPTER_PATH, map_location=DEVICE)  # NOTE: map_location=DEVICE
model.load_state_dict(state_dict, strict=False)  # LoRA adapters are partial
model.eval()

# === LOAD TOKENIZER ===
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

# === EVALUATE ===
print("Evaluating...")
average_loss = evaluate_model(model, tokenizer, DATASET_PATH)
print(f"\n Final Average Loss: {average_loss}")
