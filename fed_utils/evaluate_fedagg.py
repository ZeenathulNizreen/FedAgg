import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from evaluation import evaluate_model

# Set paths
ADAPTER_PATH = "./qlora-FedAggregation/10/final_fedavg_model.safetensors"
DATASET_PATH = "./data/10/global_test.json"
BASE_MODEL_NAME = "allenai/OLMo-1B"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
model = model.to(DEVICE)

# 2. Load the adapter properly (without set_peft_model_state_dict)
print("Loading final FedAvg adapter...")
adapter_weights = torch.load(ADAPTER_PATH, map_location="cuda")

# 3. Apply adapter manually
model.load_state_dict(adapter_weights, strict=False)

# 4. Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

# 5. Evaluate
print("Evaluating...")
average_loss = evaluate_model(model, tokenizer, DATASET_PATH)
print(f"Final Average Loss: {average_loss}")

