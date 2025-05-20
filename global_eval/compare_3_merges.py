import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from peft.utils.save_and_load import set_peft_model_state_dict

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_NAME = "allenai/OLMo-1B"
TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

PROMPTS = [
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

ADAPTERS = {
    "FedAvg": {
        "path": "../qlora-FedAggregation/10/final_fedavg_model.safetensors",
        "config_file": "../lorahub/lorahub_adapters/converted/client_2_model_epoch_0/adapter_config.json"
    },
    "MergeKit": {
        "path": "../mergekit_models/merged_adapter.safetensors",
        "config_file": "../lorahub/lorahub_adapters/converted/client_2_model_epoch_0/adapter_config.json"
    },
    "LoRAHub": {
        "path": "../lorahub/lorahub_merged_adapter.safetensors",
        "config_file": "../lorahub/lorahub_adapters/converted/client_2_model_epoch_0/adapter_config.json"
    }
}


def evaluate_adapter(adapter_path, config_path=None):
    print(f"Evaluating model at {adapter_path}...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

    if config_path and os.path.exists(config_path):
        # LoRA adapter: use safetensors + PEFT
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True).to(DEVICE)
        peft_config = PeftConfig.from_pretrained(os.path.dirname(config_path))
        model = PeftModel(base_model, peft_config)
        adapter_state = load_file(adapter_path, device="cpu")
        set_peft_model_state_dict(model, adapter_state)
        model = model.to(DEVICE)
    else:
        # Full model: use torch.load()
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True).to(DEVICE)
        state_dict = torch.load(adapter_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    losses = []

    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != "token_type_ids"}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())

    return losses

# === Evaluate All Adapters ===
all_losses = {}
for name, info in ADAPTERS.items():
    print(f"Evaluating {name}...")
    all_losses[name] = evaluate_adapter(info["path"], info["config_file"])

# === Plot Comparison Bar Chart ===
x = range(len(PROMPTS))
width = 0.25

plt.figure(figsize=(16, 6))
plt.bar([i - width for i in x], all_losses["FedAvg"], width, label="FedAvg")
plt.bar(x, all_losses["MergeKit"], width, label="MergeKit")
plt.bar([i + width for i in x], all_losses["LoRAHub"], width, label="LoRAHub")

plt.xticks(x, PROMPTS, rotation=45, ha="right", fontsize=9)
plt.ylabel("Loss")
plt.title("Prompt-wise Comparison of FedAvg, MergeKit, and LoRAHub")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()

os.makedirs("eval-logs", exist_ok=True)
plt.savefig("eval-logs/comparison_prompt_loss.png")
plt.show()
