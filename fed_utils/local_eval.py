import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import os
import csv

# === CONFIG ===
BASE_MODEL_NAME = "allenai/OLMo-1B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "../data/10/global_test.json"
CLIENT_DIRS = [
    "../qlora-FedAggregation/10/trainer_saved/local_output_2/checkpoint-7400/",
    "../qlora-FedAggregation/10/trainer_saved/local_output_3/checkpoint-7400/",
    "../qlora-FedAggregation/10/trainer_saved/local_output_4/checkpoint-7400/",
    "../qlora-FedAggregation/10/trainer_saved/local_output_5/checkpoint-7400/",
    "../qlora-FedAggregation/10/trainer_saved/local_output_8/checkpoint-7400/",
    "../qlora-FedAggregation/10/trainer_saved/local_output_9/checkpoint-7400/",
]
# === LOAD TOKENIZER and DATASET ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Save results
results = []

for client_dir in CLIENT_DIRS:
    client_id = client_dir.split("_")[-1].split("/")[0]  # extract client number
    adapter_path = os.path.join(client_dir, "adapter_model.safetensors")

    if not os.path.exists(adapter_path):
        print(f"Adapter not found for Client {client_id}, skipping...")
        continue

    print(f"\n  Evaluating Client {client_id}...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True).to(DEVICE)
    model.eval()

    adapter_weights = torch.load(adapter_path, map_location=DEVICE)

    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, adapter_weights, adapter_name="default")

    losses = []
    with torch.no_grad():
        for data in dataset:
            inputs = tokenizer(data["response"], return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != "token_type_ids"}
            labels = inputs["input_ids"].clone()

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()
            losses.append(loss)

    avg_loss = sum(losses) / len(losses)
    print(f" Client {client_id} Average Loss: {avg_loss:.4f}")
    results.append((client_id, avg_loss))

    # OPTIONAL: Save a loss curve per client
    plt.figure()
    plt.plot(losses, label=f'Client {client_id}')
    plt.xlabel('Sample')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - Client {client_id}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"client_{client_id}_loss_curve.png")
    plt.close()

# Save all results to CSV
csv_path = "local_evaluation_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Client ID", "Average Loss"])
    writer.writerows(results)

print(f"\n Local Evaluation completed! Results saved to {csv_path}")
