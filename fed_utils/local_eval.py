import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, GPT2Tokenizer
import matplotlib.pyplot as plt

# Disable warnings from transformers library
logging.set_verbosity_error()

class EvalDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length, return_tensors="pt")
        return encodings.input_ids[0], encodings.attention_mask[0]

def load_evaluation_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def evaluate_model(model, eval_loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            losses.append(loss.item())
    
    avg_loss = sum(losses) / len(losses)
    return avg_loss, losses

# Custom tokenizer class for OLMo
class OLMoTokenizer(GPT2Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization if necessary

# Register the custom tokenizer
AutoTokenizer.register("OLMoTokenizer", OLMoTokenizer)

# Load the global test set for evaluation
eval_data = load_evaluation_data('../data/10/global_test.json')
eval_responses = [item['response'] for item in eval_data]  # Adjust based on your JSON structure

# Initialize the tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)
except ValueError as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

eval_dataset = EvalDataset(eval_responses, tokenizer)
eval_loader = DataLoader(eval_dataset, batch_size=4)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to client models
client_model_paths = [
    '../qlora-FedAggregation/10/trainer_saved/local_output_2/checkpoint-400',  # Replace with the actual path to each client's model
    '../qlora-FedAggregation/10/trainer_saved/local_output_3/checkpoint-400',
    '../qlora-FedAggregation/10/trainer_saved/local_output_4/checkpoint-400',
    '../qlora-FedAggregation/10/trainer_saved/local_output_5/checkpoint-400',
    '../qlora-FedAggregation/10/trainer_saved/local_output_8/checkpoint-400',
    '../qlora-FedAggregation/10/trainer_saved/local_output_9/checkpoint-400'
]

client_evaluation_results = []

for client_id, model_path in enumerate(client_model_paths):
    print(f"Evaluating Client {client_id}'s model...")
    if not os.path.isdir(model_path):
        print(f"Model path {model_path} does not exist or is not a directory.")
        continue
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
        avg_loss, losses = evaluate_model(model, eval_loader, device)
        client_evaluation_results.append({
            'client_id': client_id,
            'avg_loss': avg_loss,
            'losses': losses
        })
        print(f"Client {client_id} Average Loss: {avg_loss}")
    except Exception as e:
        print(f"Error loading model for Client {client_id} from path {model_path}: {e}")

# Plot the loss curves for each client
for result in client_evaluation_results:
    plt.plot(result['losses'], label=f'Client {result["client_id"]}')

if client_evaluation_results:
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Evaluation Loss Curves for Each Client')
    plt.legend()
    plt.show()
else:
    print("No valid models were evaluated.")
