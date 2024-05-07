import os
import torch
import fire
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig, DataCollatorForSeq2Seq
import torch
import fire
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig, DataCollatorForSeq2Seq
from tqdm import tqdm

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from fed_utils import client_selection, GeneralClient
from utils.prompter import Prompter

datasets.utils.logging.set_verbosity_error()

def tokenize(tokenizer, cutoff_len, prompt, add_eos_token=True):
    if not isinstance(prompt, str) or prompt is None:
        prompt = ""
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding="max_length",
        return_tensors="pt",
    )
    if add_eos_token and result["input_ids"][0, -1] != tokenizer.eos_token_id:
        result["input_ids"] = torch.cat(
            (result["input_ids"], torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)), dim=1
        )
        result["attention_mask"] = torch.cat(
            (result["attention_mask"], torch.tensor([[1]], dtype=torch.long)), dim=1
        )
    result["labels"] = result["input_ids"].clone()
    return result

def fl_finetune(
    global_model='allenai/OLMo-1B',
    data_path='/root/FedAgg/data/10',
    output_dir='./qlora-FedAggregation',
    client_selection_strategy='random',
    client_selection_frac=0.1,
    num_communication_rounds=10,
    num_clients=10,
    local_batch_size=8,
    local_micro_batch_size=8,
    local_num_epochs=10,
    local_learning_rate=3e-4,
    local_val_set_size=0,
    local_save_steps=3,
    cutoff_len=512,
    lora_r=64,
    lora_alpha=16,
    lora_dropout=0.01,
    lora_target_modules=["att_proj"],
    train_on_inputs=True,
    group_by_length=True,
    prompt_template_name="olmo",
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit"
):
    if not os.path.exists(data_path):
        raise FileNotFoundError("Please generate the data files for each client")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        revision="main"
        revision="main"
    )

    tokenizer = AutoTokenizer.from_pretrained(global_model, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(global_model, use_fast=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    model.eval()

    model.eval()

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if hasattr(model.config, "gradient_checkpointing") and model.config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print("Model is prepared with QLoRA and ready for training.")

    for epoch in tqdm(range(num_communication_rounds)):
        selected_clients_set = client_selection(
            num_clients, client_selection_frac, client_selection_strategy, other_info=epoch
            num_clients, client_selection_frac, client_selection_strategy, other_info=epoch
        )

        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir)
            client.prepare_local_dataset(lambda prompt: tokenize(tokenizer, cutoff_len, prompt, True), local_val_set_size)
            client.build_local_trainer(
                tokenizer,
                local_micro_batch_size,
                local_num_epochs,
                local_learning_rate,
                group_by_length
            )
                local_micro_batch_size,
                local_num_epochs,
                local_learning_rate,
                group_by_length
            )
            client.initiate_local_training()
            client.train()
            client.terminate_local_training(epoch)

    model_path = os.path.join(output_dir, "aggregated_model.bin")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
            client.terminate_local_training(epoch)

    model_path = os.path.join(output_dir, "aggregated_model.bin")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    fire.Fire(fl_finetune)
