import os
from typing import List
from tqdm import tqdm
import fire
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, prepare_model_for_kbit_training
from fed_utils import client_selection, GeneralClient
import datasets
from utils.prompter import Prompter
from trl import SFTTrainer

datasets.utils.logging.set_verbosity_error()

def fl_finetune(
        # model/data params
        global_model: str = 'allenai/OLMo-1B',
        data_path: str = 'C:\\Uni works\\Research Implementations\\FedAgg-Qlora\\FedAgg\\data',
        output_dir: str = './qlora-FedAggreagtion',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.1,
        num_communication_rounds: int = 10,
        num_clients: int = 10,
        # Local training hyperparams
        local_batch_size: int = 2,
        local_micro_batch_size: int = 2,
        local_num_epochs: int = 10,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.01,
        # lora_target_modules: List[str] = ["q_proj"],
        lora_target_modules=["att_proj"],
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = True,  
        prompt_template_name: str = "olmo",
        lr_scheduler_type='constant',
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert global_model, "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(data_path), "Please generate the data files for each client"
    
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(global_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = Prompter(prompt_template_name).generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = Prompter(prompt_template_name).generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                                user_prompt_len:
                                ]

        return tokenized_full_prompt

    # Load LoRA configuration
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    print("The process of federated LoRA has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    for epoch in tqdm(range(num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(
            num_clients, client_selection_frac, client_selection_strategy,
            other_info=epoch
        )

        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.prepare_local_dataset(
                generate_and_tokenize_prompt, local_val_set_size
            )
            client.build_local_trainer(
                tokenizer,
                local_micro_batch_size = 8,
                gradient_accumulation_steps = 4,
                local_num_epochs = 10,
                local_learning_rate = 0.0003,
                group_by_length = True,  # Pass group_by_length here
                ddp = False,  # Assuming you don't want ddp
            )

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            print("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set
            )
            del client

        print("Collecting the weights of clients and performing aggregation")
        # Implement the logic for federated averaging here

        # Save individual client models instead of aggregating
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, "client_{}_model_epoch_{}.bin".format(client_id, epoch)),
        )

        # Save the model's configuration
        model.config.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(fl_finetune)