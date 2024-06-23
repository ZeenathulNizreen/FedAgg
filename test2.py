import os
from typing import List
from tqdm import tqdm
import fire
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, prepare_model_for_kbit_training
from fed_utils import FedAvg, merge_models_kit, client_selection, GeneralClient
from fed_utils.evaluation import evaluate_model  # Use absolute import
import datasets
from utils.prompter import Prompter
from trl import SFTTrainer
from mergekit import Linear, Ties, Passthrough, TaskArithmetic

datasets.utils.logging.set_verbosity_error()

def fl_finetune(
        global_model: str = 'allenai/OLMo-1B',
        data_path: str = '/home/jovyan/work/FedAgg/data/10/',
        output_dir: str = './qlora-FedAggregation',
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.05,
        num_communication_rounds: int = 20,
        num_clients: int = 10,
        local_batch_size: int = 32,
        local_micro_batch_size: int = 32,
        local_num_epochs: int = 10,
        local_learning_rate: float = 1.5e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.01,
        lora_target_modules=["att_proj"],
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
    assert global_model 

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

    print("The process of federated QLoRA has started..")
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
                local_micro_batch_size=local_micro_batch_size,
                gradient_accumulation_steps=local_batch_size // local_micro_batch_size,
                local_num_epochs=local_num_epochs,
                local_learning_rate=local_learning_rate,
                group_by_length=group_by_length,
                ddp=False,
            )

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            # Save the client model weights after training
            local_output_dir = os.path.join(output_dir, str(client_id), f"local_output_{epoch}")
            os.makedirs(local_output_dir, exist_ok=True)
            client_model_path = os.path.join(local_output_dir, "adapter_model.safetensors")
            torch.save(model.state_dict(), client_model_path)

            print("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set
            )
            del client

        # Perform aggregation using Linear merge method from mergekit
        print("Collecting the weights of clients and performing aggregation using Linear method")
        model_linear = merge_models_kit(model,
                                        selected_clients_set,
                                        output_dir,
                                        epoch,
                                        merge_type="linear")
        torch.save(model_linear.state_dict(), os.path.join(output_dir, str(epoch), "Linear_adapter_model.safetensors"))

        print("Collecting the weights of clients and performing aggregation using Ties method")
        model_ties = merge_models_kit(model,
                                      selected_clients_set,
                                      output_dir,
                                      epoch,
                                      merge_type="ties")
        torch.save(model_ties.state_dict(), os.path.join(output_dir, str(epoch), "Ties_adapter_model.safetensors"))

        print("Collecting the weights of clients and performing aggregation using Passthrough method")
        model_passthrough = merge_models_kit(model,
                                             selected_clients_set,
                                             output_dir,
                                             epoch,
                                             merge_type="passthrough")
        torch.save(model_passthrough.state_dict(), os.path.join(output_dir, str(epoch), "Passthrough_adapter_model.safetensors"))

        print("Collecting the weights of clients and performing aggregation using Task Arithmetic method")
        model_task_arithmetic = merge_models_kit(model,
                                                 selected_clients_set,
                                                 output_dir,
                                                 epoch,
                                                 merge_type="task_arithmetic")
        torch.save(model_task_arithmetic.state_dict(), os.path.join(output_dir, str(epoch), "TaskArithmetic_adapter_model.safetensors"))

        # Evaluate FedAvg model
        print("Evaluating FedAvg model...")
        fedavg_loss = evaluate_model(model, tokenizer, os.path.join(data_path, 'global_test.json'))
        print(f"FedAvg model loss: {fedavg_loss}")

    # Save the final aggregated models
    final_linear_model_path = os.path.join(output_dir, "final_linear_model.safetensors")
    torch.save(model_linear.state_dict(), final_linear_model_path)
    print(f"Final Linear model saved to {final_linear_model_path}")

    final_ties_model_path = os.path.join(output_dir, "final_ties_model.safetensors")
    torch.save(model_ties.state_dict(), final_ties_model_path)
    print(f"Final Ties model saved to {final_ties_model_path}")

    final_passthrough_model_path = os.path.join(output_dir, "final_passthrough_model.safetensors")
    torch.save(model_passthrough.state_dict(), final_passthrough_model_path)
    print(f"Final Passthrough model saved to {final_passthrough_model_path}")

    final_task_arithmetic_model_path = os.path.join(output_dir, "final_task_arithmetic_model.safetensors")
    torch.save(model_task_arithmetic.state_dict(), final_task_arithmetic_model_path)
    print(f"Final Task Arithmetic model saved to {final_task_arithmetic_model_path}")

    # Perform global evaluation on the final models
    print("Performing final evaluation on Linear model...")
    final_linear_loss = evaluate_model(model_linear, tokenizer, os.path.join(data_path, 'global_test.json'))
    print(f"Final Linear model loss: {final_linear_loss}")

    print("Performing final evaluation on Ties model...")
    final_ties_loss = evaluate_model(model_ties, tokenizer, os.path.join(data_path, 'global_test.json'))
    print(f"Final Ties model loss: {final_ties_loss}")

    print("Performing final evaluation on Passthrough model...")
    final_passthrough_loss = evaluate_model(model_passthrough, tokenizer, os.path.join(data_path, 'global_test.json'))
    print(f"Final Passthrough model loss: {final_passthrough_loss}")

    print("Performing final evaluation on Task Arithmetic model...")
    final_task_arithmetic_loss = evaluate_model(model_task_arithmetic, tokenizer, os.path.join(data_path, 'global_test.json'))
    print(f"Final Task Arithmetic model loss: {final_task_arithmetic_loss}")

if __name__ == "__main__":
    fire.Fire(fl_finetune)

