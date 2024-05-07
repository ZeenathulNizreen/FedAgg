
import os
import torch
import transformers
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from collections import OrderedDict
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import copy

class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, f"local_training_{self.client_id}.json")
        try:
            self.local_data = load_dataset("json", data_files=self.local_data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No data file found at {self.local_data_path}")
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(output_dir, "trainer_saved", f"local_output_{self.client_id}")

    def prepare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        try:
            if local_val_set_size > 0:
                local_train_val = self.local_data["train"].train_test_split(test_size=local_val_set_size, seed=42)
                self.local_train_dataset = local_train_val["train"].map(generate_and_tokenize_prompt)
                self.local_eval_dataset = local_train_val["test"].map(generate_and_tokenize_prompt)
            else:
                self.local_train_dataset = self.local_data["train"].map(generate_and_tokenize_prompt)
                self.local_eval_dataset = None
        except Exception as e:
            raise Exception(f"Error preparing datasets: {str(e)}")

    def build_local_trainer(self, tokenizer, local_micro_batch_size, local_num_epochs, local_learning_rate, group_by_length, ddp=False):
        self.train_args = TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_dir='./logs',
            logging_steps=10,
            weight_decay=0.01,
            evaluation_strategy="steps" if self.local_eval_dataset else "no",
            save_strategy="steps",
            eval_steps=50 if self.local_eval_dataset else None,
            save_steps=50,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_eval_dataset else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        self.local_trainer = Trainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset if self.local_eval_dataset else None,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=self.model)
        )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(OrderedDict((name, param.detach()) for name, param in self.model.named_parameters()))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters())
        self.model.state_dict = lambda instance, *_, **__: get_peft_model_state_dict(instance, self.params_dict_new)

    def train(self):
        self.local_trainer.train()

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), f"local_output_{self.client_id}")
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, os.path.join(single_output_dir, "pytorch_model.bin"))

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old)
        set_peft_model_state_dict(self.model, older_adapter_weight)
        previously_selected_clients_set.add(self.client_id)

        return self.model, local_dataset_len_dict, previously_selected_clients_set

