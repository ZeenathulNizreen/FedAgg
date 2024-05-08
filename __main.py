bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model_id = "allenai/OLMo-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

tokenizer.all_special_tokens

print(model)

from peft import prepare_model_for_kbit_training
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["att_proj"],
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

from transformers import TrainingArguments

args = TrainingArguments(
  output_dir = "OLMo_instruct_generation",
  #num_train_epochs=5,
  max_steps = 500, # comment out this line if you want to train in epochs
  per_device_train_batch_size = 4,
  warmup_steps = 0.03,
  logging_steps=10,
  save_strategy="epoch",
  #evaluation_strategy="epoch",
  evaluation_strategy="steps",
  eval_steps=20, # comment out this line if you want to evaluate at the end of each epoch
  learning_rate=2e-4,
  bf16=True,
  lr_scheduler_type='constant',
)
from trl import SFTTrainer

max_seq_length = 2048

trainer = SFTTrainer(
  model=model,
  peft_config=peft_config,
  max_seq_length=max_seq_length,
  tokenizer=tokenizer,
  packing=True,
  formatting_func=create_prompt,
  args=args,
  train_dataset=dataset["train"],
  eval_dataset=dataset["test"]
)



python main.py --global_model 'allenai/OLMo-1B' \
  --data_path '/root/FedAgg/data/10' \
  --output_dir './qlora-FedAggregation' \
  --num_communication_rounds 10 \
  --num_clients 10 \
  --client_selection_frac 0.1 \
  --local_num_epochs 10 \
  --local_batch_size 8 \
  --local_micro_batch_size 4 \
  --local_learning_rate 0.0003 \
  --lora_r 64 \
  --lora_target_modules '["att_proj"]' \
  --train_on_inputs True \
  --group_by_length True \
  --trust_remote_code True 





python main.py `
  --global_model 'allenai/OLMo-1B' `
  --data_path 'C:\\Uni works\\Research Implementations\\FedAgg-Qlora\\FedAgg\\data' `
  --output_dir './qlora-FedAggregation' `
  --num_communication_rounds 10 `
  --num_clients 10 `
  --client_selection_frac 0.1 `
  --local_num_epochs 10 `
  --local_batch_size 8 `
  --local_micro_batch_size 4 `
  --local_learning_rate 0.0003 `
  --lora_r 64 `
  --lora_target_modules '["att_proj"]' `
  --train_on_inputs True `
  --group_by_length True `
  --trust_remote_code True