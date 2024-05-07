
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_double_quant=True,
    bnb_4bit_compute_dtype=torch.floa

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



python main.py --global_model 'allenai/OLMo-1B' \
  --data_path '/root/FedAgg/data/10' \
  --output_dir './qlora-FedAggregation' \
  --num_communication_rounds 10 \
  --num_clients 10 \
  --client_selection_frac 0.1 \
  --local_num_epochs 10 \
  --local_batch_size 8 \
  --local_micro_batch_size 8 \
  --local_learning_rate 0.0003 \
  --lora_r 64 \
  --lora_target_modules '["att_proj"]' \
  --train_on_inputs True \
  --group_by_length True \
  --trust_remote_code True

