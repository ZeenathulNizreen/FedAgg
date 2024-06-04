
Run in the vscode
python main.py --global_model 'allenai/OLMo-1B' \
  --data_path '/root/FedAgg/data' \
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


run in the vram

python main.py --global_model 'allenai/OLMo-1B' \
  --data_path '/home/jovyan/work/FedAgg/data' \
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
  --group_by_length True 
  