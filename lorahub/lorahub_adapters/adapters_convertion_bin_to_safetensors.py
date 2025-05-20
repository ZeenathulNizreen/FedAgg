import torch
from safetensors.torch import save_file
import os

# === CONFIG ===
input_paths = [
    "../../qlora-FedAggregation/10/client_2_model_epoch_0.bin",
    "../../qlora-FedAggregation/10/client_2_model_epoch_1.bin",
    "../../qlora-FedAggregation/10/client_3_model_epoch_4.bin",
    "../../qlora-FedAggregation/10/client_4_model_epoch_2.bin",
    "../../qlora-FedAggregation/10/client_5_model_epoch_3.bin",
    "../../qlora-FedAggregation/10/client_8_model_epoch_6.bin",
    "../../qlora-FedAggregation/10/client_8_model_epoch_7.bin",
    "../../qlora-FedAggregation/10/client_8_model_epoch_8.bin",
    "../../qlora-FedAggregation/10/client_8_model_epoch_9.bin",
    "../../qlora-FedAggregation/10/client_9_model_epoch_5.bin"
]
output_dir = "lorahub/lorahub_adapters"

os.makedirs(output_dir, exist_ok=True)

for path in input_paths:
    state_dict = torch.load(path, map_location="cpu")
    new_name = os.path.splitext(os.path.basename(path))[0] + ".safetensors"
    save_path = os.path.join(output_dir, new_name)
    save_file(state_dict, save_path)
    print(f" Converted: {path} -> {save_path}")
