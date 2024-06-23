import os
import torch
from mergekit import Linear

def merge_models_kit(model, selected_clients_set, output_dir, epoch, merge_type="linear"):
    model_weights_list = []
    for client_id in selected_clients_set:
        single_output_dir = os.path.join(output_dir, str(epoch), f"local_output_{client_id}", "adapter_model.safetensors")
        single_weights = torch.load(single_output_dir)
        model_weights_list.append(single_weights)

    # Use Linear method for merging
    if merge_type == "linear":
        # Define weights for each model
        weights = [1.0] * len(model_weights_list)  # Adjust this if different weights are needed

        # Perform linear merge
        merged_weights = Linear.merge(model_weights_list, weights=weights)

    # Set the merged weights to the model
    model.load_state_dict(merged_weights)
    return model
