from types import SimpleNamespace
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
import yaml

# Load merge configuration
with open("merge.yaml", "r") as f:
    merge_config = MergeConfiguration.model_validate(yaml.safe_load(f))

# FINAL options to fix all errors
options = SimpleNamespace(
    transformers_cache=None,
    lazy_unpickle=False,
    trust_remote_code=True,
    random_seed=None,
    allow_overwrite=True,
    lora_merge_cache=None,
    lora_merge_dtype="float16",
    quiet=False,
    read_to_gpu=False,
    out_shard_size=2**30,  # 1GB shard
    safe_serialization=True,
    clone_tensors=True,
    multi_gpu=False,
    cuda=True,
    low_cpu_memory=False,  
)



# Run the merge
run_merge(
    merge_config,
    "./merged_models",
    options
)
