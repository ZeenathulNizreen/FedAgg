#!/bin/bash

# Install the basic requirements
pip install -r requirements.txt

# Install TRL from a specific GitHub commit
pip install git+https://github.com/huggingface/trl@a3c5b7178ac4f65569975efadc97db2f3749c65e --upgrade


import torch; assert torch.cuda.get_device_capability()[0] >= 8, 'Hardware not supported for Flash Attention'
# install flash-attn
!pip install ninja packaging
!MAX_JOBS=2 pip install flash-attn --no-build-isolation