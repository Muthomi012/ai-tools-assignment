# pytorch_demo.py
# This script verifies PyTorch installation and checks for GPU availability.

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available (GPU):", torch.cuda.is_available())
