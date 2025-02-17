import sys
import os
import torch  # Si usas PyTorch

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("sys.prefix:", sys.prefix)
print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV"))
print("CUDA available:", torch.cuda.is_available())
