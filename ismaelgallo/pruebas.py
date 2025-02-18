import sys
import os
import torch  # Si usas PyTorch

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("sys.prefix:", sys.prefix)
print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV"))
print("CUDA available:", torch.cuda.is_available())

print("Versión de PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Versión de CUDA:", torch.version.cuda)
    print("Nombre de la GPU:", torch.cuda.get_device_name(0))