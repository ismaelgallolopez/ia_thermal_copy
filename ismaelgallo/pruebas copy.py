import torch


print("Versión de PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Versión de CUDA:", torch.version.cuda)
    print("Nombre de la GPU:", torch.cuda.get_device_name(0))
