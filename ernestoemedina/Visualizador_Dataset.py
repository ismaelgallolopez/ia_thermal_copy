import torch
data = torch.load(r"C:\Users\ramse\Escritorio\Aero Ordenador\4to Aero\2do Cuatri\Repo_TFG\ia_thermal\PCB_dataset.pth", map_location=torch.device('cpu'))

print(type(data))  # Ver el tipo de datos almacenados

