import torch
from torch.utils.data import Dataset

class PCBDataset(Dataset):  # <-- Asegúrate que esta clase esté definida aquí
    def __init__(self, inputs_dataset, outputs_dataset, scalar_dataset):
        assert len(inputs_dataset) == len(outputs_dataset) == len(scalar_dataset), "All datasets must be of the same size"
        self.inputs_dataset = inputs_dataset
        self.outputs_dataset = outputs_dataset
        self.scalar_dataset = scalar_dataset

    def __len__(self):
        return len(self.inputs_dataset)

    def __getitem__(self, idx):
        input_data = self.inputs_dataset[idx]
        output_data = self.outputs_dataset[idx]
        scalar_data = self.scalar_dataset[idx]
        return input_data, output_data, scalar_data

def load_dataset(file_path):
    """Carga un dataset tipo PCB_dataset desde un archivo .pth"""
    dataset = torch.load(file_path)  # La función torch.load() debe encontrar PCBDataset aquí
    if not isinstance(dataset, PCBDataset):
        raise TypeError("El archivo cargado no es de tipo PCBDataset.")
    return dataset
