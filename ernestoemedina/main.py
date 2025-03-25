# main.py
# Este archivo principal une todos los módulos y ejecuta el entrenamiento, validación y evaluación del modelo GCN.

import torch
from torch_geometric.loader import DataLoader
from Dataset_loader import PCBDataset  # Asegúrate que Dataset_loader se encuentra en el mismo directorio
from gcn_model import GCN
from data_loader import load_pcb_dataset
from train_eval import train, evaluate
import os


def main():
    # Configuración de parámetros
    file_path = 'PCB_dataset.pth'  # Ruta al archivo de dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    
    # Mensaje de inicio
    print("Iniciando el script...")  
    print(f"Usando dispositivo: {device}")
    print("Cargando dataset...")

    # Hiperparámetros
    input_dim = 2  # Potencia e interfaz como características
    hidden_dim = 64
    output_dim = 1  # Temperatura predicha por cada nodo
    num_layers = 3
    learning_rate = 0.001
    batch_size = 32
    epochs = 10

    # Cargar el dataset
    print("Cargando dataset...")
    graphs = load_pcb_dataset(file_path)

    # Dividir en entrenamiento, validación y prueba
    train_size = int(0.8 * len(graphs))
    val_size = int(0.1 * len(graphs))
    test_size = len(graphs) - train_size - val_size

    train_dataset = graphs[:train_size]
    val_dataset = graphs[train_size:train_size + val_size]
    test_dataset = graphs[train_size + val_size:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Crear el modelo
    model = GCN(input_dim, hidden_dim, output_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Entrenamiento del modelo
    print("Iniciando entrenamiento...")
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_mse, val_mae, val_r2 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val MSE: {val_mse:.4f} - Val MAE: {val_mae:.4f} - Val R2: {val_r2:.4f}")

    # Evaluar en el conjunto de prueba
    print("Evaluando en el conjunto de prueba...")
    test_mse, test_mae, test_r2 = evaluate(model, test_loader, device)
    print(f"Test MSE: {test_mse:.4f} - Test MAE: {test_mae:.4f} - Test R2: {test_r2:.4f}")


if __name__ == "__main__":
    main()
