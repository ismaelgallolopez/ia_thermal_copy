# train_eval.py
# Este módulo se encarga de entrenar, evaluar y predecir con la GCN.

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        out = out.view(-1)  # Aplana el tensor de (5408, 1) a (5408)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, error_threshold=5.0, plot_results=True):
    model.eval()
    all_mse, all_mae, all_r2, all_accuracy = [], [], [], []
    all_true_vals, all_pred_vals = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            out = out.view(-1)

            true_batch = data.y.cpu()
            pred_batch = out.cpu()

            # Calcular métricas para este batch
            mse = mean_squared_error(true_batch, pred_batch)
            mae = mean_absolute_error(true_batch, pred_batch)
            r2 = r2_score(true_batch, pred_batch)
            within_threshold = torch.abs(true_batch - pred_batch) <= error_threshold
            accuracy_within_threshold = torch.sum(within_threshold.float()).item() / len(true_batch) * 100
            
            all_mse.append(mse)
            all_mae.append(mae)
            all_r2.append(r2)
            all_accuracy.append(accuracy_within_threshold)
            
            # Guardar los gráficos para luego seleccionar uno
            all_true_vals.append(true_batch)
            all_pred_vals.append(pred_batch)
    
    # Seleccionar un gráfico al azar para graficar
    if plot_results and len(all_true_vals) > 0:
        idx = random.randint(0, len(all_true_vals) - 1)
        plot_temperature_maps(all_true_vals[idx], all_pred_vals[idx])
        
    # Calcular métricas promedio para todo el DataLoader
    mean_mse = np.mean(all_mse)
    mean_mae = np.mean(all_mae)
    mean_r2 = np.mean(all_r2)
    mean_accuracy = np.mean(all_accuracy)

    return mean_mse, mean_mae, mean_r2, mean_accuracy


def plot_temperature_maps(true_vals, pred_vals):
    """
    Muestra gráficas de Temperaturas Reales, Temperaturas Predichas y Error Absoluto (en Kelvin) para un gráfico dado.
    
    Parámetros:
    - true_vals: Tensor o array con las temperaturas reales (dimensión: num_nodos,).
    - pred_vals: Tensor o array con las temperaturas predichas (dimensión: num_nodos,).
    """
    # Convertir a numpy si es un tensor de PyTorch
    if hasattr(true_vals, 'numpy'):
        true_vals = true_vals.numpy()
    if hasattr(pred_vals, 'numpy'):
        pred_vals = pred_vals.numpy()
    
    # Verificar que tengan el mismo tamaño
    if true_vals.shape[0] != pred_vals.shape[0]:
        raise ValueError("El número de nodos en true_vals y pred_vals debe ser igual.")
    
    total_nodos = true_vals.shape[0]
    nodos_lado = int(np.sqrt(total_nodos))  # Calcular el tamaño del lado del gráfico

    if nodos_lado ** 2 != total_nodos:
        raise ValueError(f"El número de nodos ({total_nodos}) no forma un cuadrado perfecto.")
    
    # Convertir a matrices cuadradas
    true_vals = true_vals.reshape(nodos_lado, nodos_lado)
    pred_vals = pred_vals.reshape(nodos_lado, nodos_lado)
    
    # Calcular el error absoluto en Kelvin
    error_absolute = np.abs(true_vals - pred_vals)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Mapa del Target (Temperaturas Reales)
    im1 = axes[0].imshow(true_vals, cmap='jet')
    axes[0].set_title("Mapa del Target (Temperaturas Reales)")
    fig.colorbar(im1, ax=axes[0])

    # Mapa de la Predicción (Temperaturas Predichas)
    im2 = axes[1].imshow(pred_vals, cmap='jet')
    axes[1].set_title("Mapa de la Predicción (Temperaturas Predichas)")
    fig.colorbar(im2, ax=axes[1])

    # Mapa del Error Absoluto en Kelvin
    im3 = axes[2].imshow(error_absolute, cmap='jet')
    axes[2].set_title("Mapa del Error Absoluto (K)")
    fig.colorbar(im3, ax=axes[2])

    plt.show()



def predict(model, loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            predictions.append(out.cpu())

    return torch.cat(predictions, dim=0)
