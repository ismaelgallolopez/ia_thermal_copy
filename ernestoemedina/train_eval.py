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
            accuracy_within_threshold = torch.sum(within_threshold).item() / len(true_batch) * 100
            
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
        plot_maps(all_true_vals[idx], all_pred_vals[idx])
        
    # Calcular métricas promedio para todo el DataLoader
    mean_mse = np.mean(all_mse)
    mean_mae = np.mean(all_mae)
    mean_r2 = np.mean(all_r2)
    mean_accuracy = np.mean(all_accuracy)

    return mean_mse, mean_mae, mean_r2, mean_accuracy


def plot_maps(true_vals, pred_vals):
    total_nodos = true_vals.shape[0]

    # Calcular el tamaño de la malla para el target
    nodos_lado_target = int(np.sqrt(total_nodos))  # Asumiendo que el target es siempre un cuadrado perfecto
    
    if nodos_lado_target ** 2 != total_nodos:
        raise ValueError(f"El número de nodos del target ({total_nodos}) no forma un cuadrado perfecto.")
    
    # Recortar las predicciones si tienen más nodos que el target
    if pred_vals.shape[0] > total_nodos:
        pred_vals = pred_vals[:total_nodos]
    
    # Transformar a matrices 2D con el tamaño detectado
    true_vals = true_vals.numpy().reshape(nodos_lado_target, nodos_lado_target)
    pred_vals = pred_vals.numpy().reshape(nodos_lado_target, nodos_lado_target)

    # Calcular el error relativo (en %)
    error_relative = np.abs((pred_vals - true_vals) / true_vals) * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Mapa del Target
    im1 = axes[0].imshow(true_vals, cmap='jet')
    axes[0].set_title("Mapa del Target (Temperaturas Reales)")
    fig.colorbar(im1, ax=axes[0])

    # Mapa Predicho
    im2 = axes[1].imshow(pred_vals, cmap='jet')
    axes[1].set_title("Mapa Predicho por la GCN (Temperaturas Predichas)")
    fig.colorbar(im2, ax=axes[1])

    # Mapa de Error Relativo
    im3 = axes[2].imshow(error_relative, cmap='jet')
    axes[2].set_title("Mapa del Error Relativo (%)")
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
