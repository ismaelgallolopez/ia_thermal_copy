# train_eval.py
# Este módulo se encarga de entrenar, evaluar y predecir con la GCN.

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    true_vals, pred_vals = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            true_vals.append(data.y.cpu())
            pred_vals.append(out.squeeze().cpu())

    true_vals = torch.cat(true_vals, dim=0)
    pred_vals = torch.cat(pred_vals, dim=0)

    mse = mean_squared_error(true_vals, pred_vals)
    mae = mean_absolute_error(true_vals, pred_vals)
    r2 = r2_score(true_vals, pred_vals)

    return mse, mae, r2


def predict(model, loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            predictions.append(out.cpu())

    return torch.cat(predictions, dim=0)
