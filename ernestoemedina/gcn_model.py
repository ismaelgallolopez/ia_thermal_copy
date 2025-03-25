# gcn_model.py
# Este módulo define la arquitectura de la Graph Convolutional Network (GCN).
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ModuleList

class GCN(torch.nn.Module):  
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers

        # Definir las capas de convolución
        self.convs = ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))  # Primera capa

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))  # Capas intermedias

        self.convs.append(GCNConv(hidden_dim, output_dim))  # Última capa

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x
