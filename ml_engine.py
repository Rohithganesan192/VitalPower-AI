import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNFeatureLayer(nn.Module):
    def __init__(self, num_features):
        super(GNNFeatureLayer, self).__init__()
        self.adj_matrix = nn.Parameter(torch.eye(num_features) + torch.randn(num_features, num_features) * 0.1)
        self.feature_weights = nn.Linear(num_features, num_features)
        
    def forward(self, x):
        adj_norm = F.softmax(self.adj_matrix, dim=-1)
        graph_out = torch.matmul(x, adj_norm.t())
        return F.relu(self.feature_weights(graph_out))

class HybridTransformerGNN(nn.Module):
    def __init__(self, num_features, seq_len, nhead=2, num_layers=2, hidden_dim=64): # nhead changed to 2
        super(HybridTransformerGNN, self).__init__()
        self.gnn = GNNFeatureLayer(num_features)
        self.input_projection = nn.Linear(num_features, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 2, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, num_features)
        
    def forward(self, x):
        x_graph = self.gnn(x)
        x_proj = self.input_projection(x_graph)
        x_trans = self.transformer(x_proj)
        return self.output_projection(x_trans)

    def evaluate_anomaly(self, sequence, threshold=1.0):
        self.eval()
        with torch.no_grad():
            if sequence.dim() == 2:
                sequence = sequence.unsqueeze(0)
            
            reconstructed = self.forward(sequence)
            
            # Overall MSE for anomaly score
            mse_loss = F.mse_loss(reconstructed, sequence).item()
            
            # Feature-level MSE to identify the culprit node
            mse_per_feature = F.mse_loss(reconstructed, sequence, reduction='none').mean(dim=(0, 1)).cpu().numpy().tolist()
            
            # Non-linear scaling for UI presentation (0 to 1)
            anomaly_score = 1.0 - math.exp(-mse_loss / (threshold * 0.5))
            anomaly_score = max(0.0, min(1.0, anomaly_score))
            
            is_anomaly = anomaly_score > 0.75
            return anomaly_score, is_anomaly, mse_per_feature