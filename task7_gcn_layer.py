import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj):
        # Add self-loops to adjacency matrix
        adj_cap = adj + torch.eye(adj.size(0))
        # Compute degree matrix
        D = adj_cap.sum(dim=1)
        # Compute D^(-1)
        D_inv = torch.diag(1.0 / D)
        # Normalize adjacency matrix
        norm = D_inv @ adj_cap
        # Aggregate features
        out = norm @ x
        # Linear transformation
        out = self.linear(out)
        # applying activation function(ReLU)
        return torch.relu(out)
    
class GCN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        return x
        
# Test code
adj = torch.tensor([[0, 1, 0, 1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 0, 1, 0]], dtype=torch.float32)

x = torch.rand(4, 3)  # 4 nodes, 3 features each

model = GCN(in_features=3, hidden_features=4, out_features=2)

print("Input shape:", x.shape)
output = model(x, adj)
print("Output shape:", output.shape)