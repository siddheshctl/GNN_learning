import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

dataset = Planetoid(root = 'data/Cora', name = 'Cora')
data = dataset[0]

print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Features: {data.num_node_features}")
print(f"Classes: {dataset.num_classes}")

print(f"Training nodes: {data.train_mask.sum()}")
print(f"Validation nodes: {data.val_mask.sum()}")
print(f"Test nodes: {data.test_mask.sum()}")

class GCN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.log_softmax(x, dim=1)
    
model = GCN(in_features=dataset.num_node_features, hidden_features=64, out_features=dataset.num_classes)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(201):
    model.train()

    pred = model(data)

    loss = criterion(pred[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

with torch.no_grad():
    model.eval()
    pred = model(data)
    predicted_classes = pred.argmax(dim=1)
    correct = (predicted_classes[data.test_mask] == data.y[data.test_mask]).sum().item()
    accuracy = correct / data.test_mask.sum().item()
    print(f'Accuracy: {accuracy}')