import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

edge_index = torch.tensor([
    [0, 1, 0, 3, 1, 2, 1, 4, 2, 3, 3, 4, 4, 5, 4, 7, 5, 6, 6, 7],
    [1, 0, 3, 0, 2, 1, 4, 1, 3, 2, 4, 3, 5, 4, 7, 4, 6, 5, 7, 6]
], dtype=torch.long)

x = torch.rand((8, 4), dtype=torch.float)
labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=labels)

class GCN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        return torch.log_softmax(x, dim=1)
    
model = GCN(in_features=4, hidden_features=8, out_features=2)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(201):
    model.train()

    pred = model(data)

    loss = criterion(pred, data.y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

with torch.no_grad():
    model.eval()
    pred = model(data)
    predicted_classes = pred.argmax(dim=1)
    accuracy = (predicted_classes == data.y).sum().item() / data.y.size(0)
    print(f'Accuracy: {accuracy:.4f}')