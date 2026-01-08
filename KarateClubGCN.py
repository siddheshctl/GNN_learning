import torch
import torch.nn as nn
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv

dataset = KarateClub()
data = dataset[0]

num_nodes = data.num_nodes
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[:num_nodes // 2] = True
test_mask[num_nodes // 2:] = True

data.train_mask = train_mask
data.test_mask = test_mask

print("Nodes: ", data.num_nodes)
print("Edges: ", data.num_edges)
print("Features: ", data.num_node_features)
print("Classes: ", dataset.num_classes)

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        return torch.log_softmax(x, dim=1)
    
model = GCN(in_features=dataset.num_node_features, hidden_features=4, out_features=dataset.num_classes)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)

for epoch in range(201):
    model.train()

    pred = model(data)
    
    loss = criterion(pred[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch} \t Loss: {loss}")

with torch.no_grad():
    model.eval()
    pred = model(data)
    predicted_classes = pred.argmax(dim=1)
    correct = (predicted_classes[data.test_mask] == data.y[data.test_mask]).sum().item()
    accuracy = correct / data.test_mask.sum().item()
    print(f'Accuracy: {accuracy}')
