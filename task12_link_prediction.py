import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score

dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]


transform = RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True
)

train_data, val_data, test_data = transform(data)

class LinkPredictor(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        src, dst = edge_index
        prob = (z[src] * z[dst]).sum(dim=1)
        return torch.sigmoid(prob)
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
    
model = LinkPredictor(in_features=dataset.num_node_features, hidden_features=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(201):
    model.train()

    optimizer.zero_grad()

    out = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
    loss = criterion(out, train_data.edge_label.float())

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    out = model(test_data.x, test_data.edge_index, test_data.edge_label_index)
    pred = torch.sigmoid(out)
    labels = test_data.edge_label.numpy()

    auc = roc_auc_score(labels, pred.numpy())
    print(f'Test AUC: {auc:.4f}')
