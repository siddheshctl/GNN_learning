import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')

print("Number of Graphs:", len(dataset))
print("Number of classes: ", dataset.num_classes)
print("NUmber of node features: ", dataset.num_node_features)

train_dataset = dataset[:150]
test_dataset = dataset[150:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class GraphClassifier(nn.Module):
    def __init__(self, in_feature, hidden_features, out_features):
        super().__init__()
        self.conv1 = GCNConv(in_feature, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.classifier = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)

        x = self.classifier(x)

        return torch.log_softmax(x, dim=1)
    
model = GraphClassifier(in_feature=dataset.num_node_features,
                        hidden_features=64,
                        out_features=dataset.num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr =0.05)
criterion = torch.nn.CrossEntropyLoss()
    
for epoch in range(201):
    model.train()

    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")


model.eval()
correct = 0
total = 0

for batch in test_loader:
    out = model(batch.x, batch.edge_index, batch.batch)
    pred = out.argmax(dim = 1)
    correct += (pred == batch.y).sum()
    total += batch.y.size(0)


print(f"Test Accuracy: {correct/total:.4f}")