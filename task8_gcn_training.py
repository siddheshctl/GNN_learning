import torch
import torch.nn as nn



x = torch.rand(8, 4)
adj_matrix = torch.tensor([[0, 1, 0, 1, 0, 0, 0, 0],
                           [1, 0, 1, 0, 1, 0, 0, 0],
                           [0, 1, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 1],
                           [0, 0, 0, 0, 1, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1, 0, 1],
                           [0, 0, 0, 0, 1, 0, 1, 0]], dtype=torch.float32)

labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj):
        adj_cap = adj + torch.tensor(torch.eye(adj.size(0)))

        D = adj_cap.sum(dim=1)

        D_inv = torch.diag(1.0 / D)

        norm = D_inv @ adj_cap

        out = norm @ x

        out = self.linear(out)

        return torch.log_softmax(out, dim = 1)
    

class GCN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        
        return x
    
model = GCN(in_features=4, hidden_features=8, out_features=2)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(201):
    model.train()

    pred = model(x, adj_matrix)

    loss = criterion(pred, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch % 50 == 0):
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Final results")
with torch.no_grad():
    model.eval()
    predictions = model(x, adj_matrix)
    predicted_classes = torch.argmax(predictions, dim=1)
    print("Predicted:", predicted_classes)
    print("Labels:      ", labels)
    print("Accuracy:         ", (predicted_classes == labels).sum().item() / len(labels))