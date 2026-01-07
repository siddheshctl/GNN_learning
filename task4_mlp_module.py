import torch
import torch.nn as nn

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2,4)
        self.output = nn.Linear(4,1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))

        return x
    
model = XORNet()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1.0)

# Data
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

for epoch in range(5001):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(epoch % 1000 == 0):
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("final result")
with torch.no_grad():
    predictions = model(X)
    print("Predictions:", predictions.T)
    print("Targets:    ", y.T)