import torch

x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

W1 = torch.randn(2, 4, requires_grad=True)
b1 = torch.randn(4, requires_grad=True)
W2 = torch.randn(4, 1, requires_grad=True)
b2 = torch.randn(1, requires_grad=True)

for iteration in range(1, 5001):
    hidden = torch.sigmoid(x @ W1 + b1)
    output = torch.sigmoid(hidden @ W2 + b2)

    loss = ((output - y) ** 2).mean()

    loss.backward()

    lr = 1.0 #learning rate

    W1.data -= W1.grad * lr
    b1.data  -= b1.grad * lr
    W2.data  -= W2.grad * lr
    b2.data  -= b2.grad * lr

    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()

    if(iteration % 1000 == 0):
        print(f"Iteration {iteration}, Loss: {loss}")

# After training loop
print("\n--- Final Results ---")
with torch.no_grad():
    hidden = torch.sigmoid(x @ W1 + b1)
    predictions = torch.sigmoid(hidden @ W2 + b2)
    
print("Predictions:", predictions.T)
print("Targets:    ", y.T)
