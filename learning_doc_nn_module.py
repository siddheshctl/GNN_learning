"""
================================================================================
PYTORCH nn.Module - LEARNING DOCUMENT
================================================================================
Purpose: Learn PyTorch's neural network abstractions
Prerequisite: You completed task3_mlp_scratch.py (manual neural network)

This document teaches you how PyTorch professionals build neural networks.
================================================================================
"""

import torch
import torch.nn as nn  # Neural network module
import torch.optim as optim  # Optimizers

# ============================================================================
# SECTION 1: WHY nn.Module?
# ============================================================================
"""
In task3, you manually:
    - Created weight tensors with requires_grad=True
    - Updated weights with: W.data -= lr * W.grad
    - Zeroed gradients manually

This works, but becomes painful with larger networks (imagine 100 layers!).

nn.Module provides:
    1. Automatic parameter management (no manual requires_grad)
    2. Easy layer definitions (nn.Linear does W @ x + b for you)
    3. Works with optimizers (no manual weight updates)
    4. Clean, reusable architecture
"""

# ============================================================================
# SECTION 2: ANATOMY OF nn.Module
# ============================================================================

print("=" * 60)
print("SECTION 2: ANATOMY OF nn.Module")
print("=" * 60)

# Here's the simplest possible neural network:

class SimpleNetwork(nn.Module):
    """
    Every neural network in PyTorch inherits from nn.Module.
    You must define two things:
        1. __init__: Define your layers
        2. forward: Define how data flows through layers
    """

    def __init__(self):
        # ALWAYS call parent's __init__ first!
        super().__init__()

        # Define layers as class attributes
        # nn.Linear(in_features, out_features) = a fully connected layer
        # It automatically creates weights W and bias b
        self.layer1 = nn.Linear(2, 4)   # 2 inputs -> 4 outputs
        self.layer2 = nn.Linear(4, 1)   # 4 inputs -> 1 output

    def forward(self, x):
        """
        Define the forward pass.
        This is called when you do: output = model(input)
        """
        # x goes through layer1, then sigmoid activation
        x = self.layer1(x)      # Linear transformation
        x = torch.sigmoid(x)    # Activation function

        # Then through layer2 and sigmoid
        x = self.layer2(x)
        x = torch.sigmoid(x)

        return x

# Create an instance of the network
model = SimpleNetwork()
print("\nOur model:")
print(model)

# ============================================================================
# SECTION 3: WHAT'S INSIDE nn.Linear?
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 3: WHAT'S INSIDE nn.Linear?")
print("=" * 60)

"""
nn.Linear(in_features, out_features) is equivalent to:
    output = input @ W.T + b

Where:
    W has shape (out_features, in_features)
    b has shape (out_features,)

It's exactly what you did manually in task3!
"""

# Let's inspect the layers
print("\nlayer1 weights shape:", model.layer1.weight.shape)  # (4, 2)
print("layer1 bias shape:", model.layer1.bias.shape)        # (4,)

print("\nlayer2 weights shape:", model.layer2.weight.shape)  # (1, 4)
print("layer2 bias shape:", model.layer2.bias.shape)        # (1,)

# These are automatically registered as parameters with requires_grad=True!
print("\nlayer1 weight requires_grad:", model.layer1.weight.requires_grad)

# ============================================================================
# SECTION 4: model.parameters() - ALL WEIGHTS AT ONCE
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 4: model.parameters()")
print("=" * 60)

"""
model.parameters() returns an iterator over ALL learnable parameters.
This is what optimizers use to know which tensors to update.
"""

print("\nAll parameters in our model:")
for name, param in model.named_parameters():
    print(f"  {name}: shape {param.shape}")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal trainable parameters: {total_params}")
# layer1: 2*4 + 4 = 12, layer2: 4*1 + 1 = 5, total = 17

# ============================================================================
# SECTION 5: FORWARD PASS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 5: FORWARD PASS")
print("=" * 60)

"""
To run data through the network, just call the model like a function:
    output = model(input)

This internally calls model.forward(input).
NEVER call model.forward() directly - always use model(input).
"""

# Example input: batch of 3 samples, each with 2 features
sample_input = torch.tensor([[0.0, 0.0],
                              [1.0, 0.0],
                              [0.5, 0.5]])

print("\nInput shape:", sample_input.shape)  # (3, 2)

# Forward pass
output = model(sample_input)  # Not model.forward(sample_input)!
print("Output shape:", output.shape)  # (3, 1)
print("Output values:\n", output)

# ============================================================================
# SECTION 6: LOSS FUNCTIONS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 6: LOSS FUNCTIONS")
print("=" * 60)

"""
PyTorch provides common loss functions in nn module:
    - nn.MSELoss()      : Mean Squared Error (regression)
    - nn.CrossEntropyLoss() : Classification
    - nn.BCELoss()      : Binary Cross Entropy
    - nn.L1Loss()       : Mean Absolute Error
"""

# Create a loss function (it's like creating an object)
criterion = nn.MSELoss()

# Example usage:
predictions = torch.tensor([[0.8], [0.2], [0.5]])
targets = torch.tensor([[1.0], [0.0], [0.5]])

loss = criterion(predictions, targets)
print("\nPredictions:", predictions.T)
print("Targets:", targets.T)
print("MSE Loss:", loss.item())

# This is equivalent to what you did manually:
manual_loss = ((predictions - targets) ** 2).mean()
print("Manual MSE:", manual_loss.item())

# ============================================================================
# SECTION 7: OPTIMIZERS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 7: OPTIMIZERS")
print("=" * 60)

"""
Optimizers update weights based on gradients.
They replace your manual: W.data -= lr * W.grad

Common optimizers:
    - optim.SGD       : Stochastic Gradient Descent (what you did manually)
    - optim.Adam      : Adaptive learning rate (very popular)
    - optim.RMSprop   : Another adaptive method
"""

# Create an optimizer
# It needs to know: which parameters to update, and learning rate
optimizer = optim.SGD(model.parameters(), lr=1.0)

print("\nOptimizer:", optimizer)

"""
The optimizer has two key methods:
    optimizer.zero_grad()  : Zeros all gradients (like W.grad.zero_())
    optimizer.step()       : Updates all weights (like W.data -= lr * W.grad)
"""

# ============================================================================
# SECTION 8: THE TRAINING LOOP - PUTTING IT ALL TOGETHER
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 8: THE TRAINING LOOP")
print("=" * 60)

"""
Here's the standard PyTorch training pattern:

for epoch in range(num_epochs):
    # 1. Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 2. Backward pass
    optimizer.zero_grad()   # Clear old gradients
    loss.backward()         # Compute new gradients
    optimizer.step()        # Update weights
"""

# Let's train our simple network on XOR!
print("\n--- Training on XOR ---")

# Fresh model
model = SimpleNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1.0)

# XOR data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Training loop
for epoch in range(5001):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass
    optimizer.zero_grad()  # MUST come before backward()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Final predictions
print("\n--- Final Results ---")
with torch.no_grad():
    predictions = model(X)
print("Predictions:", predictions.T)
print("Targets:    ", y.T)

# ============================================================================
# SECTION 9: COMMON PATTERNS AND TIPS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 9: COMMON PATTERNS")
print("=" * 60)

# --- Pattern 1: Using nn.Sequential ---
"""
For simple networks, you can use nn.Sequential instead of defining a class:
"""
simple_model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)
print("\nnn.Sequential model:")
print(simple_model)

# --- Pattern 2: model.train() and model.eval() ---
"""
Some layers behave differently during training vs evaluation (e.g., Dropout).
Always set the mode explicitly:
"""
model.train()  # Training mode
model.eval()   # Evaluation mode (for inference)

# --- Pattern 3: torch.no_grad() for inference ---
"""
When evaluating (not training), wrap in torch.no_grad():
"""
model.eval()
with torch.no_grad():
    test_output = model(X)
# This saves memory by not tracking gradients

# ============================================================================
# SECTION 10: COMPARISON - MANUAL vs nn.Module
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 10: MANUAL vs nn.Module COMPARISON")
print("=" * 60)

print("""
┌─────────────────────────────────────┬─────────────────────────────────────┐
│         MANUAL (Task 3)             │         nn.Module (Task 4)          │
├─────────────────────────────────────┼─────────────────────────────────────┤
│ W1 = torch.randn(2,4,               │ self.layer1 = nn.Linear(2, 4)       │
│          requires_grad=True)        │                                     │
├─────────────────────────────────────┼─────────────────────────────────────┤
│ hidden = torch.sigmoid(x @ W1 + b1) │ x = torch.sigmoid(self.layer1(x))   │
├─────────────────────────────────────┼─────────────────────────────────────┤
│ loss = ((output - y)**2).mean()     │ loss = criterion(output, y)         │
├─────────────────────────────────────┼─────────────────────────────────────┤
│ W1.data -= lr * W1.grad             │ optimizer.step()                    │
│ b1.data -= lr * b1.grad             │   (updates ALL parameters)          │
│ W2.data -= lr * W2.grad             │                                     │
│ b2.data -= lr * b2.grad             │                                     │
├─────────────────────────────────────┼─────────────────────────────────────┤
│ W1.grad.zero_()                     │ optimizer.zero_grad()               │
│ b1.grad.zero_()                     │   (zeros ALL gradients)             │
│ W2.grad.zero_()                     │                                     │
│ b2.grad.zero_()                     │                                     │
└─────────────────────────────────────┴─────────────────────────────────────┘
""")

# ============================================================================
# CHEAT SHEET
# ============================================================================

print("=" * 60)
print("CHEAT SHEET")
print("=" * 60)

print("""
DEFINING A MODEL:
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(in, out)

        def forward(self, x):
            return self.layer1(x)

COMMON LAYERS:
    nn.Linear(in, out)    - Fully connected layer
    nn.Sigmoid()          - Sigmoid activation
    nn.ReLU()             - ReLU activation (most common)
    nn.Tanh()             - Tanh activation

LOSS FUNCTIONS:
    nn.MSELoss()          - Mean Squared Error
    nn.CrossEntropyLoss() - Classification loss
    nn.BCELoss()          - Binary Cross Entropy

OPTIMIZERS:
    optim.SGD(model.parameters(), lr=0.01)
    optim.Adam(model.parameters(), lr=0.001)

TRAINING LOOP TEMPLATE:
    model = MyModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1.0)

    for epoch in range(epochs):
        output = model(X)           # Forward
        loss = criterion(output, y)

        optimizer.zero_grad()       # Clear gradients
        loss.backward()             # Compute gradients
        optimizer.step()            # Update weights

EVALUATION:
    model.eval()
    with torch.no_grad():
        predictions = model(X)
""")

print("\n" + "=" * 60)
print("END OF LEARNING DOCUMENT")
print("=" * 60)
print("\nNow attempt Task 4 in task4_mlp_module.py!")
