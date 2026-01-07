"""
================================================================================
PYTORCH FUNDAMENTALS - LEARNING DOCUMENT
================================================================================
Author: Your GNN Tutor
Purpose: Build PyTorch fluency from scratch before tackling GNNs

How to use this document:
1. Read each section carefully
2. Run each code block in your Python interpreter or as a script
3. Experiment by modifying values
4. Only move to Task 1 when you understand everything here
================================================================================
"""

import torch

# ============================================================================
# SECTION 1: WHAT IS PYTORCH?
# ============================================================================
"""
PyTorch is a deep learning library built around TENSORS.

A tensor is simply a multi-dimensional array (like numpy arrays, but with
superpowers for deep learning):
    - Scalar: 0D tensor (just a number)
    - Vector: 1D tensor (a list of numbers)
    - Matrix: 2D tensor (rows and columns)
    - 3D+ tensor: Higher dimensional data (images, videos, batches)

Why PyTorch over NumPy?
    1. GPU acceleration (runs on graphics cards for speed)
    2. Automatic differentiation (computes gradients for you)
    3. Built-in neural network modules
"""

# ============================================================================
# SECTION 2: CREATING TENSORS
# ============================================================================

print("=" * 60)
print("SECTION 2: CREATING TENSORS")
print("=" * 60)

# --- Method 1: From Python lists ---
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
print("\nFrom list [1,2,3,4,5]:")
print(tensor_from_list)

# --- Method 2: Pre-filled tensors ---

# All zeros
zeros = torch.zeros(3, 4)  # 3 rows, 4 columns
print("\nZeros (3x4):")
print(zeros)

# All ones
ones = torch.ones(2, 3)  # 2 rows, 3 columns
print("\nOnes (2x3):")
print(ones)

# Random values (uniform distribution between 0 and 1)
random_tensor = torch.rand(3, 3)  # 3x3 matrix
print("\nRandom (3x3):")
print(random_tensor)

# Random values (normal/Gaussian distribution)
random_normal = torch.randn(2, 2)  # mean=0, std=1
print("\nRandom Normal (2x2):")
print(random_normal)

# Identity matrix (1s on diagonal, 0s elsewhere)
identity = torch.eye(3)  # 3x3 identity
print("\nIdentity (3x3):")
print(identity)

# Range of values (like Python's range)
range_tensor = torch.arange(0, 10, 2)  # start, end, step
print("\nRange 0 to 10, step 2:")
print(range_tensor)

# ============================================================================
# SECTION 3: TENSOR PROPERTIES
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 3: TENSOR PROPERTIES")
print("=" * 60)

t = torch.rand(3, 4, 5)  # 3D tensor

print(f"\nTensor shape: {t.shape}")      # torch.Size([3, 4, 5])
print(f"Tensor size: {t.size()}")        # Same as shape
print(f"Number of dimensions: {t.dim()}")  # 3
print(f"Data type: {t.dtype}")           # torch.float32 (default)
print(f"Device: {t.device}")             # cpu (default)
print(f"Total elements: {t.numel()}")    # 3*4*5 = 60

# --- Specifying data types ---
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float64)
print(f"\nInt tensor dtype: {int_tensor.dtype}")
print(f"Float tensor dtype: {float_tensor.dtype}")

# ============================================================================
# SECTION 4: BASIC OPERATIONS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 4: BASIC OPERATIONS")
print("=" * 60)

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print("\nMatrix a:")
print(a)
print("\nMatrix b:")
print(b)

# --- Element-wise operations (shape must match) ---
print("\n--- Element-wise Operations ---")

print("\na + b (addition):")
print(a + b)  # or torch.add(a, b)

print("\na - b (subtraction):")
print(a - b)  # or torch.sub(a, b)

print("\na * b (element-wise multiplication):")
print(a * b)  # or torch.mul(a, b)
# NOTE: This is NOT matrix multiplication!

print("\na / b (element-wise division):")
print(a / b)  # or torch.div(a, b)

print("\na ** 2 (element-wise power):")
print(a ** 2)  # or torch.pow(a, 2)

# --- Matrix multiplication (IMPORTANT!) ---
print("\n--- Matrix Multiplication ---")

"""
Matrix multiplication: (m x n) @ (n x p) = (m x p)
The inner dimensions must match!

For a (2x2) @ (2x2) = (2x2):
    Result[i,j] = sum of (row i of first) * (column j of second)
"""

print("\na @ b (matrix multiplication):")
print(a @ b)  # or torch.matmul(a, b) or a.mm(b)

# Let's verify manually for position (0,0):
# a[0,:] = [1, 2], b[:,0] = [5, 7]
# Result[0,0] = 1*5 + 2*7 = 5 + 14 = 19
print("\nManual check: Result[0,0] = 1*5 + 2*7 =", 1*5 + 2*7)

# --- Aggregation operations ---
print("\n--- Aggregation Operations ---")

c = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print("\nMatrix c:")
print(c)

print(f"Sum of all elements: {c.sum()}")           # 21
print(f"Mean of all elements: {c.mean()}")         # 3.5
print(f"Max element: {c.max()}")                   # 6
print(f"Min element: {c.min()}")                   # 1

# Aggregation along specific dimension
print(f"\nSum along rows (dim=1): {c.sum(dim=1)}")     # [6, 15]
print(f"Sum along columns (dim=0): {c.sum(dim=0)}")   # [5, 7, 9]

"""
Understanding dimensions:
    For a 2D tensor with shape (rows, columns):
    - dim=0 means "operate along rows" (collapse rows, keep columns)
    - dim=1 means "operate along columns" (collapse columns, keep rows)
"""

# ============================================================================
# SECTION 5: INDEXING AND SLICING
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 5: INDEXING AND SLICING")
print("=" * 60)

"""
Indexing in PyTorch works like Python/NumPy:
    - Indices start at 0
    - Negative indices count from the end
    - Slicing uses [start:end:step] (end is exclusive)
"""

m = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
print("\nMatrix m (3x4):")
print(m)

# --- Single element access ---
print(f"\nm[0, 0] (first element): {m[0, 0]}")       # 1
print(f"m[1, 2] (row 1, col 2): {m[1, 2]}")          # 7
print(f"m[-1, -1] (last element): {m[-1, -1]}")     # 12

# --- Row/Column access ---
print(f"\nm[0] or m[0, :] (first row): {m[0]}")           # [1, 2, 3, 4]
print(f"m[1] (second row): {m[1]}")                        # [5, 6, 7, 8]
print(f"m[:, 0] (first column): {m[:, 0]}")               # [1, 5, 9]
print(f"m[:, -1] (last column): {m[:, -1]}")              # [4, 8, 12]

# --- Slicing ---
print(f"\nm[0:2, :] (first 2 rows):")
print(m[0:2, :])

print(f"\nm[:, 1:3] (columns 1 and 2):")
print(m[:, 1:3])

print(f"\nm[1:, 2:] (bottom-right corner):")
print(m[1:, 2:])

# ============================================================================
# SECTION 6: RESHAPING TENSORS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 6: RESHAPING TENSORS")
print("=" * 60)

"""
Reshaping changes how elements are arranged WITHOUT changing the data.
Total number of elements must stay the same!
"""

original = torch.arange(12)  # [0, 1, 2, ..., 11]
print(f"\nOriginal 1D tensor: {original}")
print(f"Shape: {original.shape}")

# --- reshape() ---
reshaped_3x4 = original.reshape(3, 4)
print(f"\nReshaped to 3x4:")
print(reshaped_3x4)

reshaped_4x3 = original.reshape(4, 3)
print(f"\nReshaped to 4x3:")
print(reshaped_4x3)

reshaped_2x2x3 = original.reshape(2, 2, 3)
print(f"\nReshaped to 2x2x3:")
print(reshaped_2x2x3)

# --- Using -1 for automatic dimension calculation ---
auto_reshape = original.reshape(3, -1)  # -1 means "figure it out"
print(f"\nreshape(3, -1) automatically becomes 3x4:")
print(auto_reshape)

# --- view() vs reshape() ---
"""
view() and reshape() are similar, but:
    - view() requires contiguous memory (faster, but stricter)
    - reshape() works always (may copy data if needed)
For beginners: just use reshape(), it's safer.
"""

# --- Flatten (convert to 1D) ---
matrix = torch.rand(2, 3)
flattened = matrix.flatten()
print(f"\nOriginal 2x3 matrix flattened: {flattened.shape}")

# --- Squeeze and Unsqueeze ---
"""
squeeze(): Removes dimensions of size 1
unsqueeze(): Adds a dimension of size 1
"""

# unsqueeze: add dimension
vec = torch.tensor([1, 2, 3])  # shape: (3,)
print(f"\nOriginal vector shape: {vec.shape}")

col_vec = vec.unsqueeze(1)  # shape: (3, 1) - column vector
print(f"After unsqueeze(1): {col_vec.shape}")
print(col_vec)

row_vec = vec.unsqueeze(0)  # shape: (1, 3) - row vector
print(f"After unsqueeze(0): {row_vec.shape}")
print(row_vec)

# squeeze: remove size-1 dimensions
squeezed = col_vec.squeeze()
print(f"After squeeze: {squeezed.shape}")  # back to (3,)

# ============================================================================
# SECTION 7: DEVICE MANAGEMENT (CPU vs GPU)
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 7: DEVICE MANAGEMENT")
print("=" * 60)

"""
PyTorch can run on:
    - CPU: Available on all computers
    - GPU (CUDA): NVIDIA graphics cards, much faster for large tensors

For GNNs, GPU acceleration becomes important with large graphs.
"""

# Check if CUDA (GPU) is available
cuda_available = torch.cuda.is_available()
print(f"\nCUDA available: {cuda_available}")

if cuda_available:
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")

# Best practice: use a device variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Creating tensor directly on device
tensor_on_device = torch.rand(3, 3, device=device)
print(f"\nTensor device: {tensor_on_device.device}")

# Moving existing tensor to device
cpu_tensor = torch.rand(2, 2)
moved_tensor = cpu_tensor.to(device)
print(f"Moved tensor device: {moved_tensor.device}")

# IMPORTANT: Operations require tensors on the SAME device!
# This would ERROR: cpu_tensor + tensor_on_device (if different devices)

# ============================================================================
# SECTION 8: PRACTICAL EXERCISE WALKTHROUGH
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 8: PUTTING IT ALL TOGETHER")
print("=" * 60)

"""
Let's walk through a mini-example that combines what you learned.
This simulates a simple operation you'll see in neural networks.
"""

# Imagine we have 3 data points, each with 4 features
# This is common in ML: (batch_size, num_features)
data = torch.rand(3, 4)
print("\nInput data (3 samples, 4 features):")
print(data)

# A weight matrix transforms 4 features into 2 features
# Shape: (4, 2) because (3,4) @ (4,2) = (3,2)
weights = torch.rand(4, 2)
print("\nWeight matrix (4 input features -> 2 output features):")
print(weights)

# Linear transformation: output = data @ weights
output = data @ weights
print("\nOutput (3 samples, 2 features):")
print(output)

# Add a bias term (one per output feature)
bias = torch.rand(2)
print(f"\nBias: {bias}")

output_with_bias = output + bias  # Broadcasting handles the shapes
print("\nOutput with bias:")
print(output_with_bias)

"""
This is exactly what a Linear layer (nn.Linear) does in PyTorch!
You just implemented: y = xW + b
"""

# ============================================================================
# SUMMARY: KEY FUNCTIONS CHEAT SHEET
# ============================================================================

print("\n" + "=" * 60)
print("CHEAT SHEET")
print("=" * 60)

print("""
CREATING TENSORS:
    torch.tensor([1,2,3])     - From list
    torch.zeros(m, n)         - All zeros
    torch.ones(m, n)          - All ones
    torch.rand(m, n)          - Random uniform [0,1)
    torch.randn(m, n)         - Random normal (mean=0, std=1)
    torch.eye(n)              - Identity matrix
    torch.arange(start,end)   - Range of values

PROPERTIES:
    t.shape / t.size()        - Dimensions
    t.dtype                   - Data type
    t.device                  - CPU or GPU

OPERATIONS:
    a + b, a - b              - Element-wise add/subtract
    a * b                     - Element-wise multiply (NOT matrix mult!)
    a @ b / torch.matmul(a,b) - Matrix multiplication
    t.sum(), t.mean()         - Aggregations
    t.sum(dim=0)              - Aggregate along dimension

INDEXING:
    t[0, 1]                   - Single element
    t[0, :]  or  t[0]         - First row
    t[:, 0]                   - First column
    t[0:2, 1:3]               - Slice

RESHAPING:
    t.reshape(m, n)           - Change shape
    t.reshape(m, -1)          - Auto-calculate one dimension
    t.flatten()               - Convert to 1D
    t.unsqueeze(dim)          - Add dimension
    t.squeeze()               - Remove size-1 dimensions

DEVICE:
    torch.cuda.is_available() - Check GPU
    t.to(device)              - Move tensor
""")

print("\n" + "=" * 60)
print("END OF LEARNING DOCUMENT")
print("=" * 60)
print("\nNow attempt Task 1 in task1_tensors.py!")
print("Refer back to this document as needed.")
