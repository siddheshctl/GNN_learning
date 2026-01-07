import torch

# 1. Create tensor
ones_tensor_3x3 = torch.ones((3, 3))
print(ones_tensor_3x3)
rand_tensor_3x3 = torch.rand(3, 3)
print(rand_tensor_3x3)
list_tensor = torch.tensor([1,2,3,4,5])
print(list_tensor)

# 2. operations:
a = torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
print(a)
b = torch.tensor([[5, 6, 7], [7, 8, 9], [10, 11, 12]])
print(b)

mul = a @ b
print(mul)
sum = torch.sum(mul)
print(sum)

list_tensor = torch.reshape(list_tensor, [-1,1])
print(list_tensor)

# 3. Indexing
print(rand_tensor_3x3[1, :])
print(rand_tensor_3x3[0, 1])

# 4. Device awareness
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_on_device = ones_tensor_3x3.to(device)