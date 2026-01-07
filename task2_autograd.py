import torch

# 1. Simple Gradient
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print("gradient x: ", x.grad)  # should print tensor([7.]) because dy/dx = 2x + 3 and at x=2, dy/dx=7

# 2. Multi-variable Gradient
a = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([2.0], requires_grad=True)

z = a * b + b ** 2

z.backward()
print("gradient a: ", a.grad)  # should print tensor([2.]) because dz/da = b
print("gradient b: ", b.grad)  # should print tensor([5.]) because dz/db = a + 2b

# 3. Gradient with Matrices
W = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64, requires_grad=True)
x = torch.tensor([[1], [2]], dtype=torch.float64, requires_grad=True)

y = W @ x
loss = y.sum()

loss.backward()
print("gradient W: \n", W.grad)  # the output is [[1., 2.], [1., 2.]] but i dont know the explanation behind it