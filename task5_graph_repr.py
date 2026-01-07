import torch

edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

adj_matrix = torch.tensor([[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]])

print("Adjacency Matrix:")
print(adj_matrix)

edge_index = torch.tensor(edges).t()
print("\nEdge Index:")
print(edge_index)

degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
print("\nDegree Matrix:")
print(degree_matrix)

x = torch.rand((4, 3))
print("\nNode Features of x:")
print(x)