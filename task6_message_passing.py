import torch

def aggregate_neighbours(adj_matrix, node_features):
    return adj_matrix @ node_features

def normalize_aggregation(adj_matrix, node_features):
    degree_inv = 1.0 / torch.sum(adj_matrix, dim=1, keepdim=True)  # Shape (4, 1) used in normalization
    return degree_inv * (adj_matrix @ node_features)  # Broadcasting


def message_passing_layer(adj_matrix, node_features, weight_matrix):
    trans_matrix = node_features @ weight_matrix #transforms the matrix based on the weights
    aggr_matrix = adj_matrix @ (node_features @ weight_matrix) # aggregates with the neighbours
    return aggr_matrix

# 4-node cycle: 0→1→2→3→0
adj_matrix = torch.tensor([[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]], dtype=torch.float32)

node_features = torch.tensor([[1.0, 0.0],
                              [0.0, 1.0],
                              [1.0, 1.0],
                              [0.5, 0.5]], dtype=torch.float32)

W = torch.randn(2, 3)

print("=== Function 1: Simple Aggregation ===")
result1 = aggregate_neighbours(adj_matrix, node_features)
print(result1)
print("Shape:", result1.shape)

print("\n=== Function 2: Normalized Aggregation ===")
result2 = normalize_aggregation(adj_matrix, node_features)
print(result2)
print("Shape:", result2.shape)

print("\n=== Function 3: Message Passing Layer ===")
result3 = message_passing_layer(adj_matrix, node_features, W)
print(result3)
print("Shape:", result3.shape)

print("\n=== Verification: Normalized < Unnormalized ===")
print("Unnormalized sum:", result1.sum(dim=1))
print("Normalized sum:", result2.sum(dim=1))
