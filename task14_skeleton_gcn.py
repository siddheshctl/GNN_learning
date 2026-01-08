import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# Human skeleton connections (edges)
# Using a simplified 15-joint skeleton
SKELETON_EDGES = [
    (0, 1),   # head -> neck
    (1, 2),   # neck -> left_shoulder
    (1, 3),   # neck -> right_shoulder
    (2, 4),   # left_shoulder -> left_elbow
    (3, 5),   # right_shoulder -> right_elbow
    (4, 6),   # left_elbow -> left_wrist
    (5, 7),   # right_elbow -> right_wrist
    (1, 8),   # neck -> spine
    (8, 9),   # spine -> hips
    (9, 10),  # hips -> left_hip
    (9, 11),  # hips -> right_hip
    (10, 12), # left_hip -> left_knee
    (11, 13), # right_hip -> right_knee
    (12, 14), # left_knee -> left_ankle
    (13, 15), # right_knee -> right_ankle
]

JOINT_NAMES = [
    'head', 'neck', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'spine', 'hips', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def create_skeleton_edge_index():
    return torch.tensor(SKELETON_EDGES, dtype=torch.long).t()

def create_synthetic_dataset(num_samples=100, num_classes=4):
    """
    Create fake skeleton data for 4 actions:
    - Class 0: Arms up (high y-values for wrists)
    - Class 1: Arms down (low y-values for wrists)
    - Class 2: Leaning left (lower x-values overall)
    - Class 3: Leaning right (higher x-values overall)
    
    Returns:
        list of PyG Data objects
    """
    dataset = []
    for i in range(num_samples):
        label = i % num_classes
        
        # Base skeleton (standing pose)
        pose = torch.randn(16, 3) * 0.1  # Small random noise
        
        # Add class-specific patterns
        if label == 0:  # Arms up
            pose[6, 1] += 1.0  # left_wrist y
            pose[7, 1] += 1.0  # right_wrist y
        elif label == 1:  # Arms down
            pose[6, 1] -= 1.0
            pose[7, 1] -= 1.0
        elif label == 2:  # Lean left
            pose[:, 0] -= 0.5  # All joints shift left
        elif label == 3:  # Lean right
            pose[:, 0] += 0.5  # All joints shift right
        
        data = Data(
            x=pose,
            edge_index=create_skeleton_edge_index(),
            y=torch.tensor([label])
        )
        dataset.append(data)
    
    return dataset

dataset = create_synthetic_dataset()

print("Number of Graphs:", len(dataset))

train_dataset = dataset[:80]
test_dataset = dataset[80:]

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class SkeletonClassifier(nn.Module):
    def __init__(self, in_features, hidden_features1, hidden_features2, out_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_features1)
        self.conv2 = GCNConv(hidden_features1, hidden_features2)
        self.classifier = nn.Linear(hidden_features2, out_features)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)

        x = self.classifier(x)

        return torch.log_softmax(x, dim=1)
    
model = SkeletonClassifier(in_features=3,
                           hidden_features1=32,
                           hidden_features2=64,
                            out_features=4)
optimizer= torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

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

with torch.no_grad():
    model.eval()
    correct = 0
    total = 0

    for batch in test_loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    
    print(f"Test Accuracy: {correct/total:.4f}")
