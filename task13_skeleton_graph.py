import torch
import torch.nn as nn
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t()
# print(edge_index.shape)

def generate_random_pose(num_joints=16):
    """
    Generate a random 3D human pose.
    
    Returns:
        Tensor of shape (num_joints, 3)
        Each row is (x, y, z) for one joint
    """
    
    pose = torch.randn(num_joints, 3) * 0.5
    return pose

def generate_meaningful_pose():
    """
    Generate a human-like standing pose in 3D.
    Returns: (16, 3) tensor
    """

    pose = torch.tensor([
        [0.0,  1.7,  0.0],   # 0 head
        [0.0,  1.5,  0.0],   # 1 neck

        [-0.3, 1.5,  0.0],   # 2 left_shoulder
        [ 0.3, 1.5,  0.0],   # 3 right_shoulder

        [-0.5, 1.2,  0.0],   # 4 left_elbow
        [ 0.5, 1.2,  0.0],   # 5 right_elbow

        [-0.6, 0.9,  0.0],   # 6 left_wrist
        [ 0.6, 0.9,  0.0],   # 7 right_wrist

        [0.0,  1.2,  0.0],   # 8 spine
        [0.0,  1.0,  0.0],   # 9 hips

        [-0.2, 1.0,  0.0],   # 10 left_hip
        [ 0.2, 1.0,  0.0],   # 11 right_hip

        [-0.2, 0.6,  0.0],   # 12 left_knee
        [ 0.2, 0.6,  0.0],   # 13 right_knee

        [-0.2, 0.2,  0.0],   # 14 left_ankle
        [ 0.2, 0.2,  0.0],   # 15 right_ankle
    ])

    return pose


def generate_action_sequence(num_frames=30, num_joints=16):
    """
    Generate a sequence of skeleton poses simulating motion.
    
    Returns:
        Tensor of shape (num_frames, num_joints, 3)
    """
    sequence = []

    current_pose = generate_random_pose(num_joints)

    for _ in range(num_frames):
        # Small smooth motion between frames
        noise = torch.randn(num_joints, 3) * 0.05
        current_pose = current_pose + noise
        sequence.append(current_pose)

    return torch.stack(sequence)

def skeleton_to_pyg_data(joint_positions, edge_index, label=None):
    """
    Convert skeleton to PyG Data object
    
    Args:
        joint_positions: (num_joints, 3) tensor
        edge_index: (2, num_edges) tensor
        label: action class (optional, scalar)
    
    Returns:
        torch_geometric.data.Data object
    """
    data = Data(
        x=joint_positions,      
        edge_index=edge_index   
    )

    if label is not None:
        data.y = torch.tensor(label, dtype=torch.long)

    return data

def visualize_skeleton(joint_positions, edges):
    """
    Plot a human skeleton in 3D.
    
    Args:
        joint_positions: Tensor or array of shape (num_joints, 3)
        edges: list of (i, j) tuples defining skeleton connections
    """
    # Convert to numpy if tensor
    if hasattr(joint_positions, "detach"):
        joint_positions = joint_positions.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot joints
    xs = joint_positions[:, 0]
    ys = joint_positions[:, 1]
    zs = joint_positions[:, 2]
    ax.scatter(xs, ys, zs, s=50)

    # Plot bones (edges)
    for i, j in edges:
        x = [joint_positions[i, 0], joint_positions[j, 0]]
        y = [joint_positions[i, 1], joint_positions[j, 1]]
        z = [joint_positions[i, 2], joint_positions[j, 2]]
        ax.plot(x, y, z)

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Skeleton")

    # Equal scaling for better visualization
    max_range = (joint_positions.max() - joint_positions.min()) / 2
    mid = joint_positions.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.show()

# Create edge index
print(f"Edge index shape: {edge_index.shape}")

# Generate random pose
pose = generate_meaningful_pose()
print(f"Pose shape: {pose.shape}")

# Create PyG data
data = skeleton_to_pyg_data(pose, edge_index, label=0)
print(f"Data: {data}")
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")

visualize_skeleton(pose, SKELETON_EDGES)
