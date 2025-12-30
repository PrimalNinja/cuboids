import torch
import math

def discoveryone_detection(data):
    """
    Baseline: Detect Discovery One ships and collisions in 3D space
    Input: N x N x N voxel grid (e.g., 30x30x30)
    Output: [num_ships, num_collisions]
    
    A Discovery One ship is detected as a linear pattern of 3+ occupied voxels
    A collision is when two ships occupy overlapping voxels
    """
    data_flat = data.flatten()
    N = int(round(len(data_flat) ** (1/3)))
    
    # Reshape to 3D grid
    grid = data.reshape(N, N, N)
    
    # Count occupied voxels (ships are marked with value > 0.5)
    occupied = (grid > 0.5).float()
    num_occupied = occupied.sum().item()
    
    # Estimate number of ships (assuming ~10 voxels per ship)
    estimated_ships = num_occupied / 10.0
    
    # Detect collisions: voxels with value > 1.5 indicate multiple ships
    collisions = (grid > 1.5).float()
    num_collisions = collisions.sum().item()
    
    # Return as tensor
    result = torch.tensor([estimated_ships, num_collisions], device=data.device)
    return result