import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import vedo

# Generate 3D points

points = pd.read_csv("normals.csv")[['x', 'y', 'z']].values
num_points = points.shape[0]

# Build KDTree for k-NN search
tree = KDTree(points)
k = 5  # Number of nearest neighbors
distances, indices = tree.query(points, k + 1)  # +1 because the closest neighbor is the point itself
indices = indices[:, 1:]  # Remove self-connections
distances = distances[:, 1:]  # Remove self-connections

# Prepare the points and lines for visualization
points_obj = vedo.Points(points, c="blue", r=8)

# Create all pairs of indices using broadcasting
i = np.repeat(np.arange(num_points), k)  # Repeat each index k times
j = indices.flatten()  # Flatten the indices of neighbors

# Get the 3D coordinates of the points from indices
i_coords = points[i]  # Coordinates for the points in i
j_coords = points[j]  # Coordinates for the points in j

# Create the lines object (edges)
lines_obj = vedo.Lines(i_coords, j_coords, c="red", lw=1)
vedo.show(lines_obj, axes=1)
