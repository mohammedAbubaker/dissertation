import pickle
import vedo
import numpy as np

# Load the Pickle file (each key is a point, each value is a list of neighbor 3D points)
with open("within_k_vectors.pkl", "rb") as f:
    within_k_vectors = pickle.load(f)

# Extract all unique points & their neighbors in a fully vectorized way
points = np.array(list(within_k_vectors.keys()))  # These are point indices
neighbors = list(within_k_vectors.values())  # This is a list of lists of 3D points

# Flatten the neighbors array (convert list of lists into a single NumPy array)
flat_neighbors = np.concatenate(neighbors, axis=0)  # Fast bulk conversion

# Create a NumPy array of sources (repeating each source for its neighbors)
source_repeated = np.repeat(points, [len(n) for n in neighbors], axis=0)

# Use vedo.Points for efficient rendering of all points
points_actor = vedo.Points(points, r=4, c="red")

# Use vedo.Lines for bulk line rendering between sources and their neighbors
if len(flat_neighbors) > 0:
    lines_actor = vedo.Lines(source_repeated, flat_neighbors, c="white", lw=1)
else:
    lines_actor = None  # No lines if there are no neighbors

# Render using Vedo (fast interactive 3D)
plotter = vedo.Plotter(title="3D Neighbor Visualization", axes=1, bg="black")
plotter.show([points_actor, lines_actor] if lines_actor else [points_actor], interactive=True)

