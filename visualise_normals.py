import numpy as np
import pandas as pd
import vedo

# Generate Data
df = pd.read_csv('normals.csv')
source_positions = df[['x', 'y', 'z']].values
direction_vectors = df[['n_x', 'n_y', 'n_z']].values
direction_vectors /= np.linalg.norm(direction_vectors, axis=1, keepdims=True)
end_positions = source_positions - 0.03 * direction_vectors

# Create Arrows
arrows = vedo.Arrows(
        source_positions,
        end_positions,
        c="blue",
        alpha=0.7,
        shaft_radius=0.005,
        head_radius=0.02,
        head_length=0.1
)

plt = vedo.show(arrows, axes=1, viewup="z", bg="white")
