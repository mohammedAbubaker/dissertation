import numpy as np
import pandas as pd
import vedo

# Load Data
df = pd.read_csv('normals.csv')

# Extract positions
source_positions = df[['x', 'y', 'z']].values

# Properly parse normal vectors (handling missing commas)
normals = np.vstack(df['normal'].apply(lambda s: np.fromstring(s.strip("[]"), sep=" ")).values)
normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize

# Compute end positions for normal vectors
end_positions = source_positions - 0.5 * normals  # Flip to align against surface

# Extract eigenvectors (handling incorrect formatting)
eigenvectors_0 = np.vstack(df['eigenvector_0'].apply(lambda s: np.fromstring(s.strip("[]"), sep=" ")).values)
eigenvectors_1 = np.vstack(df['eigenvector_1'].apply(lambda s: np.fromstring(s.strip("[]"), sep=" ")).values)
eigenvectors_2 = np.vstack(df['eigenvector_2'].apply(lambda s: np.fromstring(s.strip("[]"), sep=" ")).values)

eigenvalues = np.vstack(df[['eigenvalue_0', 'eigenvalue_1', 'eigenvalue_2']].values)  # Stack eigenvalues correctly

# Scale eigenvectors by sqrt(eigenvalues) for better visualization
scaled_eigenvectors = np.sqrt(eigenvalues)[:, None, :] * np.stack([eigenvectors_0, eigenvectors_1, eigenvectors_2], axis=2)

# Compute principal axis start and end points
principal_axes = []
colors = ["red", "green", "blue"]

for i in range(3):  # Loop over 3 principal axes
    start_positions = source_positions
    end_positions = source_positions + 0.03*scaled_eigenvectors[:, :, i]
    principal_axes.append(vedo.Lines(start_positions, end_positions, c=colors[i], lw=2))

# Create normal vectors as arrows
arrows = vedo.Arrows(
    source_positions,
    end_positions,
    c="black",
    alpha=0.8,
    shaft_radius=0.005,
    head_radius=0.02,
    head_length=0.05
)

# Render everything
plt = vedo.show([arrows] + principal_axes, axes=1, viewup="z", bg="white")
