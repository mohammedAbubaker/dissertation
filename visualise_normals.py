import numpy as np
import pandas as pd
import vedo
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R


def batch_quaternion_to_covariance(quaternions, scales):
    """
    Compute covariance matrices for a batch of Gaussian splats given quaternions and scale vectors.

    Parameters:
    - quaternions: (N, 4) array of quaternions (w, x, y, z)
    - scales: (N, 3) array of standard deviations along each axis (sx, sy, sz)

    Returns:
    - (N, 3, 3) array of covariance matrices
    """

    # Convert quaternions to (N, 3, 3) rotation matrices
    rot_matrices = R.from_quat(quaternions).as_matrix()  # Shape (N, 3, 3)

    # Construct diagonal scale matrices (N, 3, 3)
    scale_matrices = (scales[:, None, :] ** 2) * np.eye(3)
    scale_matrices = scale_matrices ** 2  # Square scales to get variances

    # Compute covariance matrices: Σ = R D R^T
    covariance_matrices = rot_matrices @ scale_matrices @ np.transpose(rot_matrices, (0, 2, 1))

    return covariance_matrices  # Shape (N, 3, 3)



def get_normals(df):
    # The columns in the dataframe describing the quaternion
    quaternions = df[['rot_0', 'rot_1', 'rot_2', 'rot_3']].values
    # The columns in the dataframe describing the scale vector
    scale_vectors = df[['scale_0', 'scale_1', 'scale_2']].values
    # Compute covariance matrix
    cov_matrices= batch_quaternion_to_covariance(quaternions, scale_vectors)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)  # Shape (N, 3) for eigenvalues, (N, 3, 3) for eigenvectors

    # Find the index of the smallest eigenvalue for each covariance matrix
    min_indices = np.argmax(eigenvalues, axis=1)  # Shape (N,)

    # Extract the corresponding eigenvectors, these will be our normals.
    return eigenvectors[np.arange(len(cov_matrices)), :, min_indices]

# Load the .ply file
ply = PlyData.read("bathtub.ply")
# Extract vertex data (assuming Gaussian splats are stored under 'vertex')
vertex_data = ply['vertex'].data  # Structured NumPy array
# Convert structured array to Pandas DataFrame
df = pd.DataFrame(vertex_data)

normals = get_normals(df)
source_positions = np.array(df[['x', 'y', 'z']].values)
end_positions = source_positions + 0.02 * normals

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
