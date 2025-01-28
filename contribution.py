from os import wait
from plyfile import PlyData
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from vedo.utils import print_table


def ray_intersects_bbox(ori, dir, min_vec, max_vec):
    """
    Check if a ray intersects a bounding box.

    Parameters:
    - ori: array-like of shape (3,), origin of the ray
    - dir: array-like of shape (3,), direction of the ray
    - min_vec: array-like of shape (3,), minimum x, y, z of the bounding box
    - max_vec: array-like of shape (3,), maximum x, y, z of the bounding box

    Returns:
    - bool: True if the ray intersects the bounding box, False otherwise
    """
    inv_dir = 1.0 / dir
    t1 = (min_vec - ori) * inv_dir
    t2 = (max_vec - ori) * inv_dir

    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)

    t_near = np.max(t_min)
    t_far = np.min(t_max)

    return (t_near <= t_far) and (t_far >= 0)


def create_bounding_boxes(means, covariances, scale=1.0):
    n = means.shape[0]
    # Perform eigen decomposition to get
    # N x 3, N x 3 x 3 matrices for eigenvalues and eigenvectors respectively.
    eigenvalues, eigenvectors = np.linalg.eig(covariances)

    # Compute the corner offsets
    # N x 3, Use the sign to permute the matrix.
    axes = scale * np.sqrt(eigenvalues)

    signs = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
[-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ]
    )
    
    repeated_axes = np.repeat(axes[:, np.newaxis, :], 8, axis=1)  # Repeat each row 8 times
    signs_matrix = np.tile(signs, (n, 1,1))
    corners_matrix = repeated_axes * signs_matrix
    
    # Translate to the mean
    reshaped_means = means[:,None,:]
    corners_matrix += reshaped_means

    # Work out edges
    edges_start = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6])
    edges_end = np.array([1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7,7])

    edges_start = corners_matrix[np.arange(n)[:, None], edges_start]
    edges_end = corners_matrix[np.arange(n)[:, None], edges_end]

    # Flatten edges
    flattened_edges_start = edges_start.reshape(n*12, 3)
    flattened_edges_end = edges_end.reshape(n*12, 3)
    return flattened_edges_start, flattened_edges_end

def create_bounding_box(mean, covariance, scale=1):
    """
    Create a set of lines representing a bounding box based on mean and covariance.

    Parameters:
    - mean: array-like of shape (3,)
    - covariance: array-like of shape (3, 3)
    - scale: float, scaling factor for the size of the bounding box

    Returns:
    - bounding_box: vedo.Lines object
    """
    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Compute the corner offsets
    l, w, h = scale * np.sqrt(eigenvalues)
    corners = np.array(
        [
            [l, w, h],
            [l, w, -h],
            [l, -w, h],
            [l, -w, -h],
            [-l, w, h],
            [-l, w, -h],
            [-l, -w, h],
            [-l, -w, -h],
        ]
    )

# Rotate the corners
    rotated_corners = corners @ eigenvectors.T

    # Translate to the mean
    rotated_corners += mean

    # Define the edges between corners
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]

    start_pts = []
    end_pts = []
    for edge in edges:
        start_pts.append(rotated_corners[edge[0]])
        end_pts.append(rotated_corners[edge[1]])
    bounding_box = vedo.Lines(start_pts, end_pts, c="red")

    # Calculate minimum and maximum points
    min_vec = np.min(rotated_corners, axis=0)
    max_vec = np.max(rotated_corners, axis=0)
    return bounding_box, min_vec, max_vec


def calculate_batched_box_wireframes(min_vecs, max_vecs):
    """
    Compute wireframes for multiple boxes in a fully vectorized manner.

    Parameters:
        min_vecs: np.array of shape (N, 3), minimum x, y, z for each box.
        max_vecs: np.array of shape (N, 3), maximum x, y, z for each box.

    Returns:
        edges: np.array of shape (N, 12, 2, 3), where:
               - N is the number of boxes
               - 12 is the number of edges per box
               - 2 is the start and end point of each edge
               - 3 is the 3D coordinates of the point
    """
    # Ensure inputs are valid
    assert min_vecs.shape == max_vecs.shape, (
        "min_vecs and max_vecs must have the same shape"
    )
    assert min_vecs.shape[1] == 3, "Each vector must have 3 components (x, y, z)"

    # Generate all 8 corners for each box using broadcasting
    corners = np.stack(
        [
            np.array(
                [min_vecs[:, 0], min_vecs[:, 1], min_vecs[:, 2]]
            ).T,  # [x_min, y_min, z_min]
            np.array(
                [max_vecs[:, 0], min_vecs[:, 1], min_vecs[:, 2]]
            ).T,  # [x_max, y_min, z_min]
            np.array(
                [max_vecs[:, 0], max_vecs[:, 1], min_vecs[:, 2]]
            ).T,  # [x_max, y_max, z_min]
            np.array(
                [min_vecs[:, 0], max_vecs[:, 1], min_vecs[:, 2]]
            ).T,  # [x_min, y_max, z_min]
            np.array(
                [min_vecs[:, 0], min_vecs[:, 1], max_vecs[:, 2]]
            ).T,  # [x_min, y_min, z_max]
            np.array(
                [max_vecs[:, 0], min_vecs[:, 1], max_vecs[:, 2]]
            ).T,  # [x_max, y_min, z_max]
            np.array(
                [max_vecs[:, 0], max_vecs[:, 1], max_vecs[:, 2]]
            ).T,  # [x_max, y_max, z_max]
            np.array(
                [min_vecs[:, 0], max_vecs[:, 1], max_vecs[:, 2]]
            ).T,  # [x_min, y_max, z_max]
        ],
        axis=1,
    )  # Shape: (N, 8, 3)

    # Define the 12 edges using indices of the corners
    edge_indices = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # Bottom face
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # Top face
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # Vertical edges
        ]
    )  # Shape: (12, 2)

    # Gather corner coordinates for the edges
    edges = corners[:, edge_indices]  # Shape: (N, 12, 2, 3)

    return edges


def load_data(path):
    ply = PlyData.read(path)
    vertex_data = ply["vertex"].data
    return pd.DataFrame(vertex_data)

def get_covariances(rotations, scales_squared):
    return rotations @ scales_squared @ rotations.transpose(0,2,1)

def evaluate_positions(positions, means, covariances):
    """
    Evaluate the value of a 3D Gaussian at given positions.

    Parameters:
        positions (ndarray): shape (N,3). Each row is a 3D point [x, y, z].
        means (ndarray): shape (N,3). Each row is the mean of a Gaussian.
        covariances (ndarray): shape (N,3,3). Covariance matrix for each Gaussian.

    Returns:
        values (ndarray): shape (N,). The 3D Gaussian values at each position.
    """
    # Perform standardisation
    pos_mean = np.mean(positions, axis=0)
    pos_std = np.std(positions, axis=0)
    standardized_positions = (positions - pos_mean) / pos_std
    standardized_means = (means - pos_mean) / pos_std

    # Vector from mean to each position
    diff = standardized_positions - standardized_means
    diff = diff[:, :, None]
    # Invert covariance matrices
    inv_covs = np.linalg.inv(covariances)  # (N,3,3)
    exponents = diff.transpose(0, 2, 1) @ inv_covs @ diff
    weights = np.exp(-0.5 * exponents).squeeze()
    return weights


def get_contributions(ori, dir, intersected_gaussians):
    """
    @params
    intersected_gaussians: [11,N] where N is the number of intersected gaussians.
    """
    N = intersected_gaussians.shape[0]

    means = intersected_gaussians[:, 0:3]
    quaternions = intersected_gaussians[:, 3:7]
    scales_vector = intersected_gaussians[:, 7:10]
    # Retrieve covariances
    covs = get_covariances(quaternions, scales_vector)
    t_values = max_gaussian_along_ray(ori, dir, means, covs)
    positions = ori + t_values[:, None] * dir
    return evaluate_positions(positions, means, covs)


def max_gaussian_along_ray(o, d, means, covs):
    """
    Computes t for which each 3D Gaussian has its maximum along ray x(t) = o + t*d.

    Parameters:
        o (array-like of shape (3,)): Origin of the ray.
        d (array-like of shape (3,)): Direction of the ray (will be normalized).
        means (ndarray of shape (N,3)): Means of the N Gaussians.
        covs (ndarray of shape (N,3,3)): Covariance matrices of the N Gaussians.

    Returns:
        t_values (ndarray of shape (N,)): Parameter t at which each Gaussian is maximal.
                                          If t < 0, that maximum is "behind" the origin
                                          relative to d.
    """
    o = np.asarray(o, dtype=float)  # (3,)
    d = np.asarray(d, dtype=float)  # (3,)
    d /= np.linalg.norm(d)  # Normalize ray direction

    means = means.astype(float)  # (N,3)
    covs = covs.astype(float)  # (N,3,3)
    N = means.shape[0]

    # Compute Σ⁻¹ for each Gaussian
    inv_covs = np.linalg.inv(covs)  # shape (N,3,3)
    # (o - μ) => shape (N,3)
    o_minus_mu = o[None, :] - means  # broadcast origin to each Gaussian
    # We need (o - μ)ᵀ Σ⁻¹ d and dᵀ Σ⁻¹ d for each Gaussian.

    # Step 1) Let Aᵢ = Σᵢ⁻¹ d => shape (N,3)
    # We can do this with einstein summation or matmul:
    #   inv_covs has shape (N,3,3), d has shape (3,)
    A = np.einsum("nij,j->ni", inv_covs, d)  # shape (N,3)
    # Step 2) dᵀ Σᵢ⁻¹ d => shape (N,)
    d_invCov_d = np.einsum("ni,i->n", A, d)  # dot each (N,3) with (3,)

    # Step 3) (o - μ)ᵀ Σᵢ⁻¹ d => shape (N,)
    o_minus_mu_invCov_d = np.einsum("ni,ni->n", o_minus_mu, A)

    # Step 4) Solve for tᵢ
    #   tᵢ = - [ (o - μᵢ)ᵀ Σᵢ⁻¹ d ] / [ dᵀ Σᵢ⁻¹ d ]
    eps = 1e-12  # small offset in case any d_invCov_d = 0
    t_values = -o_minus_mu_invCov_d / (d_invCov_d + eps)

    return t_values

def convert_scales_vector(scales_vector):
    n = scales_vector.shape[0]
    blanks = np.tile(np.eye(3), (n, 1, 1))  # Shape: (n, 3, 3)
    squared_scales = (scales_vector ** 2).reshape(n, 3, 1)  # Shape: (n, 3, 1)
    result = blanks * squared_scales  # Multiply each identity matrix by its scale vector
    return result

if __name__ == "__main__":
    import vedo

    df = load_data("bathtub.ply")
    pos_mask = ["x", "y", "z"]
    quats_mask = ["rot_0", "rot_1", "rot_2", "rot_3"]
    scales_mask = ["scale_0", "scale_1", "scale_2"]
    opacity_mask = ["opacity"]
    col_mask = [*pos_mask, *quats_mask, *scales_mask, opacity_mask]

    means = df[pos_mask].to_numpy()
    quaternions = df[quats_mask].to_numpy()
    scales_vector = df[scales_mask].to_numpy()
    opacity = df[opacity_mask].to_numpy()
    
    # Normalise the quaternions
    quaternions = quaternions  / np.linalg.norm(quaternions)
    rotations = R.from_quat(quaternions).as_matrix()
    
    scales_squared = convert_scales_vector(scales_vector)
    covariances = get_covariances(rotations, scales_squared)
    
    start_pts, end_pts = create_bounding_boxes(means, covariances,0.01)
    vizz_num = 300
    bounding_boxes = vedo.Lines(start_pts[:vizz_num*12], end_pts[:vizz_num*12], c='red')
    vedo.show(bounding_boxes)
    raise Exception


    intersected_gaussians = (df[col_mask].to_numpy())[:1000, :]
    ori = np.array([0, 0, 0])
    # set dir to point to the first gaussian
    dir = intersected_gaussians[0, 0:3] - ori
    dir = dir / np.linalg.norm(dir)
    # contributions = get_contributions(ori, dir, intersected_gaussians)
    weights = get_contributions(ori, dir, intersected_gaussians)
    print(weights)
    # Visualise the ray infinte line
    line = vedo.Line(ori, ori + 1 * dir, lw=3)
    spheres = vedo.Spheres(
        centers=intersected_gaussians[:, 0:3], c="red5", r=weights * 0.1
    )
    points = vedo.Points(intersected_gaussians[:, 0:3], c="blue")
    means = intersected_gaussians[:, 0:3]
    quaternions = intersected_gaussians[:, 3:7]
    scales_vector = intersected_gaussians[:, 7:10]
    covariances = get_covariances(quaternions, scales_vector)
    edges = calculate_batched_box_wireframes(*get_bounding_box(means, covariances))
    # Flatten the edges to a single array
    edges = edges.reshape(-1, 2, 3)
    # Create a line object from the edges
    bb_start_pts = edges[:, 0]
    bb_end_pts = edges[:, 1]

    bb = vedo.Line(bb_start_pts, bb_end_pts, c="green")
    vedo.show([spheres, line, points, bb])
