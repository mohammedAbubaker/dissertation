from os import wait
from plyfile import PlyData
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import vedo


import torch

def ray_intersects_bboxes(ray_ori, ray_dir, min_vecs, max_vecs):
    inv_dir = 1.0 / ray_dir
    t1 = (min_vecs - ray_ori) * inv_dir
    t2 = (max_vecs - ray_ori) * inv_dir

    t_min = torch.minimum(t1, t2)
    t_max = torch.maximum(t1, t2)

    t_near = torch.max(t_min, dim=-1).values
    t_far = torch.min(t_max, dim=-1).values
    
    return (t_near <= t_far) & (t_far >= 0)

def rays_intersect_bboxes(ray_ori, ray_dirs, min_vecs, max_vecs):
    intersections = {}
    for idx, ray_dir in enumerate(ray_dirs):

        does_intersect = ray_intersects_bboxes(
                torch.from_numpy(ray_ori),
                torch.from_numpy(ray_dir), 
                torch.from_numpy(min_vecs), 
                torch.from_numpy(max_vecs)
        )
        
        intersections[idx] = torch.where(does_intersect == True)[0]
    return intersections 

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
    
    # Grab the minimums and maximums for bounding intersection logic
    mins= np.min(corners_matrix, axis=1)
    maxs = np.max(corners_matrix, axis=1)
 
    # Work out edges
    edges_start = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6])
    edges_end = np.array([1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7,7])

    edges_start = corners_matrix[np.arange(n)[:, None], edges_start]
    edges_end = corners_matrix[np.arange(n)[:, None], edges_end]

    return edges_start, edges_end, mins, maxs

def load_data(path):
    ply = PlyData.read(path)
    vertex_data = ply["vertex"].data
    return pd.DataFrame(vertex_data)

def get_covariances(rotations, scales_squared):
    return rotations.transpose(0,2,1) @ scales_squared @ rotations

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

def generate_rays():
    camera_ori = np.array([0, 0, 0])
    camera_dir = np.array([0,1,0])

    viewport_upper_right = np.array([2,1,1])
    viewport_lower_left = np.array([-2,-1,1])
    
    res = 100
    
    viewport_centres_x = np.arange(
            viewport_lower_left[0], 
            viewport_upper_right[0],
            (viewport_upper_right[0] - viewport_lower_left[0]) / res
    )

    viewport_centres_y = np.arange(
            viewport_lower_left[1], 
            viewport_upper_right[1],
            (viewport_upper_right[1] - viewport_lower_left[1]) / res
    )

    grid_1, grid_2 = np.meshgrid(viewport_centres_x, viewport_centres_y, indexing="ij")
    viewport_grid = np.column_stack((grid_1.ravel(), grid_2.ravel()))
    z_layer_shape = (viewport_grid.shape[0], 1)
    z_layer = np.full(z_layer_shape, viewport_upper_right[2])  # Shape: NxMx1S
    full_viewport = np.concatenate((viewport_grid, z_layer), axis=1)
    return full_viewport
    raise Exception

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
    
    start_pts, end_pts, bbox_mins, bbox_maxs = create_bounding_boxes(means, covariances,0.001)
    ray_ori = np.array([0,0,0])
    ray_dirs = generate_rays()
    intersections = rays_intersect_bboxes(ray_ori, ray_dirs, bbox_mins, bbox_maxs)

    # Select a random ray
    selected_ray = 57

    selected_start_pts = start_pts[intersections[selected_ray]]
    selected_end_pts = end_pts[intersections[selected_ray]]
    print("Selected_start_pts: ", selected_start_pts.shape)
    print("Selected_end_pts: ", selected_end_pts.shape)
    
    
    non_selected_start_pts = start_pts.reshape(start_pts.shape[0] * 12, 3)
    non_selected_end_pts = end_pts.reshape(end_pts.shape[0] * 12, 3)

    # Flatten
    selected_start_pts = selected_start_pts.reshape(selected_start_pts.shape[0] * 12, 3)
    selected_end_pts = selected_end_pts.reshape(selected_end_pts.shape[0] * 12, 3)
    
    selected_bounding_boxes = vedo.Lines(selected_start_pts, selected_end_pts, lw=3)
    ray_lines = vedo.Line(ray_ori, ray_dirs[selected_ray] )
    non_selected_boxes = vedo.Lines(non_selected_start_pts, non_selected_end_pts, dotted=True)

    draw_objects = []

    draw_objects.append(selected_bounding_boxes)
    draw_objects.append(ray_lines)
    draw_objects.append(non_selected_boxes)

    vedo.show(draw_objects)
    # vedo.show([vedo.Points(viewport_centres), vedo.Point(np.array([0,0,3])), bounding_boxes])
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
