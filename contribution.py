from os import wait
from plyfile import PlyData
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import vedo
import cupy as torch
def ray_intersects_bboxes(ray_ori, ray_dir, min_vecs, max_vecs):
    inv_dir = 1.0 / (ray_dir + 1e-8)
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
            torch.from_numpy(max_vecs),
        )
        ray_intersections = torch.where(does_intersect == True)[0]
        intersections[idx] = ray_intersections.reshape(ray_intersections.shape[0])  
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

def create_bounding_boxes(means, covariances, scales):
    n = means.shape[0]
    # Perform eigen decomposition to get
    # N x 3, N x 3 x 3 matrices for eigenvalues and eigenvectors respectively.
    eigenvalues, eigenvectors = np.linalg.eig(covariances)

    # Compute the corner offsets
    # N x 3, Use the sign to permute the matrix.
    axes = torch.bmm() 

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

    repeated_axes = np.repeat(
        axes[:, np.newaxis, :], 8, axis=1
    )  # Repeat each row 8 times
    signs_matrix = np.tile(signs, (n, 1, 1))
    corners_matrix = repeated_axes * signs_matrix

    # Translate to the mean
    reshaped_means = means[:, None, :]
    corners_matrix += reshaped_means

    # Grab the minimums and maximums for bounding intersection logic
    mins = np.min(corners_matrix, axis=1)
    maxs = np.max(corners_matrix, axis=1)

    # Work out edges
    edges_start = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6])
    edges_end = np.array([1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7])

    edges_start = corners_matrix[np.arange(n)[:, None], edges_start]
    edges_end = corners_matrix[np.arange(n)[:, None], edges_end]

    return edges_start, edges_end, mins, maxs

def load_data(path):
    ply = PlyData.read(path)
    vertex_data = ply["vertex"].data
    return pd.DataFrame(vertex_data)

def get_covariances(rotations, scales_squared):
    return rotations.transpose(0, 2, 1) @ scales_squared @ rotations

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
    pos_mean = torch.mean(positions, dim=0)
    pos_std = torch.std(positions, dim=0)
    standardized_positions = (positions - pos_mean) / pos_std
    standardized_means = (means - pos_mean) / pos_std

    # Vector from mean to each position
    diff = standardized_positions - standardized_means
    # Invert covariance matrices
    inv_covs = torch.linalg.inv(covariances)  # (N,3,3)
    exponents = torch.bmm(torch.bmm(diff[:,None,:], inv_covs), diff[:,:,None])
    weights = torch.exp(-0.5 * exponents)
    return weights

def get_contributions(ori, dir, means, covariances):
    """
    @params
    intersected_gaussians: [11,N] where N is the number of intersected gaussians.
    """
    n = means.shape[0]
    t_values = max_gaussian_along_ray(ori, dir, means, covariances)
    positions = ori + t_values * dir
    evaluated_positions = evaluate_positions(positions, means, covariances)

    # Front to back compositing
    # Sort the gaussians
    sorted_indices = torch.argsort(means[:,2])
    sorted_values = evaluated_positions[sorted_indices]
    contributions = torch.zeros(n,1)
    accumulated_opacity = 0
    for i, value in enumerate(sorted_values):
        contributions[i] = value * (1-accumulated_opacity) 
        accumulated_opacity += value * (1 - accumulated_opacity)
    return contributions 

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
    n = means.shape[0]
    inv_covs = torch.linalg.inv(covs)
    dx_t = (means - o)[:,None,:]
    dx_t = dx_t.to(torch.float64)
    numerator = torch.bmm(torch.bmm(dx_t, inv_covs), torch.tile(d, (n,1))[:,:,None])
    denomenator = torch.bmm(torch.bmm(torch.tile(d, (n, 1))[:,None,:], inv_covs), torch.tile(d, (n,1))[:,:,None])
    return (numerator / denomenator).reshape(n,1)

def convert_scales_squared_vector(scales_vector):
    n = scales_vector.shape[0]
    blanks = torch.tile(torch.eye(3), (n, 1, 1))  # Shape: (n, 3, 3)
    squared_scales = (scales_vector**2).reshape(n, 3, 1)  # Shape: (n, 3, 1)
    result = (
        blanks * squared_scales
    )  # Multiply each identity matrix by its scale vector
    return result

def convert_scales_vector(scales_vector):
    n = scales_vector.shape[0]
    blanks = torch.tile(torch.eye(3), (n, 1, 1))  # Shape: (n, 3, 3)
    squared_scales = (scales_vector).reshape(n, 3, 1)  # Shape: (n, 3, 1)
    result = (
        blanks * squared_scales
    )  # Multiply each identity matrix by its scale vector
    return result

def test(intersections, ray_dirs, ray_ori, means, covariances):
    n = means.shape[0]
    sorted_means = means[:, 2].argsort()
    contributions = torch.zeros(n,1)
    for ray_idx, gaussians_idx in intersections.items():
        contributions[gaussians_idx] = get_contributions(ray_ori, ray_dirs[ray_idx], means[gaussians_idx], covariances[gaussians_idx])
    return contributions
        

def generate_rays(ray_ori):
    viewport_upper_right = np.array([2+ray_ori[0], 1 + ray_ori[1], 1 + ray_ori[2]])
    viewport_lower_left = np.array([-2+ray_ori[0], -1 + ray_ori[1], 1 + ray_ori[2]])
    res = 100

    viewport_centres_x = np.arange(
        viewport_lower_left[0],
        viewport_upper_right[0],
        (viewport_upper_right[0] - viewport_lower_left[0]) / res,
    )

    viewport_centres_y = np.arange(
        viewport_lower_left[1],
        viewport_upper_right[1],
        (viewport_upper_right[1] - viewport_lower_left[1]) / res,
    )

    grid_1, grid_2 = np.meshgrid(viewport_centres_x, viewport_centres_y, indexing="ij")
    viewport_grid = np.column_stack((grid_1.ravel(), grid_2.ravel()))
    z_layer_shape = (viewport_grid.shape[0], 1)
    z_layer = np.full(z_layer_shape, viewport_upper_right[2])  # Shape: NxMx1S
    full_viewport = np.concatenate((viewport_grid, z_layer), axis=1)
    return full_viewport

if __name__ == "__main__":
    import vedo

    df = load_data("bathtub.ply")
    pos_mask = ["x", "y", "z"]
    quats_mask = ["rot_0", "rot_1", "rot_2", "rot_3"]
    scales_mask = ["scale_0", "scale_1", "scale_2"]
    opacity_mask = ["opacity"]
    col_mask = [*pos_mask, *quats_mask, *scales_mask, opacity_mask]

    means = df[pos_mask].to_numpy()
    n = means.shape[0]
    quaternions = df[quats_mask].to_numpy()
    scales_vector = df[scales_mask].to_numpy()
    opacity = df[opacity_mask].to_numpy()

    # Normalise the quaternions
    quaternions = quaternions / np.linalg.norm(quaternions)

    rotations = R.from_quat(quaternions).as_matrix()

    scales = convert_scales_vector(torch.array(scales_vector.data))
    scales_squared = convert_scales_squared_vector(scales_vector)
    covariances = get_covariances(rotations, scales_squared)

    start_pts, end_pts, bbox_mins, bbox_maxs = create_bounding_boxes(
        means, covariances, scales
    )
    ray_ori = np.array([0, 0, 0])
    ray_dirs = generate_rays(ray_ori)
    intersections = rays_intersect_bboxes(ray_ori, ray_dirs, bbox_mins, bbox_maxs)
    contributions = test(
        intersections, 
        torch.from_numpy(ray_dirs), 
        torch.from_numpy(ray_ori), 
        torch.from_numpy(means), 
        torch.from_numpy(covariances),
    )

    draw_objects = []
    # draw_objects.append(vedo.Spheres(centers=means, r=contributions*0.01))
    # draw_objects.append(vedo.Lines(start_pts=np.tile(ray_ori, (ray_dirs.shape[0], 1)), end_pts=ray_dirs))
    #draw_objects.append(vedo.Points(means))
    # draw_objects.append(vedo.Lines(start_pts=start_pts.reshape(-1, 3), end_pts=end_pts.reshape(-1,3)))
    for ray_idx, gaussians_idx in intersections.items():
        box_lines = vedo.Lines(start_pts[gaussians_idx].reshape(-1,3), end_pts[gaussians_idx].reshape(-1,3)) 
        draw_objects.append(box_lines)

    vedo.show(draw_objects)
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
