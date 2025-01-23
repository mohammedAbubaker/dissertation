from plyfile import PlyData
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_data(path):
    ply = PlyData.read(path)
    vertex_data = ply["vertex"].data
    return pd.DataFrame(vertex_data)

def get_covariances(quaternions, scales_vector):
    """
    @params
    quaternions: np.array, of shape (Nx4). Quaternions of the splats.
    scales_vector: np.array, of shape (Nx3). Scales of the splats.
    @returns
    covariances: np.array, of shape (Nx3x3). Covariances of the splats.
    """
    rotations = R.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
    scales = scales_vector[:,None] * np.eye(3)
    covariances =  rotations @ scales @ scales.transpose(0,2,1) @ rotations.transpose(0, 2, 1)
    return covariances

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
    # Vector from mean to each position
    diff = (positions - means) 
    diff = diff[:, :, None] 
    # Invert covariance matrices
    inv_covs = np.linalg.inv(covariances)  # (N,3,3)
    exponents = diff.transpose(0,2,1) @ inv_covs @ diff
    weights = np.exp(-0.5*exponents).squeeze()
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
    d /= np.linalg.norm(d)          # Normalize ray direction

    means = means.astype(float)     # (N,3)
    covs = covs.astype(float)       # (N,3,3)
    N = means.shape[0]

    # Compute Σ⁻¹ for each Gaussian
    inv_covs = np.linalg.inv(covs)  # shape (N,3,3)
    # (o - μ) => shape (N,3)
    o_minus_mu = o[None, :] - means  # broadcast origin to each Gaussian
    # We need (o - μ)ᵀ Σ⁻¹ d and dᵀ Σ⁻¹ d for each Gaussian.

    # Step 1) Let Aᵢ = Σᵢ⁻¹ d => shape (N,3)
    # We can do this with einstein summation or matmul:
    #   inv_covs has shape (N,3,3), d has shape (3,)
    A = np.einsum('nij,j->ni', inv_covs, d)  # shape (N,3)
    # Step 2) dᵀ Σᵢ⁻¹ d => shape (N,)
    d_invCov_d = np.einsum('ni,i->n', A, d)  # dot each (N,3) with (3,)

    # Step 3) (o - μ)ᵀ Σᵢ⁻¹ d => shape (N,)
    o_minus_mu_invCov_d = np.einsum('ni,ni->n', o_minus_mu, A)

    # Step 4) Solve for tᵢ
    #   tᵢ = - [ (o - μᵢ)ᵀ Σᵢ⁻¹ d ] / [ dᵀ Σᵢ⁻¹ d ]
    eps = 1e-12  # small offset in case any d_invCov_d = 0
    t_values = - o_minus_mu_invCov_d / (d_invCov_d + eps)

    return t_values

if __name__ == "__main__":
    import vedo

    df = load_data("bathtub.ply")
    col_mask = [
        "x",
        "y",
        "z",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "scale_0",
        "scale_1",
        "scale_2",
        "opacity",
    ]

    intersected_gaussians = (df[col_mask].to_numpy())[:1000, :]
    ori = np.array([0,0,0])
    # set dir to point to the first gaussian
    dir = intersected_gaussians[0, 0:3] - ori
    dir = dir / np.linalg.norm(dir)
    # contributions = get_contributions(ori, dir, intersected_gaussians)
    weights = get_contributions(ori, dir, intersected_gaussians)
    print(weights)
    # Visualise the ray infinte line
    line = vedo.Line(ori, ori + 1* dir, lw=3)
    spheres = vedo.Spheres(
        centers=intersected_gaussians[:, 0:3], 
        c="red5",
        r=weights*0.1
    )
    vedo.show([spheres, line])
