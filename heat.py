import cupy as cp
import time
import vedo
import matplotlib.cm as cm
from contribution import load_data, get_covariances, convert_scales_squared_vector

# --- (a) Load and process data ---
df = load_data("klein-bottle-gs.ply")
df = df.sample(frac=0.01)
pos_mask = ["x", "y", "z"]
means = cp.array(df[pos_mask].values)
n = means.shape[0]

def knn_adjm(means, k):
    """
    Given a Cupy array 'means' of shape (N, 3), compute the full dense 0-1
    adjacency matrix for the k nearest neighbors.
    If point i is among the k nearest neighbors of point j (or vice versa),
    then A[i, j] = -1.
    """
    N = means.shape[0]
    # Compute all pairwise squared Euclidean distances.
    diff = means[:, None, :] - means[None, :, :]
    dist2 = cp.sum(diff**2, axis=2)
    # Exclude self by setting the diagonal to infinity.
    cp.fill_diagonal(dist2, cp.inf)
    # For each point, find indices of its k smallest distances.
    knn_indices = cp.argpartition(dist2, kth=k, axis=1)[:, :k]
    rows = cp.repeat(cp.arange(N), k)
    cols = knn_indices.ravel()
    # Create the dense adjacency matrix.
    A = cp.zeros((N, N), dtype=cp.float64)
    A[rows, cols] = -1.0
    # Enforce symmetry.
    A = cp.maximum(A, A.T)
    return A

def construct_laplacian(means, k=10):
    adjm = knn_adjm(means, k)
    # Degree is the sum of absolute neighbour weights (note the negatives).
    deg = -cp.sum(adjm, axis=1)
    return cp.diag(deg) - adjm

# Compute the Laplacian.
laplacian = construct_laplacian(means)

# --- (b) Eigendecomposition and solving heat flow ---
# We assume an initial condition U0. For instance, let U0 be random:
U0 = cp.zeros(n)
U0[cp.random.choice(n, 10)] = 1.0  # Set 10 random points to 1.
# Compute the eigen-decomposition. Since laplacian is symmetric, use cp.linalg.eigh.
eigvals, V = cp.linalg.eigh(laplacian)
# eigvals is a vector of eigenvalues and V is an orthonormal matrix (columns are eigenvectors).

def heat_flow_solution(U0, t):
    """
    Given an initial condition U0 (cupy array of length n) and time t,
    returns the solution U(t) = V exp(-Λ t) Vᵀ U0,
    where Λ is the diagonal matrix of eigenvalues.
    """
    # Project U0 onto the eigenbasis.
    coeff = V.T @ U0
    # Apply the exponential decay to each coefficient.
    coeff_t = coeff * cp.exp(-eigvals * t)
    # Transform back to the original basis.
    U_t = V @ coeff_t
    return U_t

# --- (c) Visualising the solution with vedo ---
# For an animation we create a point cloud actor from the 'means'
points_np = cp.asnumpy(means)  # bring the mesh positions to host memory
# Create a vedo Points actor.
pc = vedo.Points(points_np, r=6)

# Create a Plotter and Video object.
pltter = vedo.Plotter(interactive=False)
video = vedo.Video("heat_flow_eigendecomp.mp4", backend="ffmpeg", duration=5)

# Animate over time.
for t in cp.linspace(0, 1, 200):
    U_t = heat_flow_solution(U0, t)
    # Bring the scalar field (temperature) to CPU for visualisation.
    U_np = cp.asnumpy(U_t)
    # Update the color of the points.
    pc.cmap('hot', U_np)
    pltter.show(pc, resetcam=False, interactive=False)
    video.add_frame()
    
video.close()
pltter.close()