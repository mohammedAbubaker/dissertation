from utils import load_data, get_covariances, convert_scales_squared_vector, get_rotations
from viewer import ViewerController
import cupy as cp

data_gs = load_data("klein-bottle-gs.ply")
data_gs = data_gs.sample(frac=0.01)

pos_mask = ["x", "y", "z"]
scales_mask = ["scale_0", "scale_1", "scale_2"]
quats_mask = ["rot_0", "rot_1", "rot_2", "rot_3"]
opacity_mask = ["opacity"]

means = cp.array(data_gs[pos_mask].values)
scales = cp.array(data_gs[scales_mask].values)
quaternions = cp.array(data_gs[quats_mask].values)
opacities = cp.array(data_gs[opacity_mask].values)
n = means.shape[0]

rotations = get_rotations(quaternions)
covariances = get_covariances(rotations, convert_scales_squared_vector(scales))
solutions = cp.zeros((n, 1))

viewer_controller = ViewerController(640, 480)
viewer_controller.add_gaussians(
    means, 
    covariances, 
    scales, 
    opacities,
    solutions
)

viewer_controller.show()