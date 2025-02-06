import cupy as cp
from plyfile import PlyData
import pandas as pd
from scipy.spatial.transform import Rotation as R

def load_data(path):
    ply = PlyData.read(path)
    vertex_data = ply["vertex"].data
    return pd.DataFrame(vertex_data)

def convert_scales_squared_vector(scales_vector):
    n = scales_vector.shape[0]
    blanks = cp.tile(cp.eye(3), (n, 1, 1))  # Shape: (n, 3, 3)
    squared_scales = (scales_vector**2).reshape(n, 3, 1)  # Shape: (n, 3, 1)
    result = (
        blanks * squared_scales
    )  # Multiply each identity matrix by its scale vector
    return result

def get_covariances(rotations, scales_squared):
    rotations_t = rotations.transpose(0, 2, 1)
    scales_squared_t = scales_squared.transpose(0, 2, 1)
    return cp.matmul(rotations, cp.matmul(scales_squared, cp.matmul(scales_squared_t, rotations_t)))

def get_rotations(quaternions):
    n = quaternions.shape[0]
    quaternions = cp.concatenate(
        (quaternions[:,1], quaternions[:,2], quaternions[:, 3], quaternions[:,0]),
    ).reshape((n,4))    
    
    quaternions_np = quaternions.get()
    rotations = R.from_quat(quaternions_np).as_matrix()
    rotations = cp.asarray(rotations)
    return rotations
