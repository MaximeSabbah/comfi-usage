import numpy as np


def col_vector_3D(a, b, c):
    return np.array([[float(a)], [float(b)], [float(c)]], dtype=np.float64)

def orthogonalize_matrix(matrix:np.ndarray)->np.ndarray:
    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(matrix)
    # Reconstruct the orthogonal matrix
    orthogonal_matrix = U @ Vt
    # Ensure the determinant is 1
    if np.linalg.det(orthogonal_matrix) < 0:
        U[:, -1] *= -1
        orthogonal_matrix = U @ Vt
    return orthogonal_matrix

def transform_to_local_frame(D, origin, rotation_matrix):
    # Compute D relative to B
    D_relative = D - origin
    
    # Transform D to the local frame
    D_local = rotation_matrix.T @ D_relative
    
    return D_local

def transform_to_global_frame(D, origin, rotation_matrix):

    D_global =  rotation_matrix @ D + origin
    return D_global
