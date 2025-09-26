import subprocess
import numpy as np
import os  
import cv2 as cv
import yaml
from utils.utils import load_transformation
import pinocchio as pin 

def rt_to_homogeneous(R, T):
    """Convert (R, T) to a 4x4 homogeneous transformation matrix."""
    T = T.reshape(3,)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = T
    return H

def invert_homogeneous(T):
    """Invert a 4x4 homogeneous transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def decompose_homogeneous(H):
    """Extract (R, T) from a 4x4 homogeneous matrix."""
    R = H[:3, :3]
    T = H[:3, 3]
    return R, T

import numpy as np

def get_camera_params(Ks, Ds, Rs=None, Ts=None):
    """
    Build camera parameters for N cameras.

    Parameters
    ----------
    Ks : list/tuple of (3x3) intrinsics
    Ds : list/tuple of distortion arrays
    Rs : list/tuple of (3x3) rotations, optional
         If None, identities are used. Cam0 is always identity.
    Ts : list/tuple of (3,) or (3,1) translations, optional
         If None, zeros are used. Cam0 is always zeros.

    Returns
    -------
    mtxs : list of np.ndarray (3x3)
    dists : list of np.ndarray
    projections : list of np.ndarray (3x4) [R | t]
    rotations : list of np.ndarray (3x3)
    translations : list of np.ndarray (3x1)
    """
    n = len(Ks)
    if len(Ds) != n:
        raise ValueError("Ks and Ds must have the same length.")
    
    # Defaults
    if Rs is None:
        Rs = [None] * n
    if Ts is None:
        Ts = [None] * n
    if len(Rs) != n or len(Ts) != n:
        raise ValueError("Rs and Ts (if provided) must match length of Ks.")

    mtxs, dists, rotations, translations, projections = [], [], [], [], []

    for i in range(n):
        # Enforce cam0 reference frame
        if i == 0:
            R = np.eye(3, dtype=float)
            T = np.zeros((3, 1), dtype=float)
        else:
            R = np.eye(3) if Rs[i] is None else np.asarray(Rs[i], dtype=float)
            T_raw = np.zeros(3) if Ts[i] is None else np.asarray(Ts[i], dtype=float)
            T = T_raw.reshape(3, 1)

        K = np.asarray(Ks[i], dtype=float)
        D = np.asarray(Ds[i])

        P = np.concatenate([R, T], axis=1)  # 3x4

        mtxs.append(K)
        dists.append(D)
        rotations.append(R)
        translations.append(T)
        projections.append(P)

    return mtxs, dists, projections, rotations, translations



def load_cam_params(path):
    """
    Loads camera parameters from a given file.
    Args:
        path (str): The path to the file containing the camera parameters.
    Returns:
        tuple: A tuple containing the camera matrix and distortion matrix.
            - camera_matrix (numpy.ndarray): The camera matrix.
            - dist_matrix (numpy.ndarray): The distortion matrix.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return camera_matrix, dist_matrix


def load_cam_to_cam_params(path):
    """
    Loads camera-to-camera calibration parameters from a given file.
    This function reads the rotation matrix (R) and translation vector (T) from a 
    specified file using OpenCV's FileStorage. The file should contain these parameters 
    stored under the keys 'R' and 'T'.
    Args:
        path (str): The file path to the calibration parameters.
    Returns:
        tuple: A tuple containing:
            - R (numpy.ndarray): The rotation matrix.
            - T (numpy.ndarray): The translation vector.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    R = cv_file.getNode('R').mat()
    T = cv_file.getNode('T').mat()

    cv_file.release()
    return R, T

def load_global_cam_params(path, cam_index):
    """
    Loads the global camera transformation parameters for a specified camera
    from a YAML file. This function reads the rotation matrix (R) and translation
    vector (T) stored under the keys 'camera_{cam_index}_R' and 'camera_{cam_index}_T'.
    
    Args:
        path (str): The file path to the YAML file.
        cam_index (int): The camera index to load.
        
    Returns:
        tuple: A tuple containing:
            - R (numpy.ndarray): The rotation matrix.
            - T (numpy.ndarray): The translation vector.
    """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    R = cv_file.getNode(f'camera_{cam_index}_R').mat()
    T = cv_file.getNode(f'camera_{cam_index}_T').mat()
    cv_file.release()
    return R, T


def load_cam_pose(filename):
    """
        Load the rotation matrix and translation vector from a YAML file.
        Args:
            filename (str): The path to the YAML file.
        Returns:
            rotation_matrix (np.ndarray): The 3x3 rotation matrix.
            translation_vector (np.ndarray): The 3x1 translation vector.
    """

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    rotation_matrix = np.array(data['rotation_matrix']['data']).reshape((3, 3))
    translation_vector = np.array(data['translation_vector']['data']).reshape((3, 1))
    
    return rotation_matrix, translation_vector

def load_cam_pose_rpy(filename):
    """
        Load the euler angles and translation vector from a YAML file.
        Args:
            filename (str): The path to the YAML file.
        Returns:
            euler (np.ndarray): The 3x1 euler sequence.
            translation_vector (np.ndarray): The 3x1 translation vector.
    """

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    euler = np.array(data['rotation_rpy']['data']).reshape((3, 1))
    translation_vector = np.array(data['translation_vector']['data']).reshape((3, 1))
    
    return euler, translation_vector


def load_camera_parameters(intrinsics_dir, extrinsics_dir, camera_ids):
    """
    Load intrinsic and extrinsic camera parameters for multiple cameras.
    
    Args:
        config_path (str): Path to the configuration directory
        camera_ids (list, optional): List of camera IDs to load. 
                                   If None, defaults to [0, 2] for 2-camera setup.
                                   For 4-camera setup, use [0, 2, 4, 6].
    
    Returns:
        Camera parameters object from get_camera_params()
    """

    # Load intrinsic parameters for all cameras
    Ks = []
    Ds = []
    for cam_id in camera_ids:
        K, D = load_cam_params(os.path.join(intrinsics_dir, f"camera_{cam_id}_intrinsics.yaml"))
        Ks.append(K)
        Ds.append(D)
    
    # Load extrinsic parameters (transformations from camera 0 to other cameras)
    Rs = [None]  # First camera (reference) has no rotation
    Ts = [None]  # First camera (reference) has no translation

    R_first, T_first, _, _ = load_transformation(os.path.join(extrinsics_dir,f"cam_to_world/camera_{camera_ids[0]}/soder.txt"))
    SE3_first = pin.SE3(R_first,T_first)

    for i, cam_id in enumerate(camera_ids[1:], 1):  # Skip first camera (reference)
        R, T, _, _ = load_transformation(os.path.join(extrinsics_dir,f"cam_to_world/camera_{cam_id}/soder.txt"))
        SE3 = pin.SE3(R,T)
        SE3_rel = SE3_first.inverse()*SE3
        R_rel = SE3_rel.rotation
        T_rel = SE3_rel.translation
        Rs.append(R_rel)
        Ts.append(T_rel)
    
    return get_camera_params(Ks=Ks, Ds=Ds, Rs=Rs, Ts=Ts)


def load_intrinsic_cams(config_path):
    """Load intrinsic and extrinsic camera parameters."""
    K1, D1 = load_cam_params(os.path.join(config_path, "c0_params_color.yaml"))
    K2, D2 = load_cam_params(os.path.join(config_path, "c2_params_color.yaml"))
    K3, D3 = load_cam_params(os.path.join(config_path, "c4_params_color.yaml"))
    K4, D4 = load_cam_params(os.path.join(config_path, "c6_params_color.yaml"))
    return K1,D1,K2,D2,K3,D3,K4, D4

def load_extrinsic_cams(config_path):
    R02, T02 = load_cam_to_cam_params(os.path.join(config_path, "c0_to_c2_params_color.yaml"))
    R24, T24 = load_cam_to_cam_params(os.path.join(config_path, "c2_to_c4_params_color.yaml"))
    R46, T46 = load_cam_to_cam_params(os.path.join(config_path, "c4_to_c6_params_color.yaml"))
    return R02, T02,R24, T24,R46, T46
