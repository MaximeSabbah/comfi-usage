from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import numpy as np 
from .linear_algebra_utils import col_vector_3D
from example_robot_data import load
from .linear_algebra_utils import col_vector_3D, transform_to_global_frame,transform_to_local_frame, orthogonalize_matrix

class Robot(RobotWrapper):
    """_Class to load a given urdf_

    Args:
        RobotWrapper (_type_): _description_
    """
    def __init__(self,
                 robot_urdf,
                 package_dirs,
                 isFext=False,
                 freeflyer_ori = None):
        """_Init of the robot class. User can choose between floating base or not and to set the transformation matrix for this floating base._

        Args:
            robot_urdf (_str_): _path to the robot urdf_
            package_dirs (_str_): _path to the meshes_
            isFext (bool, optional): _Adds a floating base if set to True_. Defaults to False.
            freeflyer_ori (_array_, optional): _Orientation of the floating base, given as a rotation matrix_. Defaults to None.
        """

        # intrinsic dynamic parameter names
        self.params_name = (
            "Ixx",
            "Ixy",
            "Ixz",
            "Iyy",
            "Iyz",
            "Izz",
            "mx",
            "my",
            "mz",
            "m",
        )

        # defining conditions
        self.isFext = isFext

        # folder location
        self.robot_urdf = robot_urdf

        # initializing robot's models
        if not isFext:
            self.initFromURDF(robot_urdf, package_dirs=package_dirs)
        else:
            self.initFromURDF(robot_urdf, package_dirs=package_dirs,
                              root_joint=pin.JointModelFreeFlyer())
            
        if freeflyer_ori is not None and isFext == True : 
            self.model.jointPlacements[self.model.getJointId('root_joint')].rotation = freeflyer_ori
            ub = self.model.upperPositionLimit
            ub[:7] = 1
            self.model.upperPositionLimit = ub
            lb = self.model.lowerPositionLimit
            lb[:7] = -1
            self.model.lowerPositionLimit = lb
            self.data = self.model.createData()
        # else:
        #     self.model.upperPositionLimit = np.full(43, np.pi)
        #     self.model.lowerPositionLimit = np.full(43, -np.pi)
            
        ## \todo test that this is equivalent to reloading the model
        self.geom_model = self.collision_model

def build_human_model(urdf_path: str, urdf_meshes_path: str):
    robot = Robot(urdf_path, urdf_meshes_path, isFext=True)
    return robot.model, robot.collision_model, robot.visual_model, robot.data

def lock_joints(model: pin.Model,
                collision_model: pin.GeometryModel,
                visual_model: pin.GeometryModel,
                joints_to_lock):
    q0 = pin.neutral(model)
    joint_ids = [model.getJointId(jn) for jn in joints_to_lock if model.existJointName(jn)]
    model_r, (coll_r, vis_r) = pin.buildReducedModel(model, [collision_model, visual_model], joint_ids, q0)
    data_r = pin.Data(model_r)
    return model_r, coll_r, vis_r, data_r

def load_robot_panda():
    robot = load("panda")
    return robot.model, robot.collision_model, robot.visual_model, robot.data

#get_virtual_pelvis_pose, used to get thigh pose
def get_virtual_pelvis_pose(mks_positions):
    """
    Calculate the pelvis pose matrix from motion capture marker positions.
    The function computes the pelvis pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the pelvis coordinate system. Finally, it constructs the 
    pose matrix and ensures it is orthogonal.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of the motion capture markers.
                                The keys can be either 'r.PSIS_study', 'L.PSIS_study', 'r.ASIS_study', 
                                'L.ASIS_study', or 'RIPS', 'LIPS', 'RIAS', 'LIAS'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the pelvis pose.
    """

    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    center_PSIS = []
    center_ASIS = []

    center_PSIS = (mks_positions['r.PSIS_study'] + mks_positions['L.PSIS_study']).reshape(3,1)/2.0
    center_ASIS = (mks_positions['r.ASIS_study'] + mks_positions['L.ASIS_study']).reshape(3,1)/2.0
    center = (mks_positions['r.ASIS_study'] +
                mks_positions['L.ASIS_study'] +
                mks_positions['r.PSIS_study'] +
                mks_positions['L.PSIS_study'] )/4.0
    X = center_ASIS - center_PSIS
    X = X/np.linalg.norm(X)
    Z = mks_positions['r.ASIS_study'] - mks_positions['L.ASIS_study']
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = center_ASIS.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

def get_pelvis_pose(mks_positions, gender = 'male'):
    """
    Calculate the pelvis pose matrix from motion capture marker positions.
    The function computes the pelvis pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the pelvis coordinate system. Finally, it constructs the 
    pose matrix and ensures it is orthogonal.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of the motion capture markers.
                                The keys can be either 'r.PSIS_study', 'L.PSIS_study', 'r.ASIS_study', 
                                'L.ASIS_study', or 'RIPS', 'LIPS', 'RIAS', 'LIAS'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the pelvis pose.
    """

    if gender == 'male':
        ratio_x = 0.335
        ratio_y = -0.032
        ratio_z = 0.0
    else : 
        ratio_x = 0.34
        ratio_y = 0.049
        ratio_z = 0.0

    pose = np.eye(4,4)
    center_PSIS = []
    center_ASIS = []
    center_right_ASIS_PSIS = []
    center_left_ASIS_PSIS = []
    LJC=np.zeros((3,1))

    dist_rPL_lPL = np.linalg.norm(mks_positions["r.ASIS_study"]-mks_positions["L.ASIS_study"])
    virtual_pelvis_pose = get_virtual_pelvis_pose(mks_positions)
    LJC = virtual_pelvis_pose[:3, 3].reshape(3,1)


    center_PSIS = (mks_positions['r.PSIS_study'] + mks_positions['L.PSIS_study']).reshape(3,1)/2.0
    center_ASIS = (mks_positions['r.ASIS_study'] + mks_positions['L.ASIS_study']).reshape(3,1)/2.0
    
    center_right_ASIS_PSIS = (mks_positions['r.PSIS_study'] + mks_positions['r.ASIS_study']).reshape(3,1)/2.0
    center_left_ASIS_PSIS = (mks_positions['L.PSIS_study'] + mks_positions['L.ASIS_study']).reshape(3,1)/2.0
    
    offset_local = col_vector_3D(
                                -ratio_x * dist_rPL_lPL,
                                +ratio_y * dist_rPL_lPL,
                                ratio_z * dist_rPL_lPL
                                )
    LJC = LJC + virtual_pelvis_pose[:3, :3] @ offset_local
 
    X = center_ASIS - center_PSIS
    X = X/np.linalg.norm(X)
    # Z = mks_positions['r.ASIS_study'] - mks_positions['L.ASIS_study']
    Z = center_right_ASIS_PSIS - center_left_ASIS_PSIS
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)


    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ((center_right_ASIS_PSIS + center_left_ASIS_PSIS)/2.0).reshape(3,)
    # pose[:3,3] = LJC.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose
#construct torso frame and get its pose from a dictionnary of mks positions and names
def get_torso_pose(mks_positions):
    """
    Calculate the torso pose matrix from motion capture marker positions.
    The function computes a 4x4 transformation matrix representing the pose of the torso.
    The matrix includes rotation and translation components derived from the positions
    of specific markers.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers.
                                Expected keys are 'Neck', 'midHip', 'C7_study', 'CV7', 'SJN', 
                                'HeadR', 'HeadL', 'RSAT', and 'LSAT'. Each key should map to a 
                                numpy array of shape (3,).
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the torso pose.
    """

    pose = np.eye(4,4)
    X, Y, Z, trunk_center = [], [], [], []

    trunk_center = (mks_positions['r_shoulder_study'] + mks_positions['L_shoulder_study'])/2.0 
    midhip = (mks_positions['r.ASIS_study'] +
                mks_positions['L.ASIS_study'] +
                mks_positions['r.PSIS_study'] +
                mks_positions['L.PSIS_study'] )/4.0

    Y = (trunk_center - midhip).reshape(3,1)
    Y = Y/np.linalg.norm(Y)
    X = (trunk_center - mks_positions['C7_study']).reshape(3,1)
    X = X/np.linalg.norm(X)
   
    Z = np.cross(X, Y, axis=0)
    X = np.cross(Y, Z, axis=0)


    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = trunk_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose


def midpoint(p1, p2):
    return 0.5 * (np.array(p1) + np.array(p2))

def compute_hip_joint_center(L_ASIS, R_ASIS, L_PSIS, R_PSIS, knee_study, ankle_study, side="right"):
    """
    Compute hip joint center using Leardini et al. (1999) method.
    
    """
    ASIS_mid = midpoint(R_ASIS, L_ASIS)
    PSIS_mid = midpoint(R_PSIS, L_PSIS)

    # Distance between ASIS and PSIS centers
    pelvis_depth_vec = ASIS_mid - PSIS_mid
    pelvis_depth = np.linalg.norm(pelvis_depth_vec)

    # Distance between ASIS markers (pelvis width)
    pelvis_width = np.linalg.norm(R_ASIS - L_ASIS)

    ankle_knee_length = np.linalg.norm(ankle_study - knee_study)
    knee_ASIS_length = np.linalg.norm(knee_study - (R_ASIS if side == "right" else L_ASIS))
    vertical_adjust = ankle_knee_length + knee_ASIS_length

    hip_y = ASIS_mid[1] - 0.096 * vertical_adjust
    # Compute hip center
    hip_x = ASIS_mid[0] - 0.31 * pelvis_depth
    if side == "right":        
        hip_z = ASIS_mid[2] + 0.38 * pelvis_width
    elif side == "left":
        hip_z = ASIS_mid[2] - 0.38 * pelvis_width
    else:
        raise ValueError("Side must be 'right' or 'left'")

    return np.array([hip_x, hip_y, hip_z])

def compute_uptrunk(C7, CLAV):
    vec = CLAV - C7
    norm = np.linalg.norm(vec)
    angle_rad = 8 * np.pi / 180
    return np.array([
        C7[0] + np.cos(angle_rad) * 0.55 * norm,
        C7[1] + np.sin(angle_rad) * 0.55 * norm,
        C7[2]
    ])

def compute_shoulder(SHO, C7, CLAV, side='right'):
    vec = CLAV - C7
    norm = np.linalg.norm(vec)
    angle_rad = 11 * np.pi / 180
    sign = -1 if side == 'right' else -1  # both use minus sign in paper

    return np.array([
        SHO[0] + np.cos(angle_rad) * 0.43 * norm,
        SHO[1] + sign * np.sin(angle_rad) * 0.43 * norm,
        SHO[2]
    ])


def compute_joint_centers_from_mks(markers, *, gender="male"):
    """
    Compute joint center positions and segment lengths from marker positions.

    Parameters
    ----------
    markers : dict[str, np.ndarray]
        Dict of global marker positions. Each value should be shape (3,) or (3,1),
        in either millimeters ("mm") or meters ("m") depending on `units`.
    units : {"mm", "m"}, optional
        Input units for `markers`. Used only for reporting lengths (meters).

    Returns
    -------
    jcp_global : dict[str, np.ndarray]
        Joint centers in GLOBAL frame, each as 1D array shape (3,) in input units.
    segment_lengths : dict[str, float]
        Upper/lower arm segment lengths in meters.
    norms : dict[str, list[float]]
        Elbow inter-epicondyle distances in meters. (Lists so you can append per-frame upstream.)
    """
    # --- helpers ---
    def as_col(x):
        x = np.asarray(x)
        return x.reshape(3, 1) if x.shape != (3, 1) else x


    jcp = {}
    jcp_g = {}

    # Pelvis pose (global)
    pelvis_pose = get_virtual_pelvis_pose(markers)
    pelvis_position = as_col(pelvis_pose[:3, 3])
    pelvis_rotation = pelvis_pose[:3, :3]

    bi_acromial_dist = np.linalg.norm(markers['L_shoulder_study'] - markers['r_shoulder_study'])
    torso_pose = get_torso_pose(markers)

    # ---- Transform all markers into pelvis (local) frame (do NOT mutate input) ----
    markers_local = {}
    for name, coords in markers.items():
        coords_col = as_col(coords)
        markers_local[name] = transform_to_local_frame(coords_col, pelvis_position, pelvis_rotation)

    # ---- Shoulders & Neck ----
    try:
        jcp_g["RShoulder"]= markers['r_shoulder_study'].reshape(3,1) + (torso_pose[:3, :3].reshape(3,3)) @ col_vector_3D(0.0, -0.17*bi_acromial_dist, 0.0)
        jcp_g["LShoulder"] = markers['L_shoulder_study'].reshape(3,1) + (torso_pose[:3, :3].reshape(3,3)) @ col_vector_3D(0.0, -0.17*bi_acromial_dist, 0.0)

        jcp["RShoulder"] = transform_to_local_frame(jcp_g["RShoulder"], pelvis_position, pelvis_rotation)
        jcp["LShoulder"] = transform_to_local_frame(jcp_g["LShoulder"], pelvis_position, pelvis_rotation)
        jcp["Neck"] = compute_uptrunk(markers_local["C7_study"], markers_local["SJN"])
    except KeyError as e:
        pass

    # ---- Elbows ----
    try:
        jcp["RElbow"] = midpoint(markers_local["r_melbow_study"], markers_local["r_lelbow_study"])
        jcp["LElbow"] = midpoint(markers_local["L_melbow_study"], markers_local["L_lelbow_study"])

    except KeyError:
        pass

    # ---- Wrists ----
    try:
        jcp["RWrist"] = midpoint(markers_local["r_mwrist_study"], markers_local["r_lwrist_study"])
        jcp["LWrist"] = midpoint(markers_local["L_mwrist_study"], markers_local["L_lwrist_study"])
    except KeyError:
        pass

    # ---- Pelvis & Hips ----
    try:
        R_ASIS = markers_local["r.ASIS_study"]
        L_ASIS = markers_local["L.ASIS_study"]
        R_PSIS = markers_local["r.PSIS_study"]
        L_PSIS = markers_local["L.PSIS_study"]

        jcp["RHip"] = compute_hip_joint_center(L_ASIS, R_ASIS, L_PSIS, R_PSIS,
                                               markers_local["r_knee_study"],
                                               markers_local["r_ankle_study"],
                                               side="right")
        jcp["LHip"] = compute_hip_joint_center(L_ASIS, R_ASIS, L_PSIS, R_PSIS,
                                               markers_local["L_knee_study"],
                                               markers_local["L_ankle_study"],
                                               side="left")
        jcp["midHip"] = midpoint(jcp["RHip"], jcp["LHip"])
    except KeyError:
        pass

    # ---- Knees ----
    try:
        jcp["RKnee"] = midpoint(markers_local["r_mknee_study"], markers_local["r_knee_study"])
        jcp["LKnee"] = midpoint(markers_local["L_mknee_study"], markers_local["L_knee_study"])
    except KeyError:
        pass

    # ---- Ankles ----
    try:
        jcp["RAnkle"] = midpoint(markers_local["r_mankle_study"], markers_local["r_ankle_study"])
        jcp["LAnkle"] = midpoint(markers_local["L_mankle_study"], markers_local["L_ankle_study"])
    except KeyError:
        pass

    # ---- Feet / Toes ----
    try:
        jcp["RHeel"] = markers_local["r_calc_study"]
        jcp["LHeel"] = markers_local["L_calc_study"]
    except KeyError:
        pass

    try:
        jcp["RBigToe"] = markers_local["r_toe_study"]
        jcp["LBigToe"] = markers_local["L_toe_study"]
    except KeyError:
        pass

    try:
        jcp["RSmallToe"] = markers_local["r_5meta_study"]
        jcp["LSmallToe"] = markers_local["L_5meta_study"]
    except KeyError:
        pass

    # ---- Back to GLOBAL frame ----
    jcp_global = {}
    for name, coords in jcp.items():
        coords_col = as_col(coords)
        # Guard against accidental matrices (e.g., someone returns a 3x3)
        if coords_col.shape != (3,1):
            # try to coerce; if it fails, skip
            try:
                coords_col = np.asarray(coords).reshape(3,1)
            except Exception:
                print(f"⚠️ Skipping '{name}' – unexpected shape {np.asarray(coords).shape}")
                continue
        global_coords = transform_to_global_frame(coords_col, pelvis_position, pelvis_rotation)
        jcp_global[name] = global_coords.flatten()


    return jcp_global