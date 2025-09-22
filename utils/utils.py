import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from .linear_algebra_utils import col_vector_3D, transform_to_global_frame,transform_to_local_frame
from .model_utils import get_torso_pose, get_virtual_pelvis_pose, get_pelvis_pose
import yaml

def to_utc(s: pd.Series) -> pd.Series:
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")

def load_transformation(file_path):
    """
    Loads the transformation parameters (R, d, s, rms) from a text file.

    Parameters:
    file_path: str
        Path to the file from which the transformation parameters will be read.

    Returns:
    R: ndarray
        Rotation matrix (3x3)
    d: ndarray
        Translation vector (3,)
    s: float
        Scale factor
    rms: float
        Root mean square fit error
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        R_start = lines.index("Rotation Matrix (R):\n") + 1
        R = np.loadtxt(lines[R_start:R_start + 3])
        d_start = lines.index("Translation Vector (d):\n") + 1
        d = np.loadtxt(lines[d_start:d_start + 1]).flatten()
        s_line = next(line for line in lines if line.startswith("Scale Factor (s):"))
        s = float(s_line.split(":")[1].strip())
        rms_line = next(line for line in lines if line.startswith("RMS Error:"))
        rms = float(rms_line.split(":")[1].strip())
    return R, d, s, rms

def udp_csv_to_dataframe(csv_path, marker_names):
    """
    Preprocess a UDP CSV file into a DataFrame suitable for read_mks_data.

    Parameters:
        csv_path (str): Path to the CSV file.
        marker_names (list): List of marker base names (without _x/_y/_z).

    Returns:
        pd.DataFrame: A DataFrame with columns formatted as marker_x, marker_y, marker_z.
    """
    # 1. Open manually
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # 2. Skip the header
    lines = lines[:]

    # 3. Prepare all rows
    all_rows = []
    for line in lines:
        # Remove newline, then split
        line = line.strip()
        if not line:
            continue  # skip empty lines
        parts = line.split(",")
        timestamp = parts[0]
        udp_values = [float(val) for val in parts[2:]]
        all_rows.append(udp_values)

    # 4. Now create a dataframe
    udp_df = pd.DataFrame(all_rows)

    # 5. Build column names
    new_columns = []
    for marker in marker_names:
        new_columns.extend([f"{marker}_x", f"{marker}_y", f"{marker}_z"])

    if udp_df.shape[1] != len(new_columns):
        raise ValueError(f"Mismatch between expected markers ({len(new_columns)}) and data columns ({udp_df.shape[1]}). Check marker list!")

    udp_df.columns = new_columns

    return udp_df

def read_mks_data(data_markers, start_sample=0, converter = 1.0):
    #the mks are ordered in a csv like this : "time,r.ASIS_study_x,r.ASIS_study_y,r.ASIS_study_z...."
    """    
    Parameters:
        data_markers (pd.DataFrame): The input DataFrame containing marker data.
        start_sample (int): The index of the sample to start processing from.
        time_column (str): The name of the time column in the DataFrame.
        
    Returns:
        list: A list of dictionaries where each dictionary contains markers with 3D coordinates.
        dict: A dictionary representing the markers and their 3D coordinates for the specified start_sample.
    """
    # Extract marker column names
    marker_columns = [col[:-2] for col in data_markers.columns if col.endswith("_x")]
    
    # Initialize the result list
    result_markers = []
    
    # Iterate over each row in the DataFrame
    for _, row in data_markers.iterrows():
        frame_dict = {}
        for marker in marker_columns:
            x = row[f"{marker}_x"] / converter  #convert to m
            y = row[f"{marker}_y"]/ converter
            z = row[f"{marker}_z"]/ converter
            frame_dict[marker] = np.array([x, y, z])  # Store as a NumPy array
        result_markers.append(frame_dict)
    
    # Get the data for the specified start_sample
    start_sample_mks = result_markers[start_sample]
    
    return result_markers, start_sample_mks

def try_read_mks(data_or_path, **kwargs):
    """
    Funnel ALL reads through read_mks_data.
    - If given a path, load CSV -> DataFrame, then pass to read_mks_data.
    - If read_mks_data doesn't apply (e.g., it's a plain table), return the DataFrame.
    """
    if isinstance(data_or_path, (str, os.PathLike)):
        df = pd.read_csv(data_or_path, **{k:v for k,v in kwargs.items() if k in {"parse_dates"}})
    else:
        df = data_or_path

    try:
        # Try using your canonical reader first
        return read_mks_data(df, **{k:v for k,v in kwargs.items() if k != "parse_dates"})
    except Exception:
        # Fall back to raw DF if this CSV isn't an MKS blob
        return df
    
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


def load_cameras_from_soder(soder_paths):
    cams = {}
    for key, path in soder_paths.items():
        R, d, _, _ = load_transformation(path)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = d.reshape(3)
        cams[key] = T
    return cams

def load_robot_base_pose(yaml_path: str) -> np.ndarray:
    with open(yaml_path) as f:
        Y = yaml.safe_load(f)["world_T_robot"]
    R = np.array(Y["rotation_matrix"], dtype=float)
    t = np.array(Y["translation"], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_all_data(paths, start_sample: int = 0, converter: float = 1000.0):
    # mks mocap + names via your canonical reader
    mks_raw = pd.read_csv(paths.mks_csv)  # still funnel via try_read_mks next line
    mks_dict, start_sample_dict = try_read_mks(mks_raw, start_sample=start_sample, converter=converter)
    mks_names = list(start_sample_dict.keys())

    # Reference human joint 
    q_ref_df = try_read_mks(paths.q_ref_csv)
    q_ref = q_ref_df if isinstance(q_ref_df, np.ndarray) else pd.read_csv(paths.q_ref_csv).to_numpy(dtype=float)

    # Robot CSV 
    robot_df = try_read_mks(paths.robot_csv, parse_dates=["_cam_time", "timestamp"])
    if not isinstance(robot_df, pd.DataFrame):
        robot_df = pd.read_csv(paths.robot_csv, parse_dates=["_cam_time", "timestamp"])
    pos_cols = [f"position.panda_joint{i}" for i in range(1, 8)]
    q_robot = robot_df[pos_cols].to_numpy(dtype=float)
    q_robot = np.hstack([q_robot, np.zeros((q_robot.shape[0], 2), dtype=q_robot.dtype)])

    # Camera timestamps + robot timestamps (time sync)
    t_cam = try_read_mks(paths.cam0_ts_csv, parse_dates=["timestamp"])
    if not isinstance(t_cam, pd.DataFrame):
        t_cam = pd.read_csv(paths.cam0_ts_csv, parse_dates=["timestamp"])
    t_robot = try_read_mks(paths.robot_csv, parse_dates=["timestamp"])
    if not isinstance(t_robot, pd.DataFrame):
        t_robot = pd.read_csv(paths.robot_csv, parse_dates=["timestamp"])

    # Joint Center Positions (JCP) from mocap
    jcp_df = try_read_mks(paths.jcp_csv)
    if not isinstance(jcp_df, pd.DataFrame):
        jcp_df = pd.read_csv(paths.jcp_csv)
    bases = []
    for c in jcp_df.columns:
        if "_" in c:
            b, ax = c.rsplit("_", 1)
            if ax.lower() in ("x", "y", "z") and b not in bases:
                bases.append(b)
    K = len(bases)
    N = len(jcp_df)
    jcp = np.empty((N, K, 3), dtype=float)
    for k, b in enumerate(bases):
        jcp[:, k, 0] = jcp_df[f"{b}_x"].to_numpy(dtype=float) / 1000.0
        jcp[:, k, 1] = jcp_df[f"{b}_y"].to_numpy(dtype=float) / 1000.0
        jcp[:, k, 2] = jcp_df[f"{b}_z"].to_numpy(dtype=float) / 1000.0

    return {
        "mks_dict": mks_dict,
        "mks_names": mks_names,
        "q_ref": q_ref,
        "robot_df": robot_df,
        "q_robot": q_robot,
        "t_cam": t_cam,
        "t_robot": t_robot,
        "jcp": jcp,
        "jcp_bases": bases
    }

def compute_time_sync(t_cam: pd.DataFrame, t_robot: pd.DataFrame, tol_ms: int = 5):
    t_cam = t_cam.copy()
    t_robot = t_robot.copy()
    t_cam["timestamp"] = to_utc(t_cam["timestamp"])
    t_robot["timestamp"] = to_utc(t_robot["timestamp"])
    t_cam = t_cam.reset_index().rename(columns={"index": "cam_idx"})
    t_robot = t_robot.reset_index().rename(columns={"index": "robot_idx"})

    exact = t_cam.merge(t_robot, on="timestamp", how="inner")
    if not exact.empty:
        first = exact.sort_values("timestamp").iloc[0]
        return {"cam_idx": int(first["cam_idx"]), "robot_idx": int(first["robot_idx"])}

    tol = pd.Timedelta(f"{tol_ms}ms")
    nearest = pd.merge_asof(
        t_cam.sort_values("timestamp"),
        t_robot.sort_values("timestamp"),
        on="timestamp", direction="nearest", tolerance=tol,
        suffixes=("_cam", "_robot")
    ).dropna(subset=["robot_idx"])
    if not nearest.empty:
        first = nearest.iloc[0]
        return {"cam_idx": int(first["cam_idx"]), "robot_idx": int(first["robot_idx"])}
    return None
