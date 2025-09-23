import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import yaml
import pinocchio as pin

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
    # mks mocap + names 
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


def read_mmpose_file(nom_fichier):
    donnees = []
    with open(nom_fichier, 'r') as f:
        for ligne in f:
            ligne = ligne.strip().split(',')  
            donnees.append([float(valeur) for valeur in ligne[1:]])  
        # print('donnees=',donnees)
    return donnees

def read_mmpose_scores(liste_fichiers):
    all_scores= []
    for f in liste_fichiers :
        data= np.loadtxt(f, delimiter=',')
        all_scores.append(data[:, 0])
    return np.array(all_scores).transpose().tolist()


def save_to_csv(data, output_path, header=None):
    """Save 3D keypoints to a CSV file with optional header."""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, header=header if header is not None else False)
    print(f"Saved {len(data)} frames to {output_path}")

def transform_keypoints_list_cam0_to_mocap(keypoints_list, R_trans, d_trans):
    """Apply transformation to each frame of flattened 3D keypoints."""
    transformed_list = []

    for flat_coords in keypoints_list:
        # Convert to shape (N, 3)
        p3d_cam0 = np.array(flat_coords).reshape(-1, 3)  # (N, 3)
        # Apply transformation
        p3d_mocap = (R_trans @ p3d_cam0.T).T + d_trans  # (N, 3)
        # Flatten again
        transformed_list.append(p3d_mocap.flatten().tolist())

    return transformed_list

def load_force_data(csv_file_path):
    
    df = pd.read_csv(csv_file_path)
    
    # Organiser les données par capteur
    force_data = {}
    
    # Pour chaque capteur Sensix (1, 2, 3)
    for sensor_id in [1, 2, 3]:
        sensor_name = f"Sensix_{sensor_id}"
        
        if f"{sensor_name}_Fx" in df.columns:
            force_data[sensor_id] = {
                'frames': df['camera_frame'].values,
                'Fx': df[f"{sensor_name}_Fx"].values,
                'Fy': df[f"{sensor_name}_Fy"].values, 
                'Fz': df[f"{sensor_name}_Fz"].values,
                'Mx': df[f"{sensor_name}_Mx"].values,
                'My': df[f"{sensor_name}_My"].values,
                'Mz': df[f"{sensor_name}_Mz"].values,
            }
    
    return force_data

