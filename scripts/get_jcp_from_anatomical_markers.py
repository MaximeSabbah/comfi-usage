import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from  utils.utils  import read_mks_data
import pandas as pd
from utils.urdf_utils import compute_joint_centers_from_mks

subjects = [
    "Alessandro"
]
tasks = ["robot_welding"]
gender = 'male'

mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis']

jcp_names = [
        "RShoulder", "LShoulder", "Neck",  "RElbow", "LElbow", 
        "RWrist", "LWrist", "RHip", "LHip", "midHip",

        "RKnee", "LKnee", "RAnkle", "LAnkle","RHeel", "LHeel",
         "RBigToe", "LBigToe", "RSmallToe", "LSmallToe"
    ]



for subject in subjects:
    for task in tasks:
        base_path = f"./data/{subject}/mocap/"
        path_to_csv =f"{base_path}/{task}/mocap_downsampled_to_40hz.csv"
        print(path_to_csv)
         # Skip if file doesn't exist
        if not os.path.exists(path_to_csv):
            print(f"Skipping missing task: {subject} / {task}")
            continue
        
        df = pd.read_csv(path_to_csv)
        # df = udp_csv_to_dataframe(path_to_csv, mks_names)
        df.columns = [col.replace(f"{subject}:", "") for col in df.columns]
        frames = df["Frame"] if "Frame" in df.columns else range(len(df))
        mks_names = sorted(set(col.rsplit("_", 1)[0] for col in df.columns if "_x" in col))

        mks_dict, start_sample_dict = read_mks_data(df, start_sample=0,converter = 1000.0) #check data unit, if it is in m converter=1.0, if it is in mm converter=1000.0

        jcp_per_frame = []
        for frame_id in range(len(mks_dict)):
            markers_frame = mks_dict[frame_id]
            jcp = compute_joint_centers_from_mks(markers_frame)
            jcp_per_frame.append(jcp)

        jcp_rows = []
        for jcp in jcp_per_frame:
            flat_jcp = {}
            for name, coords in jcp.items():
                flat_jcp[f"{name}_x"] = coords[0]
                flat_jcp[f"{name}_y"] = coords[1]
                flat_jcp[f"{name}_z"] = coords[2]
            jcp_rows.append(flat_jcp)

        jcp_df = pd.DataFrame(jcp_rows)
        path = f"{base_path}/{task}"
        os.makedirs(path, exist_ok=True)

        output_csv_path = f"{path}/joint_center_positions_test.csv"

        jcp_df.to_csv(output_csv_path, index=False)
