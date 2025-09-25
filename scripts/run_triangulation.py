#Triangulate from 2 csv files of 2dkeypoints
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from utils.cam_utils import load_camera_parameters
from utils.triangulation_utils import triangulate_points_adaptive
from utils.utils import read_mmpose_file, save_to_csv,load_transformation,transform_keypoints_list_cam0_to_mocap,read_mmpose_scores
from utils.linear_algebra_utils import butterworth_filter


tasks = ["robot_welding"]
subjects = ["Alessandro"]


num_keypoints=26 
markers = [
        "Nose", "LEye", "REye", "LEar", "REar", 
        "LShoulder", "RShoulder", "LElbow", "RElbow", 
        "LWrist", "RWrist", "LHip", "RHip", 
        "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
        "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
    ]
header = []
for marker in markers:
    header.extend([f"{marker}_x", f"{marker}_y", f"{marker}_z"])

def main(subject,task):
    transformation_file = f"./data/{subject}/config/soder_0.txt"
    R_trans, d_trans, _, _ = load_transformation(transformation_file)

    config_path = f"./data/{subject}/config/calib"
    output_path = f"./data/{subject}/res_hpe"
    os.makedirs(output_path, exist_ok=True)

    output_csv_path = f"{output_path}/{task}/3d_keypoints_4cams.csv"

    file_paths = [
        f"./data/{subject}/res_hpe/{task}/keypoints_cam0.csv",
        f"./data/{subject}/res_hpe/{task}/keypoints_cam2.csv",
        f"./data/{subject}/res_hpe/{task}/keypoints_cam4.csv",
        f"./data/{subject}/res_hpe/{task}/keypoints_cam6.csv"
    ]
   
    camera_data = [read_mmpose_file(file) for file in file_paths]
    uvs = [
        np.array([[line[2 * i], line[2 * i + 1]] for line in data for i in range(num_keypoints)])
        .reshape(-1, num_keypoints, 2)
        for data in camera_data
    ]

    if len(file_paths) == 2:
        camera_ids=[0, 2]
        threshold = 0.0
    else: 
        camera_ids=[0, 2, 4, 6]
        threshold = 0.5
        

    mtxs, dists, projections, rotations, translations = load_camera_parameters(config_path, camera_ids=camera_ids)
    scores = read_mmpose_scores(file_paths)
    
    keypoints_in_cam0_list = triangulate_points_adaptive(uvs, mtxs, dists, projections, scores, threshold)

    keypoints_in_mocap = transform_keypoints_list_cam0_to_mocap(
        keypoints_in_cam0_list,
        R_trans,
        d_trans
    )
    filtered_data = butterworth_filter(
    data=keypoints_in_mocap,
    cutoff_frequency=10.0,  
    order=5,
    sampling_frequency=40
    )
    save_to_csv(filtered_data, output_csv_path, header=header)

if __name__ == "__main__":
    for subject in subjects:    
        for task in tasks:
            main(subject,task)