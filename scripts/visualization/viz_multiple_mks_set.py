# main_visualization.py
import os
import pandas as pd
import numpy as np
import time
import meshcat
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.viz_utils import add_sphere, place
from utils.utils import read_mks_data, udp_csv_to_dataframe

# === Paths & Marker Names ===
subject = "Mohamed"
task = "hitting_sat"
csv_1_path = f"./data/{subject}/mocap/{task}/mks_data_gapfilled.csv"
csv_2_path = f"./data/{subject}/mocap/{task}/joint_center_positions.csv"

anatomical_mks = ['r.PSIS_study','L.PSIS_study','r.ASIS_study','L.ASIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis']

# === Load CSVs as pandas DataFrames ===
df_1 = udp_csv_to_dataframe(csv_1_path, anatomical_mks) #float
df_2 = pd.read_csv(csv_2_path)

# === Convert to marker dicts using utils ===
mks_list_1, _ = read_mks_data(df_1, converter =1.0)
mks_list_2, start_sample_dict = read_mks_data(df_2, converter =1.0)
num_frames = min(len(mks_list_1), len(mks_list_2))
jcp_mocap = start_sample_dict.keys()
# === Initialize Meshcat visualizer ===
viz = meshcat.Visualizer().open()
viz["/Background"].set_property("top_color", [1.0, 1.0, 1.0])
viz["/Background"].set_property("bottom_color", [1.0, 1.0, 1.0])

# Optionnel : déplacer la grille
viz["/Grid"].set_transform(np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, -0.0],
    [0, 0, 0, 1]
]))

# Add spheres for both marker sets
for name in anatomical_mks:
    add_sphere(viz, f"world/{name}", radius=0.02, color=0xff0000)  # red

for name in jcp_mocap:
    add_sphere(viz, f"world/{name}", radius=0.02, color=0x0000ff)   #blue

# === Animate frame by frame ===
for i in range(num_frames):
    for name in anatomical_mks:
        pos = mks_list_1[i][name].reshape(3,)
        # print(pos)
        place(viz, name, pos)
        
    for name in jcp_mocap:
        pos = mks_list_2[i][name].reshape(3,)
        # print(pos)
        place(viz, name, pos)
        
    # time.sleep(0.01)  # adjust playback speed
