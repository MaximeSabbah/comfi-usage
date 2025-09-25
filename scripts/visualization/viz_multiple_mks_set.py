# main_visualization.py
import os
import pandas as pd
import numpy as np
import time
import meshcat
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.viz_utils import add_markers_to_meshcat, set_markers_frame
from utils.utils import read_mks_data
from pinocchio.visualize import MeshcatVisualizer
import imageio
# === Paths & Marker Names ===
subject = "Alessandro"
task = "robot_welding"
csv_1_path = f"./data/{subject}/mocap/{task}/mocap_downsampled_to_40hz.csv"
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
df_1 = pd.read_csv(csv_1_path)
df_2 = pd.read_csv(csv_2_path)

# === Convert to marker dicts using utils ===
mks, start_sample_mks = read_mks_data(df_1, converter =1000.0)
jcp, start_sample_jcp = read_mks_data(df_2, converter =1000.0)
mks_names = list(start_sample_mks.keys())
jcp_names = list(start_sample_jcp.keys())

num_frames = min(len(mks), len(jcp))




# === Initialize Meshcat Visualizer ===
viewer = meshcat.Visualizer()
viz = MeshcatVisualizer()
viz.initViewer(viewer, open=True)
viz.viewer.delete()  # clear if relaunch
native_viz = viz.viewer
native_viz["/Background"].set_property("top_color", list((1,1,1)))
native_viz["/Background"].set_property("bottom_color", list((1,1,1)))
native_viz["/Grid"].set_transform(np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, -0.0],
    [0, 0, 0, 1]
]))



add_markers_to_meshcat(viewer, mks, marker_names=mks_names,
                       radius=0.025, default_color=0xff0000, opacity=0.95)
add_markers_to_meshcat(viewer, jcp, marker_names=jcp_names,
                       radius=0.025, default_color=0x00ff00, opacity=0.95)



# === Animate frame by frame ===
images=[]
for i in range(len(mks)):
         # draw JCP spheres
    set_markers_frame(viewer, mks, i, marker_names=mks_names, unit_scale=1.0)
    set_markers_frame(viewer, jcp, i, marker_names=jcp_names, unit_scale=1.0)
#     images.append(viz.viewer.get_image())

# imageio.mimsave("video.mp4", images, fps=40)
        
    # time.sleep(0.01)  # adjust playback speed
