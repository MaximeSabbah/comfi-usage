import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import pandas as pd
import numpy as np
import time
import meshcat
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import read_mks_data
from utils.utils import read_mks_data
from utils.viz_utils import add_markers_to_meshcat,set_markers_frame
import imageio
from pinocchio.visualize import MeshcatVisualizer


# === Load data ===
subject = "Alessandro"
task = "robot_welding"
path_to_csv = f"./data/{subject}/mocap/{task}/mocap_downsampled_to_40hz.csv"
df = pd.read_csv(path_to_csv)


mks_dict, start_sample_dict = read_mks_data(df, start_sample=0, converter=1000.0)

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

mks_names = list(start_sample_dict.keys())

add_markers_to_meshcat(viewer, mks_dict, marker_names=mks_names,
                       radius=0.025, default_color=0xff0000, opacity=0.95)
# 0xff0000  # red
# 0x00ff00  # green
# 0x0000ff  # blue
# 0xffff00  # yellow
images=[]
for i in range(len(mks_dict)):
         # draw JCP spheres
    set_markers_frame(viewer, mks_dict, i, marker_names=mks_names, unit_scale=1.0)
#     images.append(viz.viewer.get_image())

# imageio.mimsave("video.mp4", images, fps=40)

