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
from utils.viz_utils import add_sphere, place

# === Load data ===
subject = "Alessandro"
task = "robot_welding"
path_to_csv = f"./data/{subject}/mocap/{task}/mocap_downsampled_to_40hz.csv"
df = pd.read_csv(path_to_csv)


mks_dict, start_sample_dict = read_mks_data(df, start_sample=0, converter=1000.0)

mks_names = start_sample_dict.keys()
# === Initialize Meshcat Visualizer ===
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

# Create spheres for each marker

# 0xff0000  # red
# 0x00ff00  # green
# 0x0000ff  # blue
# 0xffff00  # yellow

for name in mks_names:
    add_sphere(viz, f"world/{name}", radius=0.02, color= 0xff0000)


# === Animate frame by frame ===
for i, frame in enumerate(mks_dict):
    for name in mks_names:
        pos = frame[name].reshape(3,)
        # print(pos)
        place(viz, name, pos)

    # Uncomment for step-by-step with Enter
    # input(f"Frame {i+1}/{len(mks_dict)} - Press Enter")

    # Or add a small delay for smooth animation
    time.sleep(0.01)

