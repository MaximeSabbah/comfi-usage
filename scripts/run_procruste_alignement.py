import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from  utils.utils  import read_mks_data
import pandas as pd
from utils.utils import kabsch_global, plot_aligned_markers,compute_mpjpe

def get_marker_array(mks_list, marker_name):
    """
    mks_list: list of dicts with keys 'name' and 'value'
    marker_name: str
    returns: array (T,3)
    """
    for mk in mks_list:
        if mk["name"] == marker_name:
            return mk["value"]
    raise ValueError(f"Marker {marker_name} not found")


# === Load data ===
subject = "Alessandro"
task = "robot_welding"
path_to_jcp_mocap= f"./data/{subject}/mocap/{task}/joint_center_positions_test.csv" #data from mocap and hpe should be with same unit
df_mocap = pd.read_csv(path_to_jcp_mocap)
mks_mocap, start_sample_mocap = read_mks_data(df_mocap, start_sample=0)


path_to_jcp_hpe= f"./data/{subject}/res_hpe/{task}/3d_keypoints.csv"
df_hpe = pd.read_csv(path_to_jcp_hpe)
mks_hpe, start_sample_hpe = read_mks_data(df_hpe, start_sample=0)
# print(mks_mocap)
print(mks_mocap[0])

#take only common markers
common_mks = sorted(set(start_sample_mocap.keys()) & set(start_sample_mocap.keys()))
print("Common markers:", common_mks)

T = len(mks_mocap)
N = len(common_mks)

P_mocap_seq = np.zeros((T, N, 3))
P_hpe_seq   = np.zeros((T, N, 3))

for i, mk in enumerate(common_mks):
    P_mocap_seq[:, i, :] = np.stack([frame[mk] for frame in mks_mocap], axis=0)
    P_hpe_seq[:, i, :]   = np.stack([frame[mk] for frame in mks_hpe], axis=0)

R, t, rms = kabsch_global(P_hpe_seq, P_mocap_seq)
print("RMS alignment error:", rms)

P_hpe_seq_aligned = (P_hpe_seq @ R.T) + t  # (T,N,3)
plot_aligned_markers(P_mocap_seq, P_hpe_seq_aligned, common_mks)
mpjpe = compute_mpjpe(P_hpe_seq_aligned, P_mocap_seq)
print("MPJPE (m):", mpjpe)

T, N, _ = P_hpe_seq_aligned.shape
df_hpe_aligned = pd.DataFrame(P_hpe_seq_aligned.reshape(T, 3*N),
                              columns=df_mocap.columns,
                              index=df_mocap.index)

output_csv = f"./data/{subject}/res_hpe/{task}/3d_keypoints_aligned.csv"
df_hpe_aligned.to_csv(output_csv, index=False)
print("Aligned HPE saved to:", output_csv)
