#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import meshcat
from pinocchio.visualize import MeshcatVisualizer
import time

# ---- local imports (assuming this file is in scripts/ or similar) ----
THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))
from utils.utils import read_mks_data
from utils.viz_utils import add_markers_to_meshcat, set_markers_frame
# ----------------------------------------------------------------------

SUBJECT_IDS = [
    "1012","1118","1508","1602","1847","2112","2198","2307","3361",
    "4162","4216","4279","4509","4612","4665","4687","4801","4827"
]
DS_TASKS = [
    "Screwing","ScrewingSat","Crouching","Picking","Hammering","HammeringSat","Jumping","Lifting",
    "QuickLifting","Lower","SideOverhead", "FrontOverhead","RobotPolishing","RobotWelding",
    "Polishing","PolishingSat","SitToStand","Squatting","Static","Upper","CircularWalking","StraightWalking",
    "Welding","WeldingSat"
]

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize COMFI joint_center_positions.csv in Meshcat."
    )
    p.add_argument("--id", dest="subject_id", required=True,
                   help="ID (e.g., 1012)")
    p.add_argument("--task", required=True,
                   help="Task name (e.g., RobotWelding)")
    p.add_argument("--comfi-root", required=True,
                   help="Path to COMFI dataset root.")
    p.add_argument("--freq", type=int, choices=[40, 100], required=True,
                   help="Sampling frequency: 40 (aligned) or 100 (raw).")
    p.add_argument("--start", type=int, default=0,
                   help="Start frame index (inclusive). Default: 0")
    p.add_argument("--stop", type=int, default=None,
                   help="Stop frame index (exclusive). Default: None (till end)")
    return p.parse_args()

def main():
    args = parse_args()

    # Minimal friendly validation
    if args.subject_id not in SUBJECT_IDS:
        raise ValueError(f"Unknown subject ID '{args.subject_id}'. Allowed: {', '.join(SUBJECT_IDS)}")
    if args.task not in DS_TASKS:
        raise ValueError(f"Unknown task '{args.task}'. "
                         f"Allowed: {', '.join(DS_TASKS)}")

    split_folder = "aligned" if args.freq == 40 else "raw"

    comfi_root = Path(args.comfi_root)
    csv_path = comfi_root / "mocap" / split_folder / args.subject_id / args.task / "joint_center_positions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: the task {args.task} is not available for id {args.subject_id}")

    # Load CSV
    df = pd.read_csv(csv_path)
    jcp_dict, start_sample_dict = read_mks_data(df, start_sample=0, converter=1000.0)
    mks_names = list(start_sample_dict.keys())

    # Frame range
    n = len(jcp_dict)
    start = max(0, args.start)
    stop = n if args.stop is None else min(args.stop, n)
    if start >= stop:
        raise ValueError(f"Invalid range: start={start}, stop={stop}, total={n}")

    # Viewer
    viewer = meshcat.Visualizer()
    viz = MeshcatVisualizer()
    viz.initViewer(viewer, open=True)
    viz.viewer.delete()
    native_viz = viz.viewer
    native_viz["/Background"].set_property("top_color", list((1,1,1)))
    native_viz["/Background"].set_property("bottom_color", list((1,1,1)))
    native_viz["/Grid"].set_transform(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -0.0],
        [0, 0, 0, 1]
    ]))


    # Markers
    add_markers_to_meshcat(viewer, jcp_dict, marker_names=mks_names,
                           radius=0.025, default_color=0x00FF00, opacity=0.95)

    # Animate
    for i in range(start, stop):
        set_markers_frame(viewer, jcp_dict, i, marker_names=mks_names, unit_scale=1.0)
        time.sleep(0.80*(1/args.freq))

    print(f"[OK] Visualized {stop - start} frames | ID {args.subject_id} | Task {args.task} | {args.freq} Hz")
    print(f"[SRC] {csv_path}")

if __name__ == "__main__":
    main()
