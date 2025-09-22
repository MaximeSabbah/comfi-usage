import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import meshcat_shapes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from utils.urdf_utils import build_human_model, lock_joints, load_robot_panda
from utils.viz_utils import box_between_frames, set_tf, draw_table, addViewerBox, make_visuals_gray, animate
from example_robot_data import load
from utils.utils import compute_time_sync,load_cameras_from_soder, load_robot_base_pose, load_all_data

@dataclass
class Paths:
    mks_csv: str
    q_ref_csv: str
    robot_csv: str
    cam0_ts_csv: str
    urdf_path: str
    urdf_meshes_path: str
    robot_base_yaml: str
    jcp_csv: str
    soder_paths: Dict[str, str] 

@dataclass
class Scene:
    viewer: meshcat.Visualizer
    viz_human: MeshcatVisualizer
    viz_robot: MeshcatVisualizer
    human_model: pin.Model
    human_data: pin.Data
    robot_model: pin.Model
    robot_data: pin.Data


def define_scene(urdf_path: str,
                 urdf_meshes_path: str,
                 T_world_robot: np.ndarray,
                 cameras: Dict[str, np.ndarray],
                 forceplates_dims_and_centers: Tuple[List[Tuple[float,float]], List[Tuple[float,float,float]]],
                 bg_top=(1,1,1), bg_bottom=(1,1,1), grid_height=-0.0) -> Scene:
    """
    Builds:
      - Meshcat viewer (shared)
      - Human model (locked joints + gray)
      - Panda model (robot)
      - Background, grid, force plates, camera boxes + frames + links, labels, world frames
    Returns handles so the rest of the code is clean.
    """
    # Human base
    model_h, coll_h, vis_h, _ = build_human_model(urdf_path, urdf_meshes_path)
    # make visuals gray
    make_visuals_gray(vis_h)

    # Lock joints
    joints_to_lock = [
        "middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z",
        "left_wrist_X", "left_wrist_Z", "right_wrist_X", "right_wrist_Z"
    ]
    model_h, coll_h, vis_h, data_h = lock_joints(model_h, coll_h, vis_h, joints_to_lock)

    # Panda
    model_r, coll_r, vis_r, data_r = load_robot_panda()

    # Shared Meshcat
    viewer = meshcat.Visualizer()

    # Visualizers
    viz_human = MeshcatVisualizer(model_h, coll_h, vis_h)
    viz_human.initViewer(viewer, open=True)
    viz_human.viewer.delete()  # clear if relaunch
    viz_human.loadViewerModel("ref")
    viz_human.display(pin.neutral(model_h))

    viz_robot = MeshcatVisualizer(model_r, coll_r, vis_r)
    viz_robot.initViewer(viewer)
    viz_robot.loadViewerModel(rootNodeName="panda")
    viz_robot.viewer["panda"].set_transform(T_world_robot)

    # Background/grid
    native_viz = viz_human.viewer
    native_viz["/Background"].set_property("top_color", list(bg_top))
    native_viz["/Background"].set_property("bottom_color", list(bg_bottom))
    native_viz["/Grid"].set_transform(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, grid_height],
        [0, 0, 0, 1]
    ]))

    # Force plates
    fp_dim, fp_centers = forceplates_dims_and_centers
    for j, ((sx, sy), (cx, cy, cz)) in enumerate(zip(fp_dim, fp_centers), start=1):
        name = f"force_plate_{j}"
        addViewerBox(viz_robot, name, sx, sy, 0.01, rgba=[0.5, 0.5, 0.5, 1.0])
        T = np.eye(4)
        T[:3, 3] = [cx, cy, cz + 0.01/2.0]
        set_tf(viz_robot, name, T)

    # Some frames and labels
    meshcat_shapes.textarea(viz_robot.viewer["R_world_text"], "R0", font_size=32)
    meshcat_shapes.frame(viz_robot.viewer["R_world"], axis_length=0.4, axis_thickness=0.04, opacity=1, origin_radius=0.02)

    meshcat_shapes.frame(viz_robot.viewer["R_robot"], axis_length=0.4, axis_thickness=0.02, opacity=1, origin_radius=0.02)
    set_tf(viz_robot, "R_robot", T_world_robot)

    # Camera boxes + frames + links + text
    # Expect keys like "c0","c2","c4","c6"
    cam_order = sorted(cameras.keys())  # stable order
    def cam_frame_name(k): return f"f_cam_{k}"
    def cam_text_name(k):  return f"R_c{k}_text"
    def cam_box_name(k):   return f"cam_{k}"

    # optional: connect pairs with links if c0<->c2 and c4<->c6 exist
    def safe_box_between(a, b, name):
        if a in cameras and b in cameras:
            box_between_frames(viz_robot, name, cameras[a], cameras[b], thickness=0.1, height=0.1, rgba=(0.01,0.01,0.01,0.9))

    safe_box_between("0","2","link_c0_c2")
    safe_box_between("4","6","link_c4_c6")

    for k in cam_order:
        Tck = cameras[k]
        meshcat_shapes.frame(viz_robot.viewer[cam_frame_name(k)], axis_length=0.2, axis_thickness=0.02, opacity=0.8, origin_radius=0.02)
        set_tf(viz_robot, cam_frame_name(k), Tck)

        addViewerBox(viz_robot, cam_box_name(k), 0.1, 0.1, 0.1, rgba=[0.01, 0.01, 0.01, 1.0])
        set_tf(viz_robot, cam_box_name(k), Tck)

        meshcat_shapes.textarea(viz_robot.viewer[cam_text_name(k)], f"cam{k}", font_size=28)
        Ttxt = np.array(Tck)
        Ttxt = Ttxt.copy()
        Ttxt[2, 3] += 0.1
        set_tf(viz_robot, cam_text_name(k), Ttxt)

    # Table
    T_world_table = np.eye(4)
    T_world_table[:3, 3] = [0.9, -0.6, 0.0]
    draw_table(viz_robot, T_world_table)

    return Scene(
        viewer=viewer,
        viz_human=viz_human,
        viz_robot=viz_robot,
        human_model=model_h,
        human_data=data_h,
        robot_model=model_r,
        robot_data=data_r
    )


def main():
    #paths (adjust to your env)
    paths = Paths(
        mks_csv = "./data/Alessandro/mocap/robot_welding/mocap_downsampled_to_40hz.csv",
        q_ref_csv = "./data/Alessandro/mocap/robot_welding/q_mocap.csv",
        robot_csv = "./data/Alessandro/mocap/Alessandro_robot_welding.csv",
        cam0_ts_csv = "./data/Alessandro/camera_0_timestamps.csv",
        urdf_path = "./model/urdf/4279_scaled.urdf",
        urdf_meshes_path =os.path.abspath("model"),
        robot_base_yaml = "./data/Alessandro/mocap/robot_base_pose.yaml",
        jcp_csv = "./data/Alessandro/mocap/robot_welding/joint_center_positions.csv",
        soder_paths = {
            "0": "./data/Alessandro/config/soder_0.txt",
            "2": "./data/Alessandro/config/soder_2.txt",
            "4": "./data/Alessandro/config/soder_4.txt",
            "6": "./data/Alessandro/config/soder_6.txt",
        }
    )

    #read all data
    payload = load_all_data(paths, start_sample=0, converter=1000.0)
    mks_dict = payload["mks_dict"]
    mks_names = payload["mks_names"]
    q_ref = payload["q_ref"]
    q_robot = payload["q_robot"]
    t_cam = payload["t_cam"]
    t_robot = payload["t_robot"]
    jcp = payload["jcp"]

    # transforms (robot base + cameras)
    T_world_robot = load_robot_base_pose(paths.robot_base_yaml)
    cameras = load_cameras_from_soder(paths.soder_paths)

    # define the scene
    fp_dims = [(0.5,0.6), (0.50,0.60), (0.50,0.60), (0.9,1.8), (0.5,0.6)]
    fp_centers = [(-0.830,-0.3,0.0), (-0.25,-0.3,0.0), (0.39,-0.3,0.0), (-1.68,-0.3,0.0), (-0.25,0.3,0.0)]
    scene = define_scene(
        urdf_path=paths.urdf_path,
        urdf_meshes_path=paths.urdf_meshes_path,
        T_world_robot=T_world_robot,
        cameras=cameras,
        forceplates_dims_and_centers=(fp_dims, fp_centers),
        bg_top=(1,1,1), bg_bottom=(1,1,1), grid_height=-0.0
    )

    #time syn between cameras and robot data
    sync = compute_time_sync(t_cam, t_robot, tol_ms=5)
    if sync:
        print("Synced at:", sync)
    else:
        print("No time sync match found (even within tolerance).")

    #animattion
    animate(scene, mks_dict, mks_names, q_ref, q_robot, jcp, sync, step=5, i0=0)

if __name__ == "__main__":
    main()
