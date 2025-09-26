import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton, Custom, PoseTracker
from functools import partial
import csv
import os

# ---------------- Config ---------------- #
cam_ids = [0, 4, 6]
task = 'robot_welding'
subject = 'Alessandro'

device = 'cpu'  # 'cpu', 'cuda'
backend = 'onnxruntime'
openpose_skeleton = False
show_realtime = True   # <- set to False if you don’t want live display
# ----------------rtmlib-------------------- #
#refer to rtmlib repo for more details
custom = partial(
    Custom,
    to_openpose=openpose_skeleton,
    det_class='YOLOX',
    det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
    det_input_size=(640, 640),
    pose_class='RTMPose',
    pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip',
    pose_input_size=(192, 256),
    backend=backend,
    device=device
)

pose_tracker = PoseTracker(
    custom,
    det_frequency=10,
    tracking=False,
    tracking_thr=0.1,
    to_openpose=openpose_skeleton,
    backend=backend,
    device=device
)

for cam_id in cam_ids:
    print(f"\n=== Processing camera {cam_id} ===")

    video_path = f'./data/{subject}/videos/{task}/camera_{cam_id}.mp4'
    output_path = f'./data/{subject}/videos/{task}/keypoints_video_{cam_id}.avi'
    csv_output = f'./data/{subject}/res_hpe/{task}/keypoints_cam{cam_id}.csv'

    if not os.path.exists(video_path):
        print(f"⚠️  Video not found: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        continue

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_id = 0
    person_index = None  # keep the first detected person

    with open(csv_output, mode='w', newline='') as f:
        writer = csv.writer(f)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, scores = pose_tracker(frame)

            if keypoints.shape[0] > 0:
                if frame_id == 0:
                    person_index = 0  # choose the first person

                if person_index is not None and person_index < keypoints.shape[0]:
                    keypoints = keypoints[person_index:person_index+1]
                    scores = scores[person_index:person_index+1]
                else:
                    keypoints, scores = None, None

                if keypoints is not None:
                    frame_with_skeleton = draw_skeleton(
                        frame.copy(), keypoints, scores, kpt_thr=0.5
                    )
                else:
                    frame_with_skeleton = frame.copy()
            else:
                frame_with_skeleton = frame.copy()

            out.write(frame_with_skeleton)
            print(f"[cam {cam_id}] Frame #{frame_id}")
            frame_id += 1

            if keypoints is not None and scores is not None:
                keypoints_flat = keypoints.flatten().tolist()
                scores_flat = scores.flatten().tolist()
                scores_mean = [np.mean(scores_flat)]
                writer.writerow(scores_mean + keypoints_flat)

            # Optional real-time display
            if show_realtime:
                cv2.imshow(f'Skeleton Video - cam {cam_id}', frame_with_skeleton)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    out.release()

if show_realtime:
    cv2.destroyAllWindows()

print("\n✅ Processing finished for all cameras")
