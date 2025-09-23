import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton, Custom, PoseTracker
from functools import partial
import csv
# Configuration
cam_id = 6
task = 'robot_welding'
subject = 'Alessandro'

video_path = f'./data/{subject}/videos/{task}/camera_{cam_id}.mp4'
output_path = f'./data/{subject}/videos/{task}/keypoints_video_{cam_id}.avi'
csv_output = f'./data/{subject}/res_hpe/{task}/keypoints_cam{cam_id}.csv'

device = 'cpu'  # 'cpu', 'cuda'
backend = 'onnxruntime'
openpose_skeleton = False

# Initialisation du modèle
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
    tracking = False,
    tracking_thr = 0.1,
    to_openpose=openpose_skeleton,
    backend=backend,
    device=device
)

# Ouverture de la vidéo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Obtenir les infos de la vidéo pour créer l'output
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # ou 'MJPG', 'MP4V', etc.
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_id = 0
person_index = None  # stock first person index

with open(csv_output, mode='w', newline='') as f:
    writer = csv.writer(f)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, scores = pose_tracker(frame)

        if keypoints.shape[0] > 0:
            # we took first person detected in first frame
            if frame_id == 0:
                person_index = 0  

            # we keep only the first one detected
            if person_index is not None and person_index < keypoints.shape[0]:
                keypoints = keypoints[person_index:person_index+1]
                scores = scores[person_index:person_index+1]
            else:
                # ignore
                keypoints = None
                scores = None

            if keypoints is not None:
                frame_with_skeleton = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.5)
            else:
                frame_with_skeleton = frame.copy()
        else:
            frame_with_skeleton = frame.copy()

        out.write(frame_with_skeleton)
        print(f"Frame #{frame_id}")
        frame_id += 1

        
        if keypoints is not None and scores is not None:
            keypoints_flat = keypoints.flatten().tolist() 
            scores_flat = scores.flatten().tolist()       
            scores_mean = [np.mean(scores_flat)]
            writer.writerow(scores_mean + keypoints_flat)

        # to display in realtime
        # cv2.imshow('Skeleton Video', frame_with_skeleton)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
	
# Nettoyage
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Vidéo sauvegardée sous : {output_path}")
