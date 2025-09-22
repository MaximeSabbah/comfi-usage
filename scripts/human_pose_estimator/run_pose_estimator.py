import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton, Custom, PoseTracker
from functools import partial
import csv
# Configuration
cam_id = 0
task = 'static'
subject = 'Maxime'

video_path = f'/datasets/cosmik_data/subjects/{subject}/mouv/{task}/camera_{cam_id}.mp4'
output_path = f'/datasets/cosmik_data/subjects/{subject}/mouv/{task}/keypoints_video_{cam_id}.avi'
csv_output = f'/datasets/cosmik_data/subjects/{subject}/mouv/{task}/keypoints_cam{cam_id}.csv'

device = 'cuda'  # 'cpu', 'cuda', 'mps'
backend = 'onnxruntime'
openpose_skeleton = False

# Initialisation du modèle
custom = partial(
    Custom,
    to_openpose=openpose_skeleton,
    det_class='YOLOX',
    det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
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
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # ou 'MJPG', 'MP4V', etc.
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Traitement frame par frame
frame_id = 0
with open(csv_output, mode='w', newline='') as f:
    writer = csv.writer(f)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, scores = pose_tracker(frame)
        if keypoints.shape[0] > 0:
            frame_with_skeleton = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.5)
	else:
    	    frame_with_skeleton = frame.copy()
        
        frame_with_skeleton = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.1)

        out.write(frame_with_skeleton)
        print(f"Frame #{frame_id}")
        frame_id += 1
        
        # Sauvegarde dans le CSV
        if keypoints is not None and scores is not None:
            keypoints_flat = keypoints.flatten().tolist()  # x1, y1, x2, y2, ...
            scores_flat = scores.flatten().tolist()        # s1, s:2, ...
            scores_mean = [np.mean(scores_flat)]
            writer.writerow(scores_mean + keypoints_flat)

        # Optionnel : afficher en temps réel
        # cv2.imshow('Skeleton Video', frame_with_skeleton)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
	
# Nettoyage
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Vidéo sauvegardée sous : {output_path}")
