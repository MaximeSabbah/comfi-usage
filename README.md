# comfi-usage
Scripts to use and showcase the data from the COMFI dataset

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/MaximeSabbah/comfi-usage.git
cd comfi-usage
```
2. **Virtual environment**
```bash
python3 -m venv comfi_env
source comfi_env/bin/activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
**Usage**
This repository provides scripts for processing and visualizing COMFI dataset data.
By default, Meshcat opens a local server (usually at http://127.0.0.1:7000) to stream the visualization in your browser.

1.**Human pose estimation (HPE)**

Run the RTMlib pose estimator (YOLOX + RTMPose-M with 26 keypoints):
```bash
python run_pose_estimator.py --id 1012 --task RobotWelding --comfi-root /path/to/COMFI
```
2.**Triangulation and visualization**
Triangulate keypoints from multiple camera views:
```bash
python run_triangulation.py --id 4279 --task RobotWelding --nb-cams 4
```
**Visualize triangulated data:**
```bash
python scripts/visualization/viz_jcp.py \
    --mode hpe \
    --id 4279 \
    --task RobotWelding \
    --nb-cams 4 \
    --freq 40 \
    --start 0 \
    --stop 500
```
3.**Visualize mocap markers + JCP**
```bash
python viz_mks.py \
    --id 1012 \
    --task RobotWelding \
    --comfi-root /path/to/COMFI \
    --freq 40 \
    --mkset est \
    --with_jcp \
    --start 100 \
    --stop 500
```
```bash
python viz_all_data.....
```
4.**Extract JCP from mocap markers (using our markerset):**
```bash
python get_jcp_from_mocap_markers.py \
    --id 1012 \
    --task RobotWelding \
    --comfi-root /path/to/COMFI \
    --freq 40 \
    --mkset est

```
5.**Alignment**
```bash
python scripts/run_procruste_alignement.py \
    --id 4279 \
    --task RobotWelding \
    --comfi-root /path/to/COMFI \
    --nb-cams 4
```