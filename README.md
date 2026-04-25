# SHARE: User-Centric Collaborative SLAM for AR and Robot Workspaces

SHARE is a multi-agent SLAM system that optimises the quality of experience (QoE) for heterogeneous agents — AR headset users and mobile robots — sharing the same physical workspace. It extends [COVINS-G](https://github.com/VIS4ROB-lab/covins) with a QoE-driven scheduling layer and a selective map-merge mechanism, both coordinated through a lightweight edge server. Per-frame SLAM accuracy estimation uses a DNN whose training labels are generated with [pyslam](https://github.com/luigifreda/pyslam).

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Prerequisites](#prerequisites)
3. [COVINS-G Backend](#1-covins-g-backend)
4. [DNN SLAM Error Evaluator](#2-dnn-slam-error-evaluator)
5. [Monitoring Server](#3-monitoring-server)
6. [Vicon Data Collection Server](#4-vicon-data-collection-server)
7. [Unity Camera Trajectory Emulator](#5-unity-camera-trajectory-emulator)
8. [Running the Full System](#running-the-full-system)
9. [Credits](#credits)
10. [Citation](#citation)
11. [License](#license)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Edge Server                             │
│                                                                 │
│   ┌──────────────────┐       ┌───────────────────────────────┐  │
│   │  COVINS-G Backend│◄─────►│      Monitoring Server        │  │
│   │  (map_be.cpp)    │ UDP   │  (monitoring_server.py /      │  │
│   │                  │       │   qoe_model.py /              │  │
│   │  - Map merging   │       │   volume_overlap.py /         │  │
│   │  - Loop closure  │       │   main.py)                    │  │
│   └──────────────────┘       │                               │  │
│                              │  - QoE computation (§4.2)     │  │
│                              │  - PID scheduler   (§4.3)     │  │
│                              │  - Merge approval  (§4.4)     │  │
│                              └───────────────────────────────┘  │
└───────────┬─────────────────────────────────────────────────────┘
            │ TCP/UDP
    ┌────────┴────────────────────────────────────┐
    │                                             │
    ▼                                             ▼
┌─────────────────────────┐        ┌────────────────────────────┐
│  Human Agent (AR)       │        │     Robot Agent            │
│                         │        │                            │
│  COVINS-G frontend      │        │  COVINS-G frontend         │
│  (ORB-SLAM3 / RealSense)│        │  (ORB-SLAM3 / RealSense)  │
│                         │        │                            │
│  DNN Evaluator          │        │  DNN Evaluator             │
│  (python_evaluator_     │        │  (python_evaluator_        │
│   ondevice.py)          │        │   ondevice.py)             │
└─────────────────────────┘        └────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              Dataset Synthesis Pipeline (offline)               │
│                                                                 │
│  Vicon Server          Unity Emulator       DNN Training        │
│  (server.py /          (controller_v3.cs)   (train_cached.py /  │
│   ViconLogger.py /                           train.py /         │
│   listener_time.py)                          eval_mape.py)      │
└─────────────────────────────────────────────────────────────────┘
```

Each subsystem maps to a directory in this repository:

| Directory | Module | Runs On |
|---|---|---|
| `covins_backend/` | COVINS-G backend patch | Edge server |
| `monitoring_server/` | QoE monitoring and merge control | Edge server |
| `dnn_evaluator/` | SLAM error prediction DNN | Each agent device |
| `vicon_server/` | Ground-truth trajectory logger | Vicon workstation |
| `unity_emulator/` | Synthetic dataset capture | Unity workstation |

---

## Prerequisites

All server-side components and the COVINS-G frontend require **Ubuntu 20.04** and **ROS Noetic**. The steps below follow the [COVINS-G installation guide](https://github.com/VIS4ROB-lab/covins).

### ROS Noetic

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" \
    > /etc/apt/sources.list.d/ros-latest.list'
wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
sudo apt update
sudo apt install ros-noetic-desktop-full
```

### catkin tools

```bash
sudo apt install python3-catkin-tools python3-osrf-pycommon \
                 python3-rosdep python3-rosinstall python3-wstool
sudo rosdep init
rosdep update
```

Add to `~/.bashrc` and source it:

```bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Catkin workspace

Create the workspace that all COVINS-related components share:

```bash
mkdir -p ~/covins_ws/src
cd ~/covins_ws
catkin init
catkin config --extend /opt/ros/noetic/
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

---

## 1. COVINS-G Backend

**Module:** `covins_backend/map_be.cpp`  
**Runs on:** Edge server  
**Replaces:** `~/covins_ws/src/covins/covins_backend/src/covins_backend/map_be.cpp`

### Installing COVINS-G

Install the required system packages ([COVINS-G prerequisites](https://github.com/VIS4ROB-lab/covins)):

```bash
sudo apt install libpthread-stubs0-dev build-essential cmake git doxygen \
                 libsuitesparse-dev libyaml-cpp-dev libvtk6-dev \
                 python3-wstool libomp-dev libglew-dev \
                 libeigen3-dev libboost-all-dev libopencv-dev libceres-dev
```

Clone COVINS into the workspace:

```bash
cd ~/covins_ws/src
git clone https://github.com/VIS4ROB-lab/covins.git
```

Run the automated installation script from the workspace root. It fetches all third-party dependencies (Pangolin, DBoW3, opengv, Eigen catkin, Ceres catkin), patches their `CMakeLists.txt` for Ubuntu 20.04 compatibility, and builds the full catkin workspace. The optional argument specifies the number of parallel build jobs:

```bash
cd ~/covins_ws
chmod +x src/covins/install_file.sh
./src/covins/install_file.sh 8
```

> **Important:** If the build fails at any step, execute the script's commands one by one manually rather than re-running the script from the top. Running the install script multiple times on the same workspace without a clean build in between can leave the workspace in an inconsistent state.

> **Note on Eigen:** The script pins a specific Eigen version via the `eigen_catkin` package and redirects all Eigen dependencies to it. This prevents the fatal linking errors that arise when code is compiled against multiple Eigen versions simultaneously.

The full build (backend + frontend) is the default. To build only specific components:

```bash
# ROS frontend wrapper only (agent devices)
./src/covins/install_file.sh -f 8

# ORB-SLAM3 only, no backend (agent devices)
./src/covins/install_file.sh -o 8
```

After the build completes, source the workspace:

```bash
source ~/covins_ws/devel/setup.bash
```

### ROS Support for the ORB-SLAM3 Frontend

To pass live RealSense sensor data through ROS, build ORB-SLAM3 with ROS support. First install `cv_bridge` against the correct OpenCV version:

```bash
cd ~/covins_ws/src
git clone https://github.com/ros-perception/vision_opencv.git
cd vision_opencv && git checkout noetic
```

Open `cv_bridge/CMakeLists.txt` and add `opencv3_catkin` to the `find_package(catkin ...)` call:

```cmake
find_package(catkin REQUIRED COMPONENTS rosconsole sensor_msgs opencv3_catkin)
```

If OpenCV 4 is present on your system, also remove the lines in that `CMakeLists.txt` that search for an OpenCV 4 version. Then build:

```bash
source ~/covins_ws/devel/setup.bash
catkin build cv_bridge
catkin build ORB_SLAM3
```

Verify correct linkage (should show only `libopencv_core.so.3.4`):

```bash
ldd ~/covins_ws/devel/lib/libcv_bridge.so | grep opencv_core
```

### Configuring the Server IP

On every agent machine, set the edge server address in `covins_comm/config/config_comm.yaml`:

```yaml
sys.server_ip: "<edge-server-ip>"
```

### Applying the SHARE Patch

Replace `map_be.cpp` with the SHARE version:

```bash
cp covins_backend/map_be.cpp \
   ~/covins_ws/src/covins/covins_backend/src/covins_backend/map_be.cpp
```

Edit the four `#define` constants near the top of the file to match your network:

```cpp
#define SHARE_MONITOR_IP                "<edge-server-ip>"
#define SHARE_MERGE_SIGNAL_PORT          9955
#define SHARE_MERGE_APPROVAL_PORT        9956
#define SHARE_MERGE_APPROVAL_TIMEOUT_SEC 1.0
```

Rebuild the backend — no `CMakeLists.txt` changes are required:

```bash
cd ~/covins_ws
catkin build covins_backend
source devel/setup.bash
```

**Ports used:**

| Port | Protocol | Direction | Purpose |
|---|---|---|---|
| 9955 | UDP | COVINS-G → monitoring server | Merge event signals |
| 9956 | UDP | COVINS-G ↔ monitoring server | Merge approval requests |

---

## 2. DNN SLAM Error Evaluator

**Module:** `dnn_evaluator/`  
**Runs on:** Each agent device (live inference); training workstation (offline)

| File | Role |
|---|---|
| `model.py` | Shared `SLAMErrorPredictor` architecture (single source of truth) |
| `dataloader.py` | Dataset loader from cached `.npz` feature files |
| `train_cached.py` | Train from pre-generated cache (no pyslam needed) |
| `train.py` | Train from raw data with on-the-fly pyslam extraction |
| `eval_mape.py` | Evaluate a checkpoint with MAPE, RMSE, MAE, R² |
| `pyslam_integration.py` | Feature extraction and label generation pipeline |

### Python Dependencies

```bash
pip install torch torchvision numpy scipy scikit-learn \
            matplotlib tqdm opencv-python pillow seaborn pandas
```

### Dataset

Training data is derived from the **SenseTime Visual SLAM Dataset** (Li Jinyu et al., VRIH 2019). Expected directory layout:

```
dataset/
  InputData/
    {Scene}/                   # Scene1 … Scene8
      {Sequence}/              # A0 … A7, B0 … B7
        mav0/cam0/data/        # *.png grayscale frames
  Groundtruth/
    {Sequence}/
      groundtruth/data.csv     # timestamp,qx,qy,qz,qw,px,py,pz
```

### pyslam (required for raw feature extraction only)

pyslam is used offline to generate per-frame SLAM error training labels. For each frame, pyslam runs ORB detection and tracking with DBoW3 loop closure to produce an estimated camera pose. This is compared against MoCap ground truth to compute the per-frame localization error label. pyslam's internal tracking state at the same frame also provides the PSD matrix and the eight scalar features, so the full input–label pair for each training sample is produced in a single offline pass. Frames where pyslam reports tracking failure are excluded from the training set.

Install pyslam following the [pyslam installation guide](https://github.com/luigifreda/pyslam):

```bash
git clone --recursive https://github.com/luigifreda/pyslam.git
cd pyslam
./install_all.sh    # creates a Python virtual environment named 'pyslam'
```

The script automatically uses conda if available, or venv otherwise. An internet connection is required and the build takes a considerable amount of time. Core requirements installed by the script:

- Python 3.11.9
- OpenCV ≥ 4.10 (with non-free modules enabled)
- PyTorch ≥ 2.3.1
- Kornia ≥ 0.7.3

In each new terminal where pyslam is needed, activate the environment first:

```bash
. /path/to/pyslam/pyenv-activate.sh
```

Then set the path before running any SHARE training script:

```bash
export PYSLAM_PATH=/path/to/pyslam
```

SHARE uses pyslam's ORB detector with DBoW3 loop closure in monocular mode. The configuration YAML files are in pyslam's `settings/` folder. Pass the path to your ORB config via `--pyslam_config` when calling `train.py`.

If cached `.npz` files are already available, pyslam is not needed — use `train_cached.py` directly.

### Training

```bash
# Recommended: from pre-generated cache (no pyslam needed)
python3 train_cached.py \
  --data_root /path/to/dataset \
  --cache_dir  cached_features \
  --sequences  A0 A1 A2 A3 A4 A5 A6 A7 \
  --scenes     Scene1 Scene2 Scene3 Scene4 Scene5 Scene6 \
  --epochs 20 \
  --output_dir results

# From raw data (requires pyslam environment activated)
. /path/to/pyslam/pyenv-activate.sh
export PYSLAM_PATH=/path/to/pyslam
python3 train.py \
  --data_root     /path/to/dataset \
  --pyslam_config /path/to/pyslam/settings/orb_tum.yaml \
  --cache_dir     cached_features \
  --epochs 20 \
  --output_dir    results
```

### Evaluation

```bash
python3 eval_mape.py \
  --model_path results/best_model.pth \
  --data_root  /path/to/dataset \
  --cache_dir  cached_features \
  --output_dir evaluation_results
```

The trained `best_model.pth` is loaded by `python_evaluator_ondevice.py` on each agent device during live operation.

---

## 3. Monitoring Server

**Module:** `monitoring_server/`  
**Runs on:** Edge server

| File | Role |
|---|---|
| `main.py` | Entry point and CLI argument parsing |
| `monitoring_server.py` | Core server: QoE computation, PID scheduler, merge approval |
| `qoe_model.py` | Agent-specific QoE models (paper §4.2, Eq. 1) |
| `volume_overlap.py` | Camera frustum overlap volume calculator (paper §4.4.1) |

### Python Dependencies

```bash
pip install numpy scipy
```

### Configuration

| Argument | Default | Description |
|---|---|---|
| `--baseline-mode` | `full` | `full` / `monitor_only` / `delay_linear` / `same_qoe` |
| `--clock-offset-ms` | `0.0` | Constant NTP clock correction (ms) subtracted from raw RTT |
| `--alpha-m` | `0.5` | α_m in P_skip = max(0, 1 − α_m · V_new / V_bar) |
| `--human1-eval-port` | `9997` | TCP port for Human Agent 1 evaluation messages |
| `--human2-eval-port` | `9998` | TCP port for Human Agent 2 evaluation messages |
| `--robot-eval-port` | `9999` | TCP port for Robot evaluation messages |

### Running

Start the monitoring server **before** COVINS-G and before any agent devices connect:

```bash
# Full SHARE system
python3 main.py --baseline-mode full --clock-offset-ms <measured_ntp_offset>

# Equal-QoE ablation baseline (paper §6.3)
python3 main.py --baseline-mode same_qoe

# Linear latency-based drop baseline (paper §6.3)
python3 main.py --baseline-mode delay_linear
```

**Ports used:**

| Port | Protocol | Direction | Purpose |
|---|---|---|---|
| 9997 | TCP | Agent → server | Human Agent 1 evaluation |
| 9998 | TCP | Agent → server | Human Agent 2 evaluation |
| 9999 | TCP | Agent → server | Robot evaluation |
| 6161–6163 | TCP | Agent → server | Pose streams |
| 9955 | UDP | COVINS-G → server | Merge event signals |
| 9956 | UDP | COVINS-G ↔ server | Merge approval |

---

## 4. Vicon Data Collection Server

**Module:** `vicon_server/`  
**Runs on:** Vicon workstation (Ubuntu 20.04, ROS Noetic, Vicon bridge)

| File | Role |
|---|---|
| `server.py` | Central clock synchronisation and collection control hub |
| `listener_time.py` | ROS timestamp relay; launched automatically as a subprocess by `server.py` |
| `ViconLogger.py` | Subscribes to Vicon ROS topics and writes ground-truth poses to CSV |

### System Requirements

Ubuntu 20.04, ROS Noetic (see [Prerequisites](#prerequisites)), and the Vicon ROS bridge:

```bash
sudo apt install ros-noetic-vicon-bridge
# or build from source: https://github.com/ethz-asl/vicon_bridge
```

### Python Dependencies

```bash
pip install rospkg
```

### Startup Order

```bash
# 1. Start ROS core
roscore

# 2. Launch the Vicon bridge (publishes /vicon/<obj>/<obj> topics)
roslaunch vicon_bridge vicon.launch datastream_hostport:=<vicon-pc-ip>:801

# 3. Start the trajectory logger
python3 ViconLogger.py \
  --output-dir ./vicon_trajectory_logs \
  --udp-port 10000

# 4. Start the synchronisation and control server
#    (listener_time.py is launched automatically as a subprocess)
python3 server.py \
  --sync-port 6666 \
  --vicon-ctrl-port 10000
```

### Collection Control

Once all agent devices have registered and finished synchronisation, issue commands at the `server.py` prompt:

```
> begin    # opens CSV files in ViconLogger and broadcasts Start Collection
> end      # closes CSV files and broadcasts End Collection
> exit     # shuts down the server
```

### Output Files

| File | Content |
|---|---|
| `vicon_trajectory_logs/<Obj>_<timestamp>.csv` | Ground-truth poses (space-separated: timestamp tx ty tz qx qy qz qw) |
| `sync_completion.txt` | Vicon time when all devices finished syncing |
| `collection_start.txt` | Vicon time when collection began |
| `collection_end.txt` | Vicon time when collection ended |

**Ports used:**

| Port | Protocol | Direction | Purpose |
|---|---|---|---|
| 6666 | UDP | Device → server | Device registration (`DeviceName:DeviceIP`) |
| 8888 | UDP | Server → device | Clock sync timestamp packets |
| 10000 | UDP | Server → ViconLogger | Collection control commands |
| 11111 | UDP | Server → SensorCollector | SensorCollector control (if present) |

---

## 5. Unity Camera Trajectory Emulator

**Module:** `unity_emulator/controller_v3.cs`  
**Runs on:** Unity workstation

### Requirements

- **Unity 2022.3.46 LTS**
- **High Definition Render Pipeline (HDRP)** — install via the Unity Package Manager

### Setup

1. Create a new Unity HDRP project and import or build the virtual environment scenes used in your study.

2. For each agent (e.g. Human Agent 1, Human Agent 2, Robot), create a GameObject with:
   - A `Camera` component
   - `CameraPoseController.cs` attached

3. Assign the pose `.txt` file as the **Pose Data File** (`TextAsset`) in the Inspector.

   **Pose file format** — space-separated, one pose per line:
   ```
   timestamp  qx  qy  qz  qw  px  py  pz
   ```
   The source coordinate frame is assumed to be ROS/ENU right-handed (X-forward, Y-left, Z-up). The script converts to Unity's left-handed frame automatically. If your source data uses a different convention, adjust the two lines marked `COORDINATE CONVERSION` in `ParsePoseData()`.

4. Leave **Save Directory** empty — it will auto-name from the GameObject name, ensuring each agent saves frames to a separate folder.

5. Set **Target Frame Rate** to `30` to match the field-study recording rate.

6. Ensure **Use Relative Positioning** is **unchecked** (false) for dataset generation.

---

## Running the Full System

### Network Setup

Assign static IPs to all machines on the same local network. Replace `<server-ip>` throughout with your actual address. On every agent machine, set in `covins_comm/config/config_comm.yaml`:

```yaml
sys.server_ip: "<server-ip>"
```

### Step 1 — Edge Server: ROS core

```bash
roscore
```

### Step 2 — Edge Server: COVINS-G Backend

In a new terminal with the workspace sourced:

```bash
source ~/covins_ws/devel/setup.bash
rosrun covins_backend covins_backend_node
```

### Step 3 — Edge Server: Monitoring Server

In a new terminal:

```bash
cd monitoring_server/
python3 main.py --clock-offset-ms <measured_ntp_offset>
```

### Step 4 — Human Agent 1: DNN Evaluator

SSH into the first human agent device and navigate to the COVINS frontend source directory:

```bash
python3 python_evaluator_ondevice.py \
  --model best_model.pth \
  --monitor-port 9997 \
  --monitor-host <server-ip>
```

### Step 5 — Human Agent 1: COVINS-G Frontend

In a second terminal on the same device with the workspace sourced:

```bash
source ~/covins_ws/devel/setup.bash
roslaunch realsense_ros realsense_launch.launch
```

### Step 6 — Human Agent 2: DNN Evaluator

SSH into the second human agent device:

```bash
python3 python_evaluator_ondevice.py \
  --model best_model.pth \
  --monitor-port 9998 \
  --monitor-host <server-ip>
```

### Step 7 — Human Agent 2: COVINS-G Frontend

```bash
source ~/covins_ws/devel/setup.bash
roslaunch realsense_ros realsense_launch.launch
```

### Step 8 — Robot Agent: DNN Evaluator

SSH into the robot and open the LXC Ubuntu 20 container. Inside the container, navigate to the COVINS frontend source directory:

```bash
python3 python_evaluator_ondevice.py \
  --model best_model.pth \
  --monitor-port 9999 \
  --socket /tmp/eval.sock \
  --monitor-host <server-ip>
```

The `--socket` flag routes evaluation output through a Unix domain socket, required when the robot's SLAM process and evaluator share a container environment.

### Step 9 — Robot Agent: COVINS-G Frontend

In a second terminal in the same container with the workspace sourced:

```bash
source ~/covins_ws/devel/setup.bash
roslaunch realsense_ros realsense_launch.launch
```

### Step 10 — Verify and Monitor

Once all frontends are running and submitting keyframes, the COVINS-G backend will begin collaborative mapping. The monitoring server logs QoE values, frame drop rates, and merge decisions to CSV files in its working directory. COVINS-G's Pangolin visualiser opens automatically on the server when `covins_backend_node` starts.

---

## Credits

SHARE builds on the following open-source projects:

**COVINS-G** — Generic back-end for collaborative visual-inertial SLAM.  
Patel, M., Karrer, M., Bänninger, P., and Chli, M. *COVINS-G: A Generic Back-end for Collaborative Visual-Inertial SLAM.* ICRA 2023.  
Repository: [https://github.com/VIS4ROB-lab/covins](https://github.com/VIS4ROB-lab/covins)  
The SHARE backend patch modifies COVINS-G source code and is therefore released under the same GNU General Public License v3.0.

**pyslam** — Open-source, modular, and extensible framework for SLAM.  
Freda, L. *pySLAM: An Open-Source, Modular, and Extensible Framework for SLAM.* RSS 2025 Workshop: Unifying Visual SLAM.  
Repository: [https://github.com/luigifreda/pyslam](https://github.com/luigifreda/pyslam)  
Used in SHARE for offline per-frame SLAM error label generation and ORB feature extraction during DNN training.

---

## Citation

```bibtex
@inproceedings{share2026imwut,
  title     = {SHARE: User-Centric Collaborative SLAM for AR and Robot Workspaces},
  booktitle = {Proceedings of the ACM on Interactive, Mobile, Wearable
               and Ubiquitous Technologies (IMWUT)},
  year      = {2026},
}

@inproceedings{patel2023covinsg,
  title     = {{COVINS-G}: A Generic Back-end for Collaborative Visual-Inertial SLAM},
  author    = {Patel, Manthan and Karrer, Marco and B{\"a}nninger, Philipp
               and Chli, Margarita},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2023},
}

@inproceedings{freda2025pyslam,
  title     = {pySLAM: An Open-Source, Modular, and Extensible Framework for SLAM},
  author    = {Freda, Luigi},
  booktitle = {RSS 2025 Workshop: Unifying Visual SLAM},
  year      = {2025},
}

@article{li2019survey,
  title     = {Survey and Evaluation of Monocular Visual-Inertial SLAM Algorithms
               for Augmented Reality},
  author    = {Li, Jinyu and Xu, Bangbang and Chen, Guofeng and Chen, Shaojie
               and Luo, Yijin and Tian, Wei and Bao, Hujun},
  journal   = {Journal of Virtual Reality and Intelligent Hardware},
  volume    = {1},
  number    = {4},
  pages     = {386--410},
  year      = {2019},
}
```

---

## License

This project is released under the **GNU General Public License v3.0**, consistent with the COVINS-G codebase on which it depends.
