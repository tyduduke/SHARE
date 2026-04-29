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

> **Resource requirement:** Building COVINS-G is memory-intensive. Ensure the build machine has at least 16 GB RAM. If building in a virtual machine, allocate at least 16 GB RAM and 10 GB swap. Instructions for changing swap size can be found [here](https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-20-04).

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

### What this file is and what was changed

`map_be.cpp` is a **direct modification of the original COVINS-G source file** located at:

```
~/covins_ws/src/covins/covins_backend/src/covins_backend/map_be.cpp
```

The SHARE version of this file must **replace** the original after COVINS-G is built. It is not a new file or a plugin — it modifies the existing COVINS-G map manager (`MapManager` class) in place.

Two capabilities were added to the original COVINS-G `PerformMerge()` function:

1. **Merge event signalling:** Before and after every map merge, the patched `PerformMerge()` sends a UDP notification to the monitoring server (`SendMergeSignal`). The `merge_complete` signal is how the monitoring server maintains its historical overlap volume buffer (V_bar) for the P_skip formula. Without this signal, the monitoring server cannot compute a meaningful merge approval decision.

2. **Merge approval gating:** Before executing any merge, the patched `PerformMerge()` calls `RequestMergeApproval()`, which sends a UDP request to the monitoring server and waits for a binary approved/denied response. The monitoring server applies the selective merge criterion (paper §4.4.2, Eq. 6, `P_skip = max(0, 1 - α_m · V_new / V_bar)`) and returns its decision. If denied, the merge request is silently discarded — this is the intended selective merging behaviour. If the monitoring server is unreachable or times out, the default is to approve (fail-safe).

No other parts of the COVINS-G codebase were modified. No `CMakeLists.txt` changes are required. All new networking code uses standard POSIX headers (`sys/socket.h`, `poll.h`, `arpa/inet.h`) that are available on any Linux system without additional packages.

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

> **Note on `CMAKE_BUILD_TYPE`:** Pay attention to the `CMAKE_BUILD_TYPE` set in the catkin config above (`RelWithDebInfo`). In particular, building parts of the code with `march=native` can cause problems on some machines. Do not override this flag without careful consideration.

> **Note on Eigen:** The script pins a specific Eigen version via the `eigen_catkin` package and redirects all Eigen dependencies to it. This prevents the fatal linking errors that arise when code is compiled against multiple Eigen versions simultaneously.

> **Build failure troubleshooting:** If the build process of COVINS-G or ORB-SLAM3 fails, make sure you have correctly sourced the workspace before building, and verify that the libraries in the third-party folders — in particular DBoW2 and g2o — have been built correctly before attempting to build the main packages.

The COVINS-G **backend only needs to be built on the edge server**. For agent devices, building just the frontend is sufficient. The COVINS-G ROS frontend wrapper (`-f` flag) is useful when you want to use a RealSense T265 tracking camera or any arbitrary VIO/stereo frontend without modifying the frontend code.

The full build (backend + frontend) is the default. To build only specific components:

```bash
# Full build — backend + frontend (edge server)
./src/covins/install_file.sh 8

# ROS frontend wrapper only — for agent devices using arbitrary VIO/tracking cameras
./src/covins/install_file.sh -f 8

# ORB-SLAM3 only, no backend — for agent devices using ORB-SLAM3 directly
./src/covins/install_file.sh -o 8
```

After the build completes, source the workspace:

```bash
source ~/covins_ws/devel/setup.bash
```

### Configuring the COVINS-G Backend Type

COVINS-G supports two backend modes: `COVINS` (uses map points, supports global bundle adjustment) and `COVINS_G` (generic, works with any frontend, no map points in backend). SHARE requires `COVINS_G`. Set this in the backend configuration file at `~/covins_ws/src/covins/covins_backend/config/config_backend.yaml`:

```yaml
placerec.type: COVINS_G
```

### ROS Support for the ORB-SLAM3 Frontend

To pass live sensor data through ROS, build ORB-SLAM3 with ROS support. First install `cv_bridge` against the correct OpenCV version:

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

Verify correct linkage for `cv_bridge` (should show only `libopencv_core.so.3.4`):

```bash
ldd ~/covins_ws/devel/lib/libcv_bridge.so | grep opencv_core
```

Verify correct linkage for the ORB-SLAM3 ROS node itself (should also list only `libopencv_core.so.3.4`):

```bash
ldd ~/covins_ws/src/covins/orb_slam3/Examples/ROS/ORB_SLAM3/Mono_Inertial | grep opencv_core
```

If either check lists a version other than `libopencv_core.so.3.4`, there is an OpenCV linkage conflict that will cause runtime crashes. Resolve it by re-checking the `CMakeLists.txt` edits above before proceeding.

### Configuring the Server IP

On every agent machine, set the edge server address in `covins_comm/config/config_comm.yaml`:

```yaml
sys.server_ip: "<edge-server-ip>"
```

### Applying the SHARE Patch

After COVINS-G is fully built, replace the original `map_be.cpp` with the SHARE-modified version. The file in this repository (`covins_backend/map_be.cpp`) must overwrite the original at its exact location inside the COVINS-G workspace:

```bash
cp covins_backend/map_be.cpp \
   ~/covins_ws/src/covins/covins_backend/src/covins_backend/map_be.cpp
```

Edit the four `#define` constants near the top of the file to match your network. `SHARE_MONITOR_IP` must be the IP of the machine running `main.py`:

```cpp
#define SHARE_MONITOR_IP                "<edge-server-ip>"
#define SHARE_MERGE_SIGNAL_PORT          9955
#define SHARE_MERGE_APPROVAL_PORT        9956
#define SHARE_MERGE_APPROVAL_TIMEOUT_SEC 1.0
```

Rebuild the backend. Only `covins_backend` needs to be rebuilt — no other packages are affected:

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

The DNN evaluator predicts per-keyframe SLAM tracking error from three input modalities: visual keyframe images, Point Spatial Distribution (PSD) matrices, and scalar tracking-quality features. The predicted error is used as the accuracy term in the QoE model (paper §4.2). At runtime, the inference client `python_evaluator_ondevice.py` (located in the COVINS frontend source directory at `~/covins_ws/src/covins/covins_frontend/src/`) loads the trained model checkpoint and sends per-keyframe error estimates to the monitoring server over TCP.

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

The install script creates a single Python environment named `pyslam` that hosts all supported components. On Ubuntu, three environment managers are supported:

**With venv (recommended):**
The default path when conda is not detected. The script installs and configures a venv automatically. No additional steps are needed before running `./install_all.sh`.

**With conda:**
If conda is available on the system, the script automatically uses it to create and populate the `pyslam` environment. After installation, activate with:
```bash
conda activate pyslam
```
instead of using `pyenv-activate.sh`.

**With pixi (experimental):**
Run `pixi shell` in the root folder of the pyslam repository before launching `./install_all.sh`. This option may encounter issues with building and linking on some machines and is not recommended for production use.

An internet connection is required and the build takes a considerable amount of time. Core requirements installed by the script:

- Python 3.11.9
- OpenCV ≥ 4.10 (with non-free modules enabled — handled automatically by the install script)
- PyTorch ≥ 2.3.1
- Kornia ≥ 0.7.3

After installation, verify that the correct OpenCV version with non-free modules was installed:

```bash
. pyenv-activate.sh           # activate the pyslam environment (venv path)
./scripts/opencv_check.py     # prints the installed OpenCV version and module list
```

In each new terminal where pyslam is needed, activate the environment first (venv path):

```bash
. /path/to/pyslam/pyenv-activate.sh
```

Then set the path before running any SHARE training script:

```bash
export PYSLAM_PATH=/path/to/pyslam
```

SHARE uses pyslam's ORB detector with DBoW3 loop closure in monocular mode. The configuration YAML files are in pyslam's `settings/` folder. Pass the path to your ORB config via `--pyslam_config` when calling `train.py`.

> **Troubleshooting:** If you encounter errors during installation or at runtime, check `docs/TROUBLESHOOTING.md` inside the pyslam repository before opening an issue. Common causes include missing CUDA toolkit, conflicting conda environments, and missing submodules (ensure `git clone --recursive` was used).

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

Additional training arguments for both scripts:

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | `16` | Training batch size |
| `--lr` | `0.001` | Initial learning rate (Adam optimizer) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--test_size` | `0.3` | Fraction of data held out for test + validation |
| `--val_size` | `0.5` | Fraction of the held-out set used for validation |

### Evaluation

```bash
python3 eval_mape.py \
  --model_path results/best_model.pth \
  --data_root  /path/to/dataset \
  --cache_dir  cached_features \
  --output_dir evaluation_results
```

Additional evaluation arguments:

| Argument | Default | Description |
|---|---|---|
| `--eval_on` | `test` | Which split to evaluate: `train`, `val`, `test`, or `all` |
| `--batch_size` | `16` | Inference batch size |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--test_size` | `0.3` | Must match the value used during training |
| `--val_size` | `0.5` | Must match the value used during training |

The trained `best_model.pth` is loaded by `python_evaluator_ondevice.py` on each agent device during live operation. This inference client connects to the monitoring server over TCP on the agent's assigned evaluation port (9997, 9998, or 9999) and sends one error estimate per keyframe.

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

The monitoring server exposes all parameters as CLI arguments. The most important are listed below. For the full list, run `python3 main.py --help`.

**Mode and baseline selection:**

| Argument | Default | Description |
|---|---|---|
| `--baseline-mode` | `full` | `full` (SHARE) / `monitor_only` (fixed drop) / `delay_linear` (linear latency drop) / `same_qoe` (equal QoE for all agents) |
| `--fixed-drop-rate` | `0.5` | Drop rate used when `--baseline-mode monitor_only` |
| `--delay-threshold` | `20.0` | Latency threshold (ms) below which no frames are dropped in `delay_linear` mode |
| `--max-delay` | `1000.0` | Latency (ms) at which drop rate reaches 1.0 in `delay_linear` mode |

**Network:**

| Argument | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Listening address |
| `--human1-eval-port` | `9997` | TCP port for Human Agent 1 evaluation |
| `--human2-eval-port` | `9998` | TCP port for Human Agent 2 evaluation |
| `--robot-eval-port` | `9999` | TCP port for Robot evaluation |

**Clock correction:**

| Argument | Default | Description |
|---|---|---|
| `--clock-offset-ms` | `0.0` | Constant NTP clock correction (ms) subtracted from raw RTT. Measure via NTP analysis between agent and server clocks, then pass the measured value here. |

**Merge approval:**

| Argument | Default | Description |
|---|---|---|
| `--alpha-m` | `0.5` | α_m in P_skip = max(0, 1 − α_m · V_new / V_bar). Higher value → more merges approved. |

**QoE targets (per agent):**

| Argument | Default | Description |
|---|---|---|
| `--qoe-target-human1` | `0.9` | QoE target for Human Agent 1 |
| `--qoe-target-human2` | `0.9` | QoE target for Human Agent 2 |
| `--qoe-target-robot` | `0.9` | QoE target for Robot |

**PID gains (per agent, paper §4.3 grid-searched values: Kp=0.8, Ki=0.15, Kd=0.2):**

| Argument | Default | Description |
|---|---|---|
| `--pid-kp-human1` | `0.8` | Proportional gain for Human Agent 1 |
| `--pid-ki-human1` | `0.15` | Integral gain for Human Agent 1 |
| `--pid-kd-human1` | `0.2` | Derivative gain for Human Agent 1 |
| `--pid-kp-human2` | `0.8` | Proportional gain for Human Agent 2 |
| `--pid-ki-human2` | `0.15` | Integral gain for Human Agent 2 |
| `--pid-kd-human2` | `0.2` | Derivative gain for Human Agent 2 |
| `--pid-kp-robot` | `0.8` | Proportional gain for Robot |
| `--pid-ki-robot` | `0.15` | Integral gain for Robot |
| `--pid-kd-robot` | `0.2` | Derivative gain for Robot |

**Frustum overlap parameters:**

| Argument | Default | Description |
|---|---|---|
| `--camera-hfov` | `90.0` | Camera horizontal field of view (degrees) |
| `--camera-vfov` | `60.0` | Camera vertical field of view (degrees) |
| `--camera-near` | `0.1` | Near clipping plane (metres) |
| `--camera-far` | `10.0` | Far clipping plane (metres) |
| `--overlap-resolution` | `0.2` | 3-D grid sampling step δ (metres, paper §4.4.1) |

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
| 6161 | TCP | Agent → server | Robot pose stream |
| 6162 | TCP | Agent → server | Human Agent 2 pose stream |
| 6163 | TCP | Agent → server | Human Agent 1 pose stream |
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

All arguments and their defaults:

| Argument | Default | Description |
|---|---|---|
| `--sync-port` | `6666` | UDP port for device registration |
| `--device-port` | `8888` | UDP port on each device that receives sync packets |
| `--vicon-ctrl-port` | `10000` | UDP port ViconLogger listens on |
| `--sensor-port` | `11111` | UDP port SensorCollector listens on (if present) |
| `--pose-cache-timeout` | `5.0` | Seconds before a cached agent pose is considered stale |

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

**Module:** `unity_emulator/`  
**Runs on:** Unity workstation

> **Virtual scene not included:** The virtual environments (warehouse and fire station scenes) used for dataset synthesis constitute a Unity project exceeding 10 GB and cannot be conveniently distributed in this repository. The scripts in `unity_emulator/` are provided as-is and require the scene project to be built or imported separately before they can be used.

The `unity_emulator/` directory contains three C# scripts with distinct roles:

| File | Role |
|---|---|
| `controller_v3.cs` | Production trajectory replay controller for dataset generation |
| `controller_v2.cs` | Earlier per-agent trajectory controller with interactive FPS mode |
| `frame_extractor.cs` | Scene verification utility for testing render quality during setup |

### Requirements

- **Unity 2022.3.46 LTS**
- **High Definition Render Pipeline (HDRP)** — install via the Unity Package Manager

### controller_v3.cs — Production Trajectory Replay

This is the primary script used for dataset generation. It reads recorded field-study trajectories and drives the camera along them at exactly 30 fps, capturing one PNG frame per interpolation step. It is described in full in the [Running the Full System](#running-the-full-system) section.

**Setup:**

1. Create a new Unity HDRP project and import the virtual environment scenes.

2. For each agent (Human Agent 1, Human Agent 2, Robot), create a GameObject with:
   - A `Camera` component
   - `CameraPoseController.cs` (`controller_v3.cs`) attached

3. Assign the Vicon pose `.txt` file as the **Pose Data File** (`TextAsset`) in the Inspector.

   **Pose file format** — space-separated, one pose per line:
   ```
   timestamp  qx  qy  qz  qw  px  py  pz
   ```
   The source coordinate frame is ROS/ENU right-handed (X-forward, Y-left, Z-up). The script converts to Unity's left-handed frame automatically. If your source data uses a different convention, adjust the two lines marked `COORDINATE CONVERSION` in `ParsePoseData()`.

4. Leave **Save Directory** empty — it auto-names from the GameObject name, ensuring each agent's frames go to a separate folder (e.g. `HumanAgent1/`, `Robot/`).

5. Set **Target Frame Rate** to `30`.

6. Ensure **Use Relative Positioning** is **unchecked** (false). This flag is only for live AR deployment where cameras start from different Unity scene positions; for dataset generation the camera must follow the exact recorded trajectory.

7. Press Play. Frames are written to `Application.persistentDataPath/<GameObjectName>/{timestamp:F6}.png`. The six-decimal timestamp filename is required for `pyslam_integration.py`'s `process_sequence()` to correctly pair frames with ground-truth poses.

**Keyboard controls in Play mode:**

| Key | Action |
|---|---|
| `P` | Play / Pause |
| `R` | Restart from beginning |
| `O` | Recalculate relative pose offset (only relevant when `useRelativePositioning` is true) |
| `T` | Toggle relative positioning mode |

### controller_v2.cs — Per-Agent Interactive and Replay Controller

This is an earlier agent-specific controller for the human agent. It is kept in the repository as a reference implementation. It operates in two modes, selected by the **Use Controlled Translations** (`useControlledTranslations`) toggle in the Inspector:

**Interactive FPS mode** (`useControlledTranslations = false`):  
Standard first-person controller driven by WASD keys and mouse look. Includes sprint (`Left Shift`), jump (`Space`), and a head-bob effect. This mode is used to manually navigate the virtual scene for scene verification and framing checks.

**Trajectory replay mode** (`useControlledTranslations = true`):  
Reads a `.txt` TextAsset assigned in the Inspector, parses it as a sequence of `(timestamp, quaternion, position)` entries, and drives the camera along the trajectory using Slerp/Lerp interpolation timed to the recorded timestamps. WASD and mouse are disabled in this mode. Frames are saved to `pathToSaveData/frames/` numbered by frame index (e.g. `0.png`, `1.png`, ...).

This script also applies per-agent layer culling: the human camera's culling mask excludes the `Human2` and `Robot` layers so each agent camera renders only its own view without seeing the other agents' scene objects.

Inspector fields specific to trajectory replay mode:

| Field | Description |
|---|---|
| `txtFile` | TextAsset containing the recorded pose sequence |
| `pathToSaveData` | Absolute directory path where `frames/` will be created |
| `saveCameraFrames` | Enable/disable frame saving during replay |
| `startingCoordinateOffset` | World-space offset applied to all positions |
| `useHighRes` | Enable high-resolution capture via RenderTexture |
| `captureResMultiplier` | Resolution multiplier when `useHighRes` is true |
| `useCustomRes` | Use fixed custom resolution instead of a multiplier |
| `customCaptureWidth / Height` | Custom resolution dimensions |

> **Note:** `controller_v2.cs` uses `float` for timestamps, which limits precision for large Unix timestamps. For production dataset generation use `controller_v3.cs`, which parses timestamps as `double`. Frame filenames in v2 are sequential integers (`0.png`, `1.png`, ...) rather than the floating-point timestamps required by `pyslam_integration.py`.

### frame_extractor.cs — Scene Verification Utility

This is a lightweight utility for verifying scene render quality during initial setup. It is **not** used for trajectory replay or dataset generation. Attach it to a camera in the scene and use it to check that HDRP lighting, asset placement, and post-processing look correct before running the full trajectory capture.

**How it works:** The script detects camera movement by comparing position and rotation each frame. When the camera has been stationary for `movementStopDelay` seconds (default 0.5 s), it automatically captures a burst of `maxFramesToCapture` frames (default 3) and saves them to `Application.dataPath/<saveDirectory>/frame_XXXX.png`.

Inspector fields:

| Field | Default | Description |
|---|---|---|
| `saveDirectory` | `CapturedFrames` | Subdirectory name inside `Application.dataPath` |
| `useHighResolution` | false | Use RenderTexture-based high-resolution capture |
| `resolutionMultiplier` | 2 | Resolution multiplier when `useHighResolution` is true |
| `maxFramesToCapture` | 3 | Number of frames to capture per stationary burst |
| `movementStopDelay` | 0.5 s | Seconds of stillness before a burst is triggered |
| `mouseSensitivity` | 2 | Mouse look sensitivity |
| `maxLookAngle` | 90° | Maximum vertical look angle |

Use mouse look to navigate the scene and pause at any viewpoint. The script will automatically capture frames after 0.5 s of stillness. Press `Escape` to toggle cursor lock.

---

## Running the Full System

### Network Setup

Assign static IPs to all machines on the same local network. Replace `<server-ip>` throughout with your actual address. On every agent machine, set in `covins_comm/config/config_comm.yaml`:

```yaml
sys.server_ip: "<server-ip>"
```

Ensure `placerec.type: COVINS_G` is set in `~/covins_ws/src/covins/covins_backend/config/config_backend.yaml` on the edge server (see [Configuring the COVINS-G Backend Type](#configuring-the-covins-g-backend-type)).

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
python3 main.py --baseline-mode full --clock-offset-ms <measured_ntp_offset>
```

### Step 4 — Human Agent 1: DNN Evaluator

SSH into the first human agent device. Navigate to the COVINS frontend source directory (`~/covins_ws/src/covins/covins_frontend/src/`) and run the inference client:

```bash
python3 python_evaluator_ondevice.py \
  --model best_model.pth \
  --monitor-port 9997 \
  --monitor-host <server-ip>
```

### Step 5 — Human Agent 1: COVINS-G Frontend

In a second terminal on the same device with the workspace sourced, launch your sensor driver. For example, with an Intel RealSense camera:

```bash
source ~/covins_ws/devel/setup.bash
roslaunch realsense_ros <your_sensor_launch_file>.launch
```

Replace `<your_sensor_launch_file>` with the launch file appropriate for your sensor and calibration. The COVINS-G frontend will connect to the backend automatically using the server IP set in `config_comm.yaml`.

### Step 6 — Human Agent 2: DNN Evaluator

SSH into the second human agent device and run:

```bash
python3 python_evaluator_ondevice.py \
  --model best_model.pth \
  --monitor-port 9998 \
  --monitor-host <server-ip>
```

### Step 7 — Human Agent 2: COVINS-G Frontend

```bash
source ~/covins_ws/devel/setup.bash
roslaunch realsense_ros <your_sensor_launch_file>.launch
```

### Step 8 — Robot Agent: DNN Evaluator

SSH into the robot and open the LXC Ubuntu 20 container. Inside the container, navigate to `~/covins_ws/src/covins/covins_frontend/src/` and run:

```bash
python3 python_evaluator_ondevice.py \
  --model best_model.pth \
  --monitor-port 9999 \
  --socket /tmp/eval.sock \
  --monitor-host <server-ip>
```

The `--socket` flag routes evaluation output through a Unix domain socket, required when the robot's SLAM process and evaluator share a container environment where the SLAM process cannot reach the monitoring server's TCP port directly.

### Step 9 — Robot Agent: COVINS-G Frontend

In a second terminal in the same container with the workspace sourced:

```bash
source ~/covins_ws/devel/setup.bash
roslaunch realsense_ros <your_sensor_launch_file>.launch
```

### Step 10 — Verify and Monitor

Once all frontends are running and submitting keyframes, the COVINS-G backend will begin collaborative mapping. The monitoring server logs QoE values, frame drop rates, and merge decisions to CSV files in its working directory. COVINS-G's Pangolin visualiser opens automatically on the server when `covins_backend_node` starts.

---

## Credits

SHARE builds on the following open-source projects:

**COVINS-G** — Generic back-end for collaborative visual-inertial SLAM.  
Patel, M., Karrer, M., Bänninger, P., and Chli, M. *COVINS-G: A Generic Back-end for Collaborative Visual-Inertial SLAM.* ICRA 2023.  
Repository: [https://github.com/VIS4ROB-lab/covins](https://github.com/VIS4ROB-lab/covins)  
The SHARE patch (`covins_backend/map_be.cpp`) modifies COVINS-G source code and is therefore released under the same GNU General Public License v3.0.

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
