# SHARE: Towards Head-Mounted AR with User-Centric SLAM in Shared Human-Robot Workspaces

[![Paper](https://img.shields.io/badge/Paper-ACM%20IMWUT%202026-blue)](https://doi.org/10.1145/3832004) [![arXiv](https://img.shields.io/badge/arXiv-2607.23901-b31b1b.svg)](https://arxiv.org/abs/2607.23901) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Official implementation of SHARE**, a user-centric collaborative SLAM system for shared augmented reality (AR) and robot workspaces.

**Publication:** Tianyuan Du, Tianyi Hu, Hanting Ye, and Maria Gorlatova. *SHARE: Towards Head-Mounted AR with User-Centric SLAM in Shared Human-Robot Workspaces.* Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT), 10(3), 2026.

**[ACM Digital Library](https://doi.org/10.1145/3832004) | [arXiv preprint](https://arxiv.org/abs/2607.23901) | [I3T Lab](https://gorlatova.pratt.duke.edu/)**

## Overview

SHARE is a multi-agent simultaneous localization and mapping (SLAM) system designed to improve the quality of experience (QoE) of heterogeneous agents—head-mounted AR users and mobile robots—operating in a shared physical workspace. Conventional collaborative SLAM systems treat these agents uniformly despite their different latency requirements. SHARE prioritizes the responsiveness of AR interaction while preserving robot tracking accuracy.

SHARE extends [COVINS-G](https://github.com/VIS4ROB-lab/covins) with two edge-coordinated mechanisms:

- **QoE-aware scheduling:** Agent-specific experience models and adaptive transmission scheduling allocate resources according to the requirements of AR users and robots.
- **Selective map merging:** Camera-frustum overlap estimation and merge-approval gating reduce redundant edge computation.

A deep neural network (DNN) predicts per-keyframe SLAM localization error using visual frames, point spatial distribution (PSD) matrices, and tracking-quality features. Offline training labels and features are generated using [pySLAM](https://github.com/luigifreda/pyslam).

In the published evaluation, SHARE achieved **13.22 ms average latency for AR users** (a **43.3% reduction** from the baseline) while maintaining **sub-2-cm tracking accuracy**, with user-study results showing statistically significant improvements in perceived experience. See the [paper](https://doi.org/10.1145/3832004) for experimental details.

## Repository Contents

| Directory | Component | Runs on |
|---|---|---|
| `covins_backend/` | Patched COVINS-G map backend and merge gating | Edge server |
| `monitoring_server/` | QoE monitoring, PID scheduling, and merge control | Edge server |
| `dnn_evaluator/` | SLAM error prediction model, training, and evaluation | Agent / training workstation |
| `vicon_server/` | Ground-truth logging and clock synchronization | Vicon workstation |
| `unity_emulator/` | Synthetic camera-trajectory replay and frame capture | Unity workstation |

## Documentation and Reproduction

The original implementation documentation is preserved below, covering prerequisites, building and patching COVINS-G, training the DNN, deploying the monitoring server, collecting Vicon trajectories, using the Unity emulator, configuring the TurtleBot 4 LXC environment, and running the full multi-agent system.

**Environment:** Ubuntu 20.04 and ROS Noetic for the COVINS-G components; Unity 2022.3.46 LTS for synthetic data generation. The TurtleBot 4 uses an Ubuntu 20.04 LXC container on its Ubuntu 22.04 / ROS 2 Humble host.

**Reproducibility note:** The Unity warehouse and fire-station scene assets are not distributed in this repository. The `unity_emulator/` scripts require separately prepared scenes. Hardware-dependent settings (server addresses, sensor calibration, NTP offset, and Vicon configuration) must be adapted to the deployment.

## Acknowledgments

This work was supported in part by NSF grants CSR-2312760, CNS-2112562, and IIS-2231975, NSF CAREER Award IIS-2046072, NSF NAIAD Award 2332744, a CISCO Research Award, a Meta Research Award, Defense Advanced Research Projects Agency Young Faculty Award HR0011-24-1-0001, and the Army Research Laboratory under Cooperative Agreement Number W911NF-23-2-0224.

The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Defense Advanced Research Projects Agency, the Army Research Laboratory, or the U.S. Government. This paper has been approved for public release; distribution is unlimited. No official endorsement should be inferred. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

The authors gratefully acknowledge the assistance of **Dhun Pandya** from the **North Carolina School of Science and Mathematics** on this work.

SHARE builds on [COVINS-G](https://github.com/VIS4ROB-lab/covins) and [pySLAM](https://github.com/luigifreda/pyslam). We acknowledge the developers and contributors of these open-source projects.

## Contact

For questions about SHARE, its implementation, or research collaboration, please contact:

**Tianyuan Du** (Corresponding Author)  
Department of Electrical and Computer Engineering, Duke University  
Email: [alex.du@duke.edu](mailto:alex.du@duke.edu)

Other authors: Tianyi Hu, Hanting Ye, and Maria Gorlatova.  
Research group: [Intelligent Interactive Internet of Things (I3T) Lab](https://gorlatova.pratt.duke.edu/).

For code-related questions and bug reports, please use the repository's GitHub Issues page.

## Citation

If SHARE is useful in your research, please cite:

```bibtex
@article{du2026share,
  title   = {{SHARE}: Towards Head-Mounted {AR} with User-Centric {SLAM} in Shared Human-Robot Workspaces},
  author  = {Du, Tianyuan and Hu, Tianyi and Ye, Hanting and Gorlatova, Maria},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume  = {10},
  number  = {3},
  year    = {2026},
  doi     = {10.1145/3832004},
  url     = {https://doi.org/10.1145/3832004}
}
```

## License

The project is distributed under the **GNU General Public License v3.0**, consistent with its COVINS-G-derived backend. See the repository's license file for the applicable terms and preserve third-party notices where required.

