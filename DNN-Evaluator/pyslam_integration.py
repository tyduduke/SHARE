#!/usr/bin/env python3

import os
import sys
import time
import logging
import numpy as np
import cv2
from scipy.stats import entropy
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# pyslam path resolution
# ------------------------------------------------------------------
PYSLAM_PATH = os.environ.get('PYSLAM_PATH', '')
if not PYSLAM_PATH or not os.path.isdir(PYSLAM_PATH):
    raise EnvironmentError(
        "pyslam installation not found. "
        "Set the PYSLAM_PATH environment variable to its location, e.g.:\n"
        "    export PYSLAM_PATH=/path/to/pyslam\n"
        "See https://github.com/luigifreda/pyslam for installation instructions.")
sys.path.append(PYSLAM_PATH)

from config import Config
from slam import Slam
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes
from camera import PinholeCamera
from dataset_types import SensorType, DatasetEnvironmentType
from loop_detector_configs import LoopDetectorConfigs


# ===========================================================================
# SLAM Feature Extractor
# ===========================================================================

class PySLAMFeatureExtractor:
    """
    Wraps pyslam to extract per-frame features for DNN training.
    """

    def __init__(self, config_file: str, kc: int = 16, fe: float = 2.0):
        """
        Parameters
        ----------
        config_file : str
            Path to pyslam config YAML (e.g. config/orb_tum.yaml).
        kc : int
            PSD compression ratio (patch side length in pixels).
        fe : float
            PSD frustum expansion factor.
        """
        self.kc = kc
        self.fe = fe

        self.config = Config(config_file)
        self.camera = PinholeCamera(self.config)

        feature_tracker_config = {
            'num_features':    self.config.num_features_to_extract,
            'detector_type':   FeatureDetectorTypes.ORB,
            'descriptor_type': FeatureDescriptorTypes.ORB,
            'tracker_type':    FeatureTrackerTypes.DES_BF,
            'match_ratio_test': 0.8,
        }
        loop_detector_config = LoopDetectorConfigs.DBOW3

        self.slam = Slam(
            camera=self.camera,
            feature_tracker_config=feature_tracker_config,
            loop_detector_config=loop_detector_config,
            sensor_type=SensorType.MONOCULAR,
            environment_type=DatasetEnvironmentType.INDOOR,
        )
        self.feature_tracker = feature_tracker_factory(**feature_tracker_config)

        self.tracking_state   = 'NOT_INITIALIZED'
        self.initialized      = False
        self.keypoints        = []
        self.keypoint_matches = []
        self.map_points       = []
        self.image_shape      = (self.config.height, self.config.width)
        self.current_frame    = None
        self.current_pose     = np.eye(4)
        self.tracking_quality = 0
        self.frame_counter    = 0

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, timestamp: float):
        """
        Track one grayscale frame through pyslam.

        Returns the 4×4 world-from-camera pose matrix if tracking succeeded,
        or None if tracking was lost.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return None

        self.image_shape   = frame.shape[:2]
        self.current_frame = frame

        img_id = self.frame_counter
        self.frame_counter += 1

        try:
            # Frame normalisation was tested here to improve tracking on
            # low-contrast sequences. Disabled after finding it caused
            # pyslam to lose tracks more frequently on high-contrast frames
            # by clipping the ORB response distribution.
            # frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            self.slam.track(frame, None, None, img_id, timestamp)

            if (self.slam.tracking.state.name == 'OK' and
                    self.slam.tracking.pose_is_ok):
                self.tracking_state = 'OK'

                # Extract pose
                if (hasattr(self.slam.tracking, 'cur_R') and
                        self.slam.tracking.cur_R is not None and
                        hasattr(self.slam.tracking, 'cur_t') and
                        self.slam.tracking.cur_t is not None):
                    pose = np.eye(4)
                    pose[:3, :3] = self.slam.tracking.cur_R
                    pose[:3,  3] = self.slam.tracking.cur_t
                    self.current_pose = pose
                elif (hasattr(self.slam.tracking, 'f_cur') and
                      self.slam.tracking.f_cur is not None):
                    try:
                        self.current_pose = np.linalg.inv(
                            self.slam.tracking.f_cur.pose)
                    except Exception:
                        self.current_pose = np.eye(4)
                else:
                    self.current_pose = np.eye(4)

                self._extract_features_from_current_frame()
                self.tracking_quality = self._calculate_tracking_quality()
                return self.current_pose

            else:
                self.tracking_state = 'LOST'
                logger.debug(f"Tracking lost at frame {img_id}")
                return None

        except IndexError as e:
            if "too many indices" in str(e):
                logger.error("Keypoints array dimension issue (known pyslam bug). "
                             "Apply the recommended pyslam patches.")
            else:
                logger.error(f"IndexError at frame {img_id}: {e}")
            self.tracking_state = 'ERROR'
            return None
        except Exception as e:
            logger.error(f"Tracking error at frame {img_id}: {e}")
            self.tracking_state = 'ERROR'
            return None

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _extract_features_from_current_frame(self):
        """Populate keypoints, keypoint_matches, map_points from pyslam state."""
        if (not hasattr(self.slam.tracking, 'f_cur') or
                self.slam.tracking.f_cur is None):
            self.keypoints        = []
            self.keypoint_matches = []
            self.map_points       = []
            return

        f_cur = self.slam.tracking.f_cur
        self.keypoints  = f_cur.kps if hasattr(f_cur, 'kps') else []
        self.map_points = f_cur.points if hasattr(f_cur, 'points') else []

        if (hasattr(self.slam.tracking, 'idxs_cur') and
                hasattr(self.slam.tracking, 'idxs_ref')):
            self.keypoint_matches = list(zip(
                self.slam.tracking.idxs_cur,
                self.slam.tracking.idxs_ref))
        else:
            self.keypoint_matches = []

    def _calculate_tracking_quality(self) -> float:
        """Return a 0-100 quality score based on map points and distribution."""
        if (not hasattr(self.slam.tracking, 'f_cur') or
                self.slam.tracking.f_cur is None):
            return 0.0

        f_cur = self.slam.tracking.f_cur
        active_kps = len(f_cur.kps) if hasattr(f_cur, 'kps') else 0

        valid_mps = 0
        if hasattr(f_cur, 'points') and f_cur.points is not None:
            valid_mps = sum(
                1 for i, mp in enumerate(f_cur.points)
                if mp is not None and
                not (hasattr(f_cur, 'outliers') and f_cur.outliers[i]))

        if active_kps == 0:
            return 0.0

        mp_ratio = valid_mps / active_kps

        uncertainty = 50.0
        if (hasattr(self.slam.tracking, 'mean_pose_opt_chi2_error') and
                self.slam.tracking.mean_pose_opt_chi2_error is not None):
            uncertainty = max(0.0, 100.0 - min(100.0,
                self.slam.tracking.mean_pose_opt_chi2_error * 20.0))

        quality = (30 * min(1.0, valid_mps / 100) +
                   30 * mp_ratio +
                   20 * self._spatial_distribution_score() +
                   20 * (uncertainty / 100))
        return float(min(100.0, max(0.0, quality)))

    def _spatial_distribution_score(self) -> float:
        """Fraction of 3×3 image grid cells that contain at least one map point."""
        if self.map_points is None or len(self.map_points) < 5:
            return 0.0

        pts_2d = []
        for mp in self.map_points:
            if mp is None:
                continue
            try:
                pos_3d   = mp.position_3d
                pos_2d   = self.camera.project_point_to_image(pos_3d)
                pts_2d.append(pos_2d)
            except Exception:
                continue

        if not pts_2d:
            return 0.0

        h, w     = self.image_shape
        gh, gw   = h // 3, w // 3
        occupied = set()
        for x, y in pts_2d:
            gx = min(2, max(0, int(x / gw)))
            gy = min(2, max(0, int(y / gh)))
            occupied.add((gx, gy))
        return len(occupied) / 9.0

    # ------------------------------------------------------------------
    # PSD matrix  (paper §4.2.1 "Point Spatial Distribution")
    # ------------------------------------------------------------------

    def generate_psd_matrix(self) -> np.ndarray:
        """
        Return a (H_exp, W_exp, 3) PSD matrix:
          Channel 0 — keypoint response strength
          Channel 1 — inlier map-point indicator
          Channel 2 — map-point depth
        """
        h, w         = self.image_shape
        comp_h       = h // self.kc
        comp_w       = w // self.kc
        exp_h        = int(comp_h * self.fe)
        exp_w        = int(comp_w * self.fe)

        kp_mat    = np.zeros((comp_h, comp_w), dtype=np.float32)
        match_mat = np.zeros((comp_h, comp_w), dtype=np.float32)
        depth_mat = np.zeros((exp_h,  exp_w),  dtype=np.float32)

        for kp in self.keypoints:
            if kp is None:
                continue
            try:
                x, y     = kp.pt
                response = kp.response
            except AttributeError:
                continue
            if not (0 <= x < w and 0 <= y < h):
                continue
            px, py = int(x // self.kc), int(y // self.kc)
            if px < comp_w and py < comp_h:
                kp_mat[py, px] = max(kp_mat[py, px], response)

        for match in self.keypoint_matches:
            if match is None:
                continue
            try:
                idx_cur = match[0]
                if idx_cur >= len(self.keypoints) or self.keypoints[idx_cur] is None:
                    continue
                x, y = self.keypoints[idx_cur].pt
            except Exception:
                continue
            if not (0 <= x < w and 0 <= y < h):
                continue
            px, py = int(x // self.kc), int(y // self.kc)
            if px < comp_w and py < comp_h:
                match_mat[py, px] = 255.0

        for mp in self.map_points:
            if mp is None:
                continue
            try:
                pos_3d = mp.position_3d
                pos_2d = self.camera.project_point_to_image(pos_3d)
                x, y   = pos_2d
                depth  = pos_3d[2]
            except Exception:
                continue
            ex = x * exp_w / w
            ey = y * exp_h / h
            if 0 <= ex < exp_w and 0 <= ey < exp_h:
                depth_mat[int(ey), int(ex)] = depth

        # Resize compressed channels to the expanded size
        if (exp_h, exp_w) != (comp_h, comp_w):
            kp_mat    = cv2.resize(kp_mat,    (exp_w, exp_h))
            match_mat = cv2.resize(match_mat, (exp_w, exp_h))

        return np.stack([kp_mat, match_mat, depth_mat], axis=2)   # (H, W, 3)

    # ------------------------------------------------------------------
    # Scalar feature extraction  (8 features for the scalar branch)
    # ------------------------------------------------------------------

    def analyze_image(self, image_path: str = None) -> dict:
        """
        Compute 8 scalar image/tracking features:
          brightness, contrast, entropy, laplacian_var,
          tracking_quality, num_keypoints, num_matches, num_map_points
        """
        img = self.current_frame if image_path is None else cv2.imread(image_path)
        if img is None:
            raise ValueError("No image available for analysis")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        norm = img / 255.0
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_n = hist.flatten() / hist.sum()
        hist_n = hist_n[hist_n > 0]

        return {
            'brightness':      float(np.mean(norm)),
            'contrast':        float(np.std(norm)),
            'entropy':         float(entropy(hist_n)),
            'laplacian':       float(np.var(cv2.Laplacian(img, cv2.CV_64F))),
            'tracking_quality': float(self.tracking_quality),
            'num_keypoints':   float(len(self.keypoints)),
            'num_matches':     float(len(self.keypoint_matches)),
            'num_map_points':  float(sum(1 for mp in self.map_points
                                         if mp is not None)),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def get_current_pose(self) -> np.ndarray:
        return self.current_pose

    def get_tracking_state(self) -> str:
        return self.tracking_state

    def reset(self):
        self.slam.reset()
        self.tracking_state   = 'NOT_INITIALIZED'
        self.initialized      = False
        self.keypoints        = []
        self.keypoint_matches = []
        self.map_points       = []
        self.current_pose     = np.eye(4)
        self.tracking_quality = 0

    def shutdown(self):
        self.slam.quit()


# ===========================================================================
# SLAM error computation
# ===========================================================================

def compute_slam_error(estimated_pose: np.ndarray,
                       groundtruth_pose: np.ndarray) -> float:
    """
    Compute a combined position + orientation error between two 4×4 poses.

    position_error  = ||t_est - t_gt||_2            (metres)
    rotation_error  = arccos((tr(R_err) - 1) / 2)  (degrees)
    total_error     = position_error + 0.1 * rotation_error

    The 0.1 weight is an application-specific tuning parameter that
    keeps the rotation contribution commensurate with the position term
    (1 cm ≈ 1° error on typical SLAM trajectories).
    """
    p_est = estimated_pose[:3, 3]
    p_gt  = groundtruth_pose[:3, 3]
    pos_err = np.linalg.norm(p_est - p_gt)

    R_err     = groundtruth_pose[:3, :3].T @ estimated_pose[:3, :3]
    trace_val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    rot_err   = np.degrees(np.arccos(trace_val))

    return float(pos_err + 0.1 * rot_err)


# ===========================================================================
# Dataset processor
# ===========================================================================

class CustomPySLAMProcessor:
    """
    Processes the SenseTime Visual SLAM dataset through pyslam and
    produces (poses, psd_matrices, image_features, gt_poses, errors)
    tuples per sequence for cache generation.
    """

    def __init__(self, data_root: str, config_file: str,
                 sequences=None, scenes=None):
        self.data_root   = data_root
        self.config_file = config_file
        self.gt_dir      = os.path.join(data_root, "Groundtruth")

        self.scenes = (scenes if scenes is not None
                       else [f'Scene{i}' for i in range(1, 9)])

        all_seq = [f'A{i}' for i in range(8)] + [f'B{i}' for i in range(8)]
        if sequences is not None:
            self.sequences = sequences
        elif self.scenes:
            first = os.path.join(data_root, "InputData", self.scenes[0])
            self.sequences = [s for s in all_seq
                              if os.path.exists(os.path.join(first, s))]
        else:
            self.sequences = []

        self.slam_extractor = None
        self.image_paths    = []

    def process_sequence(self, scene: str, sequence: str,
                         start_frame: int = 0, max_frames: int = None):
        """
        Process one scene/sequence and return extracted data tuple.

        Returns
        -------
        (poses, psd_matrices, image_features, ground_truth_poses, errors)
        or None on failure.
        """
        img_dir = os.path.join(self.data_root, "InputData", scene,
                               sequence, "mav0", "cam0", "data")
        if not os.path.exists(img_dir):
            logger.error(f"Image directory not found: {img_dir}")
            return None

        gt_file = os.path.join(self.gt_dir, sequence, "groundtruth", "data.csv")
        if not os.path.exists(gt_file):
            logger.error(f"Ground truth file not found: {gt_file}")
            return None

        gt_data = self._load_ground_truth(gt_file)
        if gt_data is None:
            return None

        image_files = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
        self.image_paths = [os.path.join(img_dir, f) for f in image_files]

        self.slam_extractor = PySLAMFeatureExtractor(self.config_file)

        poses, psds, feats, gt_poses, errors = [], [], [], [], []
        end_frame = (min(start_frame + max_frames, len(image_files))
                     if max_frames else len(image_files))

        # Feed the first two frames to initialise the SLAM system
        for i in range(min(2, end_frame)):
            img   = cv2.normalize(
                cv2.imread(os.path.join(img_dir, image_files[i]),
                           cv2.IMREAD_GRAYSCALE),
                None, 0, 255, cv2.NORM_MINMAX)
            ts    = float(os.path.splitext(image_files[i])[0])
            iid   = self.slam_extractor.frame_counter
            self.slam_extractor.frame_counter += 1
            self.slam_extractor.slam.track(img, None, None, iid, ts)

        start_frame = max(start_frame, 2)

        for i in range(start_frame, end_frame):
            img_path = os.path.join(img_dir, image_files[i])
            ts_str   = os.path.splitext(image_files[i])[0]
            try:
                ts = float(ts_str)
            except ValueError:
                ts_str = ''.join(c for c in ts_str if c.isdigit() or c == '.')
                ts = float(ts_str)

            gt_pose = gt_data.get(ts)
            if gt_pose is None:
                closest = min(gt_data.keys(), key=lambda x: abs(x - ts))
                gt_pose = gt_data[closest]

            frame    = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            est_pose = self.slam_extractor.process_frame(frame, ts)

            if est_pose is not None:
                psd     = self.slam_extractor.generate_psd_matrix()
                feat    = self.slam_extractor.analyze_image()
                error   = compute_slam_error(est_pose, gt_pose)
                poses.append(est_pose)
                psds.append(psd)
                feats.append(feat)
                gt_poses.append(gt_pose)
                errors.append(error)
                logger.info(f"Frame {i}/{end_frame-1}  error={error:.4f} m")
            else:
                logger.warning(f"Tracking lost at frame {i}")

        self.slam_extractor.shutdown()
        self.slam_extractor = None
        return poses, psds, feats, gt_poses, errors

    def process_all_sequences(self, max_frames_per_sequence=None) -> dict:
        results = {}
        for scene in self.scenes:
            results[scene] = {}
            for seq in self.sequences:
                logger.info(f"Processing {scene}/{seq}")
                r = self.process_sequence(scene, seq,
                                          max_frames=max_frames_per_sequence)
                if r is not None:
                    results[scene][seq] = r
        return results

    def _load_ground_truth(self, gt_file: str):
        """
        Load ground truth poses from a comma-separated CSV file.

        Expected format per line (no header, or header lines starting with #):
            timestamp, qx, qy, qz, qw, px, py, pz
        """
        try:
            gt_data = {}
            with open(gt_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    vals = line.strip().split(',')
                    if len(vals) < 8:
                        continue
                    try:
                        ts  = float(vals[0])
                        qx, qy, qz, qw = map(float, vals[1:5])
                        px, py, pz     = map(float, vals[5:8])
                        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
                        pose = np.eye(4)
                        pose[:3, :3] = rot
                        pose[:3,  3] = [px, py, pz]
                        gt_data[ts] = pose
                    except ValueError as e:
                        logger.warning(f"GT parse error: {line.strip()} — {e}")
            logger.info(f"Loaded {len(gt_data)} GT poses from {gt_file}")
            return gt_data
        except Exception as e:
            logger.error(f"Failed to load GT file {gt_file}: {e}")
            return None
