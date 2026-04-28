#!/usr/bin/env python3


import numpy as np
import logging
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class VolumeOverlapCalculator:
    """
    Calculates pairwise camera frustum overlap volumes.
    """

    def __init__(self, camera_params=None):
        """
        Parameters
        ----------
        camera_params : dict or None
            Camera frustum parameters.  Keys:
              horizontal_fov_deg  -- horizontal field of view in degrees
              vertical_fov_deg    -- vertical field of view in degrees
              near_distance       -- near clipping plane (metres)
              far_distance        -- far clipping plane (metres)
              sampling_resolution -- 3-D grid step size δ (metres, paper §4.4.1)
        """
        self.camera_params = camera_params or {
            'horizontal_fov_deg': 90.0,
            'vertical_fov_deg':   60.0,
            'near_distance':       0.1,
            'far_distance':       10.0,
            'sampling_resolution': 0.2,
        }

        self.h_fov_rad = np.radians(self.camera_params['horizontal_fov_deg'])
        self.v_fov_rad = np.radians(self.camera_params['vertical_fov_deg'])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_pairwise_overlap(self, pose1: dict, pose2: dict) -> dict:
        """
        Compute the frustum overlap between two agents.

        Parameters
        ----------
        pose1, pose2 : dict
            Keys: x, y, z, qx, qy, qz, qw

        Returns
        -------
        dict
            volume_cam1         -- visible volume of camera 1 (m³)
            volume_cam2         -- visible volume of camera 2 (m³)
            volume_overlap      -- V_overlap = |V_i ∩ V_j| · δ³  (m³)
            volume_union        -- V_union   = |V_i ∪ V_j| · δ³  (m³)
            overlap_ratio_iou   -- V_overlap / V_union  (for logging only)
            overlap_ratio_cam1  -- V_overlap / V_cam1
            overlap_ratio_cam2  -- V_overlap / V_cam2
            total_sample_points
            sample_resolution   -- δ (metres)
        """
        try:
            T1 = self._pose_to_transform(pose1)
            T2 = self._pose_to_transform(pose2)

            min_coords, max_coords = self._bounding_box([pose1, pose2])
            delta = self.camera_params['sampling_resolution']
            points = self._sample_points(min_coords, max_coords, delta)

            in1 = np.array([self._in_frustum(p, T1) for p in points])
            in2 = np.array([self._in_frustum(p, T2) for p in points])

            n1        = int(np.sum(in1))
            n2        = int(np.sum(in2))
            n_overlap = int(np.sum(in1 & in2))
            n_union   = int(np.sum(in1 | in2))

            voxel = delta ** 3
            V1        = n1        * voxel
            V2        = n2        * voxel
            V_overlap = n_overlap * voxel
            V_union   = n_union   * voxel

            return {
                'volume_cam1':        V1,
                'volume_cam2':        V2,
                'volume_overlap':     V_overlap,   # used in P_skip formula
                'volume_union':       V_union,
                'overlap_ratio_iou':  V_overlap / max(V_union,  1e-9),
                'overlap_ratio_cam1': V_overlap / max(V1,        1e-9),
                'overlap_ratio_cam2': V_overlap / max(V2,        1e-9),
                'total_sample_points': len(points),
                'sample_resolution':  delta,
            }

        except Exception as e:
            logger.error(f"Error calculating pairwise overlap: {e}")
            return self._empty_result()

    def calculate_all_overlaps(self, poses: list) -> dict:
        """
        Calculate pairwise frustum overlaps for all agent pairs.

        Parameters
        ----------
        poses : list of dict or None
            One entry per agent; None entries are skipped.

        Returns
        -------
        dict  keyed by 'agent_i_agent_j' strings
        """
        if len(poses) < 2:
            return {}

        overlaps = {}
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                if poses[i] is not None and poses[j] is not None:
                    key = f"agent_{i+1}_agent_{j+1}"
                    overlaps[key] = self.calculate_pairwise_overlap(poses[i], poses[j])

        return overlaps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pose_to_transform(self, pose: dict) -> np.ndarray:
        """Convert pose dict to 4×4 homogeneous transformation matrix."""
        position = np.array([pose['x'], pose['y'], pose['z']])
        quat     = [pose['qx'], pose['qy'], pose['qz'], pose['qw']]
        R        = Rotation.from_quat(quat).as_matrix()
        T        = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = position
        return T

    def _in_frustum(self, point: np.ndarray, T: np.ndarray) -> bool:
        """
        Test whether a 3-D point lies inside a camera frustum (paper Eq. 5).

        Camera is assumed to look along its -Z axis (standard OpenCV / SLAM
        convention).
        """
        p_cam = np.linalg.inv(T) @ np.append(point, 1.0)
        xc, yc, zc = p_cam[:3]

        # Depth must fall within [near, far] along -Z
        if zc > -self.camera_params['near_distance'] or \
           zc < -self.camera_params['far_distance']:
            return False

        dist = abs(zc)
        # Horizontal and vertical half-angle checks (paper Eq. 5)
        if np.arctan2(abs(xc), dist) > self.h_fov_rad / 2:
            return False
        if np.arctan2(abs(yc), dist) > self.v_fov_rad / 2:
            return False

        return True

    def _bounding_box(self, poses: list):
        """Axis-aligned bounding box around all agent positions + far margin."""
        positions = np.array([[p['x'], p['y'], p['z']] for p in poses])
        margin    = self.camera_params['far_distance']
        return positions.min(axis=0) - margin, positions.max(axis=0) + margin

    def _sample_points(self, min_c: np.ndarray, max_c: np.ndarray,
                       delta: float) -> np.ndarray:
        """Generate regular 3-D grid of sample points with spacing δ."""
        xs = np.arange(min_c[0], max_c[0] + delta, delta)
        ys = np.arange(min_c[1], max_c[1] + delta, delta)
        zs = np.arange(min_c[2], max_c[2] + delta, delta)
        xx, yy, zz = np.meshgrid(xs, ys, zs)
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def _empty_result(self) -> dict:
        """Return zeroed result dict for error paths."""
        return {
            'volume_cam1': 0.0, 'volume_cam2': 0.0,
            'volume_overlap': 0.0, 'volume_union': 0.0,
            'overlap_ratio_iou': 0.0,
            'overlap_ratio_cam1': 0.0, 'overlap_ratio_cam2': 0.0,
            'total_sample_points': 0, 'sample_resolution': 0.0,
        }
