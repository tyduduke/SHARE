#!/usr/bin/env python3
"""
Dataset loader for SLAM error prediction training.

Loads pre-cached .npz feature files produced by pyslam_integration.py.
Each .npz file contains:
    psd      -- Point Spatial Distribution matrix  (H, W, 3)  numpy float32
    features -- 8 scalar features                  (8,)        numpy float32
    error    -- SLAM error label                   scalar      numpy float32

Raw data loading (without cache) is not implemented here; run
pyslam_integration.py first to generate the cache directory.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image


class CustomSLAMDataset(Dataset):
    """
    Dataset for SLAM error prediction backed by cached .npz feature files.
    """

    def __init__(self, data_root: str, transform=None,
                 sequences=None, scenes=None,
                 use_cached: bool = True, cache_dir: str = None):
        """
        Parameters
        ----------
        data_root : str
            Root of the dataset (parent of InputData/ and Groundtruth/).
        transform : callable or None
            Torchvision transform applied to the RGB keyframe image.
        sequences : list[str] or None
            Sequence names, e.g. ['A0', 'A1'].  None → all A0-A7 + B0-B7.
        scenes : list[str] or None
            Scene names, e.g. ['Scene1'].  None → Scene1-Scene8.
        use_cached : bool
            Must be True.  Raw loading is not implemented.
        cache_dir : str
            Directory containing .npz cache files.
        """
        self.data_root = data_root
        self.transform = transform

        self.scenes = scenes if scenes is not None else [f'Scene{i}' for i in range(1, 9)]
        self.sequences = (sequences if sequences is not None
                          else [f'A{i}' for i in range(8)] + [f'B{i}' for i in range(8)])

        self.image_paths     = []
        self.psd_matrices    = []
        self.scalar_features = []
        self.slam_errors     = []

        if use_cached and cache_dir:
            self._load_from_cache(cache_dir)
        else:
            raise NotImplementedError(
                "Raw data loading is not implemented. "
                "Run pyslam_integration.py to generate cached features, "
                "then pass use_cached=True and cache_dir=<path>.")

    # ------------------------------------------------------------------
    def _load_from_cache(self, cache_dir: str):
        print(f"Loading dataset from cache: {cache_dir}")
        files = []
        for scene in self.scenes:
            for seq in self.sequences:
                files.extend(glob.glob(
                    os.path.join(cache_dir, f"{scene}_{seq}_*.npz")))
        files.sort()

        if not files:
            print(f"WARNING: No cache files found in {cache_dir}")

        # Earlier version loaded all .npz files into memory upfront to avoid
        # repeated disk reads during training. Abandoned after the full dataset
        # exceeded available RAM; lazy loading per __getitem__ call is slower
        # but keeps the memory footprint bounded regardless of dataset size.
        # all_data = {f: np.load(f, allow_pickle=True) for f in files}

        for fpath in files:
            try:
                basename = os.path.basename(fpath)
                parts    = basename.split('_')
                if len(parts) < 3:
                    print(f"WARNING: Unexpected filename format: {basename}")
                    continue

                scene = parts[0]
                seq   = parts[1]
                ts    = parts[2].split('.')[0]

                if scene not in self.scenes or seq not in self.sequences:
                    continue

                data = np.load(fpath, allow_pickle=True)
                if not all(k in data for k in ('psd', 'features', 'error')):
                    print(f"WARNING: Missing keys in {fpath}")
                    continue

                img_dir  = os.path.join(self.data_root, "InputData", scene,
                                        seq, "mav0", "cam0", "data")
                img_path = os.path.join(img_dir, f"{ts}.png")

                self.image_paths.append(img_path)
                self.psd_matrices.append(data['psd'])          # (H, W, 3)
                self.scalar_features.append(data['features'])   # (8,)
                self.slam_errors.append(data['error'])

            except Exception as e:
                print(f"Error loading {fpath}: {e}")

        print(f"Loaded {len(self.image_paths)} samples")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self._load_image(self.image_paths[idx])

        if self.transform:
            image = self.transform(image)

        # PSD: stored as (H, W, 3), model expects (3, H, W)
        psd = torch.tensor(self.psd_matrices[idx], dtype=torch.float32)
        if psd.dim() == 3 and psd.shape[2] == 3:
            psd = psd.permute(2, 0, 1)     # (H, W, 3) → (3, H, W)

        scalar  = torch.tensor(self.scalar_features[idx], dtype=torch.float32)
        error   = torch.tensor(self.slam_errors[idx],     dtype=torch.float32)

        return image, psd, scalar, error

    # ------------------------------------------------------------------
    def _load_image(self, img_path: str) -> Image.Image:
        """Load image with fallback to nearest-timestamp file."""
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

        # Fallback: find nearest available timestamp in same directory
        dir_path = os.path.dirname(img_path)
        ts_str   = os.path.splitext(os.path.basename(img_path))[0]
        try:
            all_pngs = [f for f in os.listdir(dir_path) if f.endswith('.png')]
            if all_pngs:
                closest = min(all_pngs,
                              key=lambda f: abs(float(os.path.splitext(f)[0]) - float(ts_str)))
                alt_path = os.path.join(dir_path, closest)
                img = cv2.imread(alt_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        except Exception:
            pass

        # Final fallback: black image
        return Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
