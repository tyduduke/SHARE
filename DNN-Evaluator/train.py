#!/usr/bin/env python3
"""
SHARE DNN Training Script (with on-the-fly pyslam feature extraction)

Extracts features from raw dataset images using pyslam, caches them as
.npz files, then trains the SLAMErrorPredictor.

Usage
-----
# Recommended: pre-generate cache with pyslam_integration.py, then use
# train_cached.py (faster, no pyslam dependency at training time).

# On-the-fly extraction (requires pyslam installation):
python train.py \
  --data_root   /path/to/dataset \
  --pyslam_config config/orb_tum.yaml \
  --sequences   A0 A1 A2 A3 A4 A5 A6 A7 \
  --scenes      Scene1 Scene2 Scene3 Scene4 Scene5 Scene6 \
  --cache_dir   cached_features \
  --epochs      20 \
  --output_dir  results
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Shared model and dataset
from model               import SLAMErrorPredictor
from dataloader          import CustomSLAMDataset
from pyslam_integration  import CustomPySLAMProcessor


# ===========================================================================
# Dataset preparation
# ===========================================================================

def prepare_dataset(data_root, pyslam_config, sequences=None, scenes=None,
                    use_cached=False, cache_dir='cached_features'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if use_cached and cache_dir and os.path.exists(cache_dir):
        print("Loading from cached features …")
        return CustomSLAMDataset(data_root=data_root, transform=transform,
                                 sequences=sequences, scenes=scenes,
                                 use_cached=True, cache_dir=cache_dir)

    # Extract features on-the-fly and write to cache
    print("Extracting features via pyslam …")
    processor = CustomPySLAMProcessor(
        data_root=data_root, config_file=pyslam_config,
        sequences=sequences, scenes=scenes)
    os.makedirs(cache_dir, exist_ok=True)

    for scene in processor.scenes:
        for seq in processor.sequences:
            print(f"  Processing {scene}/{seq} …")
            result = processor.process_sequence(scene, seq)
            if result is None:
                print(f"  Skipping {scene}/{seq} — could not process")
                continue

            poses, psds, img_features, gt_poses, errors = result
            for i, (pose, psd, feats, error) in enumerate(
                    zip(poses, psds, img_features, errors)):
                ts = float(os.path.splitext(
                    os.path.basename(processor.image_paths[i]))[0])
                scalar = np.array([
                    feats['brightness'], feats['contrast'],
                    feats['entropy'],    feats['laplacian'],
                    feats['tracking_quality'],
                    feats['num_keypoints'],
                    feats['num_matches'],
                    feats['num_map_points'],
                ], dtype=np.float32)
                np.savez(os.path.join(cache_dir,
                                      f"{scene}_{seq}_{int(ts)}.npz"),
                         pose=pose, psd=psd, features=scalar,
                         gt_pose=gt_poses[i], error=error)

    return CustomSLAMDataset(data_root=data_root, transform=transform,
                             sequences=sequences, scenes=scenes,
                             use_cached=True, cache_dir=cache_dir)


# ===========================================================================
# Training / evaluation  (same logic as train_cached.py)
# ===========================================================================

def train_model(model, train_loader, val_loader, num_epochs, lr,
                device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    best_path  = os.path.join(output_dir, 'best_model.pth')
    criterion  = nn.MSELoss()
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    device     = torch.device(device if torch.cuda.is_available() else 'cpu')
    model      = model.to(device)
    best_loss  = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train(); running = 0.0
        for imgs, psds, scalars, errors in tqdm(
                train_loader, desc=f"Epoch {epoch+1} [train]"):
            imgs    = imgs.to(device); psds    = psds.to(device)
            scalars = scalars.to(device)
            errors  = errors.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs, psds, scalars), errors)
            loss.backward(); optimizer.step()
            running += loss.item() * imgs.size(0)

        train_loss = running / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval(); running = 0.0
        with torch.no_grad():
            for imgs, psds, scalars, errors in tqdm(
                    val_loader, desc=f"Epoch {epoch+1} [val]"):
                imgs    = imgs.to(device); psds    = psds.to(device)
                scalars = scalars.to(device)
                errors  = errors.to(device).float().unsqueeze(1)
                running += criterion(model(imgs, psds, scalars),
                                     errors).item() * imgs.size(0)

        val_loss = running / len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved → {best_path}")
        print('-' * 50)

    return model, train_losses, val_losses, best_path


def evaluate_model(model, test_loader, device):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, psds, scalars, errors in tqdm(test_loader, desc="Test"):
            out = model(imgs.to(device), psds.to(device), scalars.to(device))
            preds.extend(out.cpu().numpy().flatten())
            gts.extend(errors.numpy())
    p, g = np.array(preds), np.array(gts)
    mse  = np.mean((p - g) ** 2)
    return mse, np.sqrt(mse), np.mean(np.abs(p - g)), p, g


def plot_results(train_losses, val_losses, preds, gts, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train'); plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_curves.png')); plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(gts, preds, alpha=0.5)
    hi = max(gts.max(), preds.max())
    plt.plot([0, hi], [0, hi], 'r--')
    plt.xlabel('Ground Truth (cm)'); plt.ylabel('Predicted (cm)')
    plt.savefig(os.path.join(output_dir, 'predictions_vs_gt.png')); plt.close()


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Train SLAM error predictor')
    parser.add_argument('--data_root',     required=True)
    parser.add_argument('--pyslam_config', required=True)
    parser.add_argument('--sequences',     nargs='+')
    parser.add_argument('--scenes',        nargs='+')
    parser.add_argument('--cache_dir',     default='cached_features')
    parser.add_argument('--use_cached',    action='store_true')
    parser.add_argument('--epochs',        type=int,   default=20)
    parser.add_argument('--device',        default='cuda')
    parser.add_argument('--output_dir',    default='results')
    args = parser.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = prepare_dataset(
        args.data_root, args.pyslam_config,
        sequences=args.sequences, scenes=args.scenes,
        use_cached=args.use_cached, cache_dir=args.cache_dir)

    if len(dataset) == 0:
        raise ValueError("No data found.")

    train_idx, temp_idx = train_test_split(range(len(dataset)),
                                           test_size=0.3, random_state=42)
    val_idx, test_idx   = train_test_split(temp_idx, test_size=0.5, random_state=42)

    kw = dict(batch_size=16, num_workers=4)
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True,  **kw)
    val_loader   = DataLoader(Subset(dataset, val_idx),   shuffle=False, **kw)
    test_loader  = DataLoader(Subset(dataset, test_idx),  shuffle=False, **kw)

    model = SLAMErrorPredictor(num_scalar_features=8)
    model, train_losses, val_losses, best_path = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs, lr=0.001,
        device=args.device, output_dir=args.output_dir)

    model.load_state_dict(torch.load(best_path, map_location=device))
    mse, rmse, mae, preds, gts = evaluate_model(model, test_loader, args.device)
    print(f"\nRMSE: {rmse:.4f} cm  MAE: {mae:.4f} cm")

    plot_results(train_losses, val_losses, preds, gts, args.output_dir)
    print(f"Done. Results → {args.output_dir}/")


if __name__ == '__main__':
    main()
