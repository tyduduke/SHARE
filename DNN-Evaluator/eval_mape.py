#!/usr/bin/env python3
"""
SHARE DNN Evaluation Script

Loads a trained SLAMErrorPredictor checkpoint and evaluates it with MAPE
and additional metrics on a cached feature dataset.

Usage
-----
python eval_mape.py \
  --model_path results/best_model.pth \
  --data_root  /path/to/dataset \
  --cache_dir  cached_features \
  --output_dir evaluation_results
"""

import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Shared model (must match the architecture used during training)
from model      import SLAMErrorPredictor
from dataloader import CustomSLAMDataset


# ===========================================================================
# Metrics
# ===========================================================================

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray,
                   epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    denom = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse    = float(np.mean((y_true - y_pred) ** 2))
    rmse   = float(np.sqrt(mse))
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    mape   = calculate_mape(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
    metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2,
               'max_error':    float(np.max(np.abs(y_true - y_pred))),
               'median_error': float(np.median(np.abs(y_true - y_pred)))}
    for thr in [0.1, 0.2, 0.5, 1.0]:
        metrics[f'within_{thr}'] = float(
            np.mean(np.abs(y_true - y_pred) <= thr) * 100)
    return metrics


# ===========================================================================
# Inference
# ===========================================================================

def run_inference(model: SLAMErrorPredictor,
                  loader: DataLoader,
                  device) -> tuple:
    """Return (predictions, ground_truths) as numpy arrays."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, psds, scalars, errors in tqdm(loader, desc="Evaluating"):
            imgs    = imgs.to(device)
            psds    = psds.to(device)
            scalars = scalars.to(device)
            out     = model(imgs, psds, scalars)
            preds.extend(out.cpu().numpy().flatten())
            gts.extend(errors.numpy())
    return np.array(preds), np.array(gts)


# ===========================================================================
# Plots
# ===========================================================================

def plot_evaluation_results(predictions, ground_truths, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_palette("husl")

    # Scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(ground_truths, predictions, alpha=0.6, s=50)
    lo = min(ground_truths.min(), predictions.min())
    hi = max(ground_truths.max(), predictions.max())
    plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect')
    plt.xlabel('Ground Truth (cm)'); plt.ylabel('Predicted (cm)')
    plt.title(f"MAPE={metrics['mape']:.2f}%  R²={metrics['r2']:.3f}")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_gt.png'), dpi=300)
    plt.close()

    # Error distribution
    errors = predictions - ground_truths
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Prediction Error (cm)'); axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[1].boxplot(errors)
    axes[1].set_ylabel('Prediction Error (cm)'); axes[1].set_title('Error Box Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300)
    plt.close()


def save_results(metrics, predictions, ground_truths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({
        'ground_truth': ground_truths,
        'prediction':   predictions,
        'abs_error':    np.abs(ground_truths - predictions),
    }).to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate SLAM error predictor')
    parser.add_argument('--model_path', default='results/best_model.pth')
    parser.add_argument('--data_root',  required=True)
    parser.add_argument('--cache_dir',  required=True)
    parser.add_argument('--sequences',  nargs='+')
    parser.add_argument('--scenes',     nargs='+')
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--output_dir', default='evaluation_results')
    parser.add_argument('--test_size',  type=float, default=0.3,
                        help='Must match the value used in training')
    parser.add_argument('--val_size',   type=float, default=0.5,
                        help='Must match the value used in training')
    parser.add_argument('--eval_on',    default='test',
                        choices=['train', 'val', 'test', 'all'])
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomSLAMDataset(
        data_root=args.data_root, transform=transform,
        sequences=args.sequences, scenes=args.scenes,
        use_cached=True, cache_dir=args.cache_dir)

    # Reproduce the same splits as training (fixed random_state=42)
    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=args.test_size, random_state=42)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=args.val_size, random_state=42)

    split_map = {'train': train_idx, 'val': val_idx,
                 'test': test_idx,   'all': list(range(len(dataset)))}
    eval_idx  = split_map[args.eval_on]
    print(f"Evaluating on {args.eval_on} split: {len(eval_idx)} samples")

    loader = DataLoader(Subset(dataset, eval_idx),
                        batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = SLAMErrorPredictor(num_scalar_features=8)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    predictions, ground_truths = run_inference(model, loader, device)
    metrics = calculate_all_metrics(ground_truths, predictions)

    print(f"\n{'='*60}")
    print(f"MAPE  : {metrics['mape']:.2f}%")
    print(f"RMSE  : {metrics['rmse']:.4f} cm")
    print(f"MAE   : {metrics['mae']:.4f} cm")
    print(f"R²    : {metrics['r2']:.4f}")
    print(f"{'='*60}")

    plot_evaluation_results(predictions, ground_truths, metrics, args.output_dir)
    save_results(metrics, predictions, ground_truths, args.output_dir)
    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
