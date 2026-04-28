#!/usr/bin/env python3


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

# Shared model definition (single source of truth for the architecture)
from model      import SLAMErrorPredictor
from dataloader import CustomSLAMDataset


# ===========================================================================
# Training
# ===========================================================================

def train_model(model, train_loader, val_loader,
                num_epochs=20, lr=0.001, device='cuda',
                output_dir='results'):
    """Train and save the best model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    device     = torch.device(device if torch.cuda.is_available() else 'cpu')
    model      = model.to(device)
    best_loss  = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        running = 0.0
        for imgs, psds, scalars, errors in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"):
            imgs    = imgs.to(device)
            psds    = psds.to(device)
            scalars = scalars.to(device)
            errors  = errors.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            preds = model(imgs, psds, scalars)
            loss  = criterion(preds, errors)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)

        train_loss = running / len(train_loader.dataset)
        train_losses.append(train_loss)

        # --- Validation phase ---
        model.eval()
        running = 0.0
        with torch.no_grad():
            for imgs, psds, scalars, errors in tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"):
                imgs    = imgs.to(device)
                psds    = psds.to(device)
                scalars = scalars.to(device)
                errors  = errors.to(device).float().unsqueeze(1)
                preds   = model(imgs, psds, scalars)
                running += criterion(preds, errors).item() * imgs.size(0)

        val_loss = running / len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved → {best_model_path}")

        print('-' * 50)

    return model, train_losses, val_losses, best_model_path


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_model(model, test_loader, device='cuda'):
    """Return (mse, rmse, mae, predictions, ground_truths)."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for imgs, psds, scalars, errors in tqdm(test_loader, desc="Evaluating"):
            imgs    = imgs.to(device)
            psds    = psds.to(device)
            scalars = scalars.to(device)
            out     = model(imgs, psds, scalars)
            preds.extend(out.cpu().numpy().flatten())
            gts.extend(errors.numpy())

    preds = np.array(preds)
    gts   = np.array(gts)
    mse   = np.mean((preds - gts) ** 2)
    rmse  = np.sqrt(mse)
    mae   = np.mean(np.abs(preds - gts))
    return mse, rmse, mae, preds, gts


# ===========================================================================
# Plotting
# ===========================================================================

def plot_results(train_losses, val_losses, predictions, ground_truths,
                 output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(ground_truths, predictions, alpha=0.5)
    lim = max(np.max(ground_truths), np.max(predictions))
    plt.plot([0, lim], [0, lim], 'r--')
    plt.xlabel('Ground Truth (cm)'); plt.ylabel('Predicted (cm)')
    plt.title('Predicted vs Ground Truth SLAM Error')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'predictions_vs_gt.png'))
    plt.close()


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train SLAM error predictor from cached features')
    parser.add_argument('--data_root',  required=True)
    parser.add_argument('--cache_dir',  required=True)
    parser.add_argument('--sequences',  nargs='+')
    parser.add_argument('--scenes',     nargs='+')
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--test_size',  type=float, default=0.3)
    parser.add_argument('--val_size',   type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("Creating dataset …")
    dataset = CustomSLAMDataset(
        data_root=args.data_root,
        transform=transform,
        sequences=args.sequences,
        scenes=args.scenes,
        use_cached=True,
        cache_dir=args.cache_dir,
    )
    if len(dataset) == 0:
        raise ValueError("No data found. Check --cache_dir and --scenes/--sequences.")

    print("Splitting dataset …")
    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=args.test_size, random_state=42)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=args.val_size, random_state=42)

    kw = dict(batch_size=args.batch_size, num_workers=4)
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True,  **kw)
    val_loader   = DataLoader(Subset(dataset, val_idx),   shuffle=False, **kw)
    test_loader  = DataLoader(Subset(dataset, test_idx),  shuffle=False, **kw)
    print(f"Splits: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    model = SLAMErrorPredictor(num_scalar_features=8)
    print("Starting training …")
    model, train_losses, val_losses, best_path = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs, lr=args.lr,
        device=args.device, output_dir=args.output_dir)

    # Reload best weights for evaluation
    model.load_state_dict(torch.load(best_path, map_location=device))

    print("Evaluating …")
    mse, rmse, mae, preds, gts = evaluate_model(model, test_loader, args.device)
    print(f"\nRMSE: {rmse:.4f} cm   MAE: {mae:.4f} cm   MSE: {mse:.4f}")

    plot_results(train_losses, val_losses, preds, gts, args.output_dir)
    print(f"Done. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
