#!/usr/bin/env python3
"""
Training utilities for UNet model for spine segmentation.
Contains loss functions and training utilities.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from tqdm.notebook import tqdm
    import matplotlib.pyplot as plt
except ImportError:
    from tqdm import tqdm
    import matplotlib.pyplot as plt


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation"""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten predictions and target
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum()

        # Return Dice Loss
        return 1 - ((2. * intersection + self.smooth) /
                    (pred_flat.sum() + target_flat.sum() + self.smooth))


def combined_loss(pred, target, alpha=0.5):
    """
    Combine Binary Cross Entropy and Dice loss

    Args:
        pred: Model predictions (logits)
        target: Ground truth masks
        alpha: Weight for BCE loss (1-alpha for Dice loss)

    Returns:
        Combined loss value
    """
    # Combine BCE and Dice loss
    bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    dice_loss = DiceLoss()(torch.sigmoid(pred), target)
    return alpha * bce_loss + (1 - alpha) * dice_loss


def advanced_training_monitoring(model, device, train_loader, val_loader, epochs=100,
                                 lr=1e-4, save_dir=None, checkpoint_freq=5):
    """
    Model training with enhanced monitoring and visualization
    """
    # Set up saving directory
    if save_dir is None:
        save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # Lists for monitoring
    train_losses = []
    val_losses = []
    dice_scores = []
    epochs_list = []
    lr_rates = []

    # Best model tracking
    best_val_loss = float('inf')

    # Training start time
    total_start_time = time.time()

    print(f"\n{'=' * 15} TRAINING STARTED {'=' * 15}")
    print(f"Total epochs: {epochs}")
    print(f"Initial learning rate: {lr}")
    print(f"Device: {device}")
    print(f"Number of batches: {len(train_loader)}")
    print(f"{'=' * 40}\n")

    # Epoch loop
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # -------------- Training Phase --------------
        model.train()
        train_epoch_loss = 0

        # Training progress bar
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]", leave=False)
        for batch in train_loop:
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update progress bar
            batch_loss = loss.item()
            train_epoch_loss += batch_loss
            train_loop.set_postfix(loss=f"{batch_loss:.4f}")

        # Average training loss
        train_loss = train_epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # -------------- Validation Phase --------------
        model.eval()
        val_epoch_loss = 0
        epoch_dice_score = 0

        # Validation progress bar
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]", leave=False)
        with torch.no_grad():
            for batch in val_loop:
                images, masks = batch
                images, masks = images.to(device), masks.to(device)

                # Forward pass
                outputs = model(images)
                loss = combined_loss(outputs, masks)

                # Calculate Dice score
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (pred_masks * masks).sum()
                dice = (2. * intersection) / (pred_masks.sum() + masks.sum() + 1e-8)

                # Add values
                val_epoch_loss += loss.item()
                epoch_dice_score += dice.item()

                # Update progress bar
                val_loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")

        # Average validation metrics
        val_loss = val_epoch_loss / len(val_loader)
        dice_score = epoch_dice_score / len(val_loader)

        val_losses.append(val_loss)
        dice_scores.append(dice_score)
        epochs_list.append(epoch + 1)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        lr_rates.append(current_lr)

        # Calculate epoch duration
        epoch_time = time.time() - epoch_start_time

        # --------- Print Results ---------
        print(f"\nEpoch {epoch + 1}/{epochs} - {epoch_time:.1f}s")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Dice Score: {dice_score:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # --------- Save Model ---------
        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "unet_model_best_colab.pth"))
            print(f"  ✅ New best model saved! (Val Loss: {val_loss:.4f})")

        # Checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'dice_score': dice_score,
            }, checkpoint_path)
            print(f"  📁 Checkpoint saved: epoch_{epoch + 1}")

        # --------- Visualize Progress ---------
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            # Draw loss and dice graphs
            plt.figure(figsize=(15, 5))

            # Loss graph
            plt.subplot(1, 3, 1)
            plt.plot(epochs_list, train_losses, 'b-', label='Training Loss')
            plt.plot(epochs_list, val_losses, 'r-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Losses')
            plt.grid(True, alpha=0.3)

            # Dice score graph
            plt.subplot(1, 3, 2)
            plt.plot(epochs_list, dice_scores, 'g-')
            plt.xlabel('Epoch')
            plt.ylabel('Dice Score')
            plt.title('Validation Dice Score')
            plt.grid(True, alpha=0.3)

            # Learning rate graph
            plt.subplot(1, 3, 3)
            plt.plot(epochs_list, lr_rates, 'm-')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Changes')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')

            plt.tight_layout()
            plt.show()

    # Training completed
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'=' * 15} TRAINING COMPLETED {'=' * 15}")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final Dice score: {dice_scores[-1]:.4f}")
    print(f"Model saved: {os.path.join(save_dir, 'unet_model_best_colab.pth')}")
    print(f"{'=' * 46}")

    # Draw final graphs
    plt.figure(figsize=(15, 10))

    # Loss graph
    plt.subplot(2, 2, 1)
    plt.plot(epochs_list, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_list, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.grid(True, alpha=0.3)

    # Dice score graph
    plt.subplot(2, 2, 2)
    plt.plot(epochs_list, dice_scores, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice Score')
    plt.grid(True, alpha=0.3)

    # Learning rate graph
    plt.subplot(2, 2, 3)
    plt.plot(epochs_list, lr_rates, 'm-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Changes')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'dice_scores': dice_scores,
        'best_val_loss': best_val_loss,
        'final_dice_score': dice_scores[-1]
    }


def train_model(model, device, train_loader, val_loader, epochs, lr, save_dir, save_frequency=5):
    """
    Basic training function - kept for compatibility
    """
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # Initialize variables
    best_val_loss = float('inf')
    start_time = time.time()

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training phase
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(imgs)

            # Calculate loss
            loss = combined_loss(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        # Calculate average training loss
        train_loss = epoch_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        dice_score = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                # Forward pass
                outputs = model(imgs)

                # Calculate loss
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()

                # Calculate Dice score
                pred = torch.sigmoid(outputs) > 0.5
                dice_score += (2 * (pred * masks).sum()) / ((pred + masks).sum() + 1e-8)

        # Calculate average validation metrics
        val_loss /= len(val_loader)
        dice_score /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        # Print epoch summary
        print(f'Epoch {epoch + 1}/{epochs} completed in {time.time() - start_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice Score: {dice_score:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'unet_model_best.pth'))
            print(f'New best model saved with validation loss: {val_loss:.4f}')

        # Save checkpoint
        if (epoch + 1) % save_frequency == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'dice_score': dice_score
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch + 1}.pth'))
            print(f'Checkpoint saved at epoch {epoch + 1}')

        # Reset timer for next epoch
        start_time = time.time()

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'unet_model_final.pth'))
    print('Training completed. Final model saved.')