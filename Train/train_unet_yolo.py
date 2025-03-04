import torch
import numpy as np
import logging
import os
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
from datasets import UrineStripDataset
from models import UNetYOLO
from losses import dice_loss, focal_loss
from config import (
    TRAIN_IMAGE_FOLDER, 
    TRAIN_MASK_FOLDER,
    VALID_IMAGE_FOLDER, 
    VALID_MASK_FOLDER,
    NUM_CLASSES, 
    BATCH_SIZE,
    PATIENCE,
    get_model_folder
)
from utils import compute_class_weights  # Import the new function
from config import LR_SCHEDULER_STEP_SIZE, LR_SCHEDULER_GAMMA  # Import scheduler config
from torch.amp import GradScaler, autocast  # Import mixed precision training tools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Regularization techniques
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to the batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def label_smoothing(targets, num_classes, smoothing=0.1):
    """Apply label smoothing to reduce overconfidence"""
    batch_size = targets.size(0)
    targets_one_hot = torch.zeros(batch_size, num_classes).to(device)
    targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
    targets_one_hot = targets_one_hot * (1 - smoothing) + smoothing / num_classes
    return targets_one_hot

def validate_model(model, dataloader, epoch):
    """Run validation on the model and return metrics."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        val_progress = tqdm(dataloader, desc=f"Validation Epoch {epoch+1}", 
                            position=2, leave=False)
        
        for images, targets, _ in val_progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = dice_loss(outputs, targets) + focal_loss(outputs, targets)
            
            # Track loss and accuracy
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate accuracy per image (using most common class)
            for i in range(preds.shape[0]):
                img_pred = preds[i].flatten()
                values, counts = torch.unique(img_pred, return_counts=True)
                mode_idx = torch.argmax(counts)
                most_common_class = values[mode_idx].item()
                
                if most_common_class == targets[i].item():
                    correct += 1
                total += 1
                
            val_progress.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    val_loss = val_loss / len(dataloader)
    val_accuracy = correct / total if total > 0 else 0
    
    return val_loss, val_accuracy

def train_model(num_epochs=50, batch_size=4, learning_rate=0.001, save_interval=1, 
               weight_decay=1e-4, dropout_prob=0.5, mixup_alpha=0.2, 
               label_smoothing_factor=0.1, grad_clip_value=1.0):    
    """ 
    Train the UNet-YOLO model with enhanced features and regularization:
    - Validation after each epoch
    - Model saving (best and latest)
    - Early stopping
    - GPU optimization
    - Advanced regularization techniques
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("training")
    
    # Create output directory for model checkpoints
    model_dir = get_model_folder()
    os.makedirs(model_dir, exist_ok=True)
    
    # File paths for model saving
    best_model_path = os.path.join(model_dir, "best_model.pt")
    latest_model_path = os.path.join(model_dir, "latest_model.pt")
    
    # Save training parameters for reproducibility
    training_params = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'dropout_prob': dropout_prob,
        'mixup_alpha': mixup_alpha,
        'label_smoothing': label_smoothing_factor,
        'grad_clip': grad_clip_value
    }
    
    with open(os.path.join(model_dir, "training_params.txt"), "w") as f:
        for param, value in training_params.items():
            f.write(f"{param}: {value}\n")
    
    # Load datasets with data augmentation
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    valid_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,  # Speed up data transfer to GPU
        num_workers=2,    # Increase number of workers for faster data loading
        drop_last=True    # Avoid problems with small batches
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2  # Increase number of workers for faster data loading
    )

    # Initialize model with specified dropout probability
    model = UNetYOLO(in_channels=3, out_channels=NUM_CLASSES, dropout_prob=dropout_prob).to(device)
    
    # Use mixed precision training if available
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Compute class weights
    class_weights = compute_class_weights(train_dataset, NUM_CLASSES).to(device)

    # Setup optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        amsgrad=True  # Use AMSGrad variant for more stable training
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    
    # Training metrics tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Create tqdm progress bars
    epoch_progress = tqdm(range(num_epochs), desc="Training epochs", position=0)
    
    # Main training loop
    for epoch in epoch_progress:
        # Training phase
        model.train()
        epoch_loss = 0.0
        
        # Track batch times for performance monitoring
        batch_times = []
        
        # Progress bar for training batches
        batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                              position=1, leave=False)
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start_time = time.time()
        for images, targets, _ in batch_progress:
            batch_start = time.time()
            
            # Move data to device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Apply mixup augmentation with probability 0.5
            apply_mixup = epoch >= 5 and np.random.random() < 0.5  # Start mixup after 5 epochs
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            if scaler is not None:
                # Use mixed precision training
                with autocast(device_type='cuda'):
                    if apply_mixup:
                        mixed_images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=mixup_alpha)
                        outputs = model(mixed_images)
                        loss = lam * dice_loss(outputs, targets_a, class_weights=class_weights) + (1 - lam) * dice_loss(outputs, targets_b, class_weights=class_weights)
                        loss += lam * focal_loss(outputs, targets_a, class_weights=class_weights) + (1 - lam) * focal_loss(outputs, targets_b, class_weights=class_weights)
                    else:
                        outputs = model(images)
                        loss = dice_loss(outputs, targets, class_weights=class_weights) + focal_loss(outputs, targets, class_weights=class_weights)
                
                # Scale loss and compute gradients
                scaler.scale(loss).backward()
                
                # Apply gradient clipping
                if grad_clip_value > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                if apply_mixup:
                    mixed_images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=mixup_alpha)
                    outputs = model(mixed_images)
                    loss = lam * dice_loss(outputs, targets_a, class_weights=class_weights) + (1 - lam) * dice_loss(outputs, targets_b, class_weights=class_weights)
                    loss += lam * focal_loss(outputs, targets_a, class_weights=class_weights) + (1 - lam) * focal_loss(outputs, targets_b, class_weights=class_weights)
                else:
                    outputs = model(images)
                    loss = dice_loss(outputs, targets, class_weights=class_weights) + focal_loss(outputs, targets, class_weights=class_weights)
                
                loss.backward()
                
                # Apply gradient clipping
                if grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                
                optimizer.step()
            
            # Update metrics
            current_loss = loss.item()
            epoch_loss += current_loss
            
            # Update progress bar
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            batch_progress.set_postfix({
                "Loss": f"{current_loss:.4f}",
                "Batch time": f"{batch_time:.3f}s"
            })
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate training metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_batch_time = sum(batch_times) / len(batch_times)
        epoch_time = time.time() - start_time
        train_losses.append(avg_epoch_loss)
        
        # Validation phase
        val_loss, val_accuracy = validate_model(model, valid_loader, epoch)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update epoch progress bar
        epoch_progress.set_postfix({
            "Train Loss": f"{avg_epoch_loss:.4f}",
            "Val Loss": f"{val_loss:.4f}", 
            "Val Acc": f"{val_accuracy:.4f}",
            "LR": f"{current_lr:.6f}"
        })
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}, "
            f"LR: {current_lr:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Save latest model every save_interval epochs
        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }, latest_model_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'best_val_loss': best_val_loss
            }, best_model_path)
            
            logger.info(f"✓ Saved best model with val_loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epochs")
            
            # Check for early stopping
            if epochs_no_improve >= PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                early_stop = True
                break
        
        # Clear GPU memory cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final model saving
    final_model_path = os.path.join(model_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    
    # Report training completion
    status = "Early stopped" if early_stop else "Completed"
    logger.info(f"Training {status} after {epoch+1} epochs")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Models saved in {model_dir}")
    
    print(f"\nTraining {status}! Models saved to {model_dir}")
    
    # Load the best model for return
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'early_stopped': early_stop,
        'epochs_completed': epoch + 1,
        'model_directory': model_dir
    }

if __name__ == "__main__":
    train_model()
