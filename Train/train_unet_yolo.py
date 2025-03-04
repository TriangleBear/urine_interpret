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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train_model(num_epochs=50, batch_size=16, learning_rate=0.001, save_interval=1):    
    """
    Train the UNet-YOLO model with enhanced features:
    - Validation after each epoch
    - Model saving (best and latest)
    - Early stopping
    - GPU optimization
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
    
    # Load datasets
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    valid_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,  # Speed up data transfer to GPU
        num_workers=2,    # Parallel data loading
        drop_last=True    # Avoid problems with small batches
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )

    # Initialize model, optimizer and scheduler
    model = UNetYOLO(in_channels=3, out_channels=NUM_CLASSES).to(device)
    
    # Use mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Setup optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )

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
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            if scaler is not None:
                # Use mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = dice_loss(outputs, targets) + focal_loss(outputs, targets)
                
                # Scale loss and compute gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                outputs = model(images)
                loss = dice_loss(outputs, targets) + focal_loss(outputs, targets)
                loss.backward()
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
        
        # Calculate training metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_batch_time = sum(batch_times) / len(batch_times)
        epoch_time = time.time() - start_time
        train_losses.append(avg_epoch_loss)
        
        # Validation phase
        val_loss, val_accuracy = validate_model(model, valid_loader, epoch)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
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
            f"Time: {epoch_time:.2f}s, "
            f"Avg Batch: {avg_batch_time:.3f}s"
        )
        
        # Save latest model every epoch
        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
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
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'best_val_loss': best_val_loss
            }, best_model_path)
            
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
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
        'epochs_completed': epoch + 1
    }
