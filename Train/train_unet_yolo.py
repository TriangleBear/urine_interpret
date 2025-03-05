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
from utils import compute_class_weights  # Import the correct function
from config import LR_SCHEDULER_STEP_SIZE, LR_SCHEDULER_GAMMA  # Import scheduler config
from torch.amp import GradScaler, autocast  # Import mixed precision training tools
import tracemalloc  # Import for memory profiling

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

def validate_model(model, dataloader, epoch, logger):
    """Run validation on the model and return metrics."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        val_progress = tqdm(dataloader, desc=f"Validation Epoch {epoch+1}", 
                            position=2, leave=False)
        
        for images, targets, _ in val_progress:
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                logger.info(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    val_loss = val_loss / len(dataloader)
    val_accuracy = correct / total if total > 0 else 0
    
    return val_loss, val_accuracy

def train_model(num_epochs=None, batch_size=None, learning_rate=None, save_interval=None, 
               weight_decay=None, dropout_prob=0.5, mixup_alpha=0.2, 
               label_smoothing_factor=0.1, grad_clip_value=1.0):
    """
    Train the UNet-YOLO model with T4 GPU optimizations for Colab
    """
    # Use config values if not specified
    num_epochs = num_epochs or NUM_EPOCHS
    batch_size = batch_size or BATCH_SIZE
    learning_rate = learning_rate or LEARNING_RATE  
    save_interval = save_interval or SAVE_INTERVAL
    weight_decay = weight_decay or WEIGHT_DECAY
    
    # Start memory profiling
    tracemalloc.start()
    
    # Clean memory before starting
    clean_memory()  # Use the function from config
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("training")
    logger.info(f"Training on device: {device}")
    
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
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'mixed_precision': USE_MIXED_PRECISION,
        'grad_clip': grad_clip_value
    }
    
    # Load datasets with data augmentation
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    valid_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty! Check path: {TRAIN_IMAGE_FOLDER}")
    if len(valid_dataset) == 0:
        raise ValueError(f"Validation dataset is empty! Check path: {VALID_IMAGE_FOLDER}")
        
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(valid_dataset)} validation samples")
    
    # Use a smaller validation subset to save memory
    valid_subset_size = min(50, len(valid_dataset))
    valid_dataset = torch.utils.data.Subset(valid_dataset, range(valid_subset_size))
    
    # Configure data loaders with T4 optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,  # Reduced workers for Colab
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size*2,  # Can use larger batch size for validation
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )

    # Initialize model with specified dropout probability
    model = UNetYOLO(in_channels=3, out_channels=NUM_CLASSES, dropout_prob=dropout_prob).to(device)
    
    # Enable gradient checkpointing to save memory
    if USE_GRADIENT_CHECKPOINTING:
        model.unet.use_checkpointing = True
    
    # Use mixed precision training for T4
    scaler = GradScaler() if USE_MIXED_PRECISION and torch.cuda.is_available() else None
    
    # Compute class weights
    class_weights = compute_class_weights(train_dataset, NUM_CLASSES).to(device)
    
    # Setup optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        amsgrad=True  # More stable for T4
    )
    
    # T4-optimized learning rate scheduler
    scheduler = torch.optim.OneCycleLR(
        optimizer, 
        max_lr=learning_rate*10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader) // GRADIENT_ACCUMULATION_STEPS,
        pct_start=0.3,  # Warm up for 30% of training
        div_factor=25,  # LR_max / initial_lr
        final_div_factor=10000,  # LR_min / initial_lr
    )

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    
    # Main training loop with gradient accumulation
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)  # Start with clean gradients
        
        # Progress bar for training batches
        batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, targets, _) in enumerate(batch_progress):
            # Move data to device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Apply mixup with probability 0.5 after warmup
            apply_mixup = epoch >= 5 and np.random.random() < 0.5
            
            # Use mixed precision for T4
            if scaler:
                with autocast(device_type='cuda'):
                    if apply_mixup:
                        mixed_images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=mixup_alpha)
                        outputs = model(mixed_images)
                        loss = lam * dice_loss(outputs, targets_a, class_weights=class_weights) + \
                               (1 - lam) * dice_loss(outputs, targets_b, class_weights=class_weights)
                        loss += lam * focal_loss(outputs, targets_a, class_weights=class_weights) + \
                                (1 - lam) * focal_loss(outputs, targets_b, class_weights=class_weights)
                    else:
                        outputs = model(images)
                        loss = dice_loss(outputs, targets, class_weights=class_weights) + \
                               focal_loss(outputs, targets, class_weights=class_weights)
                    
                    # Scale loss by accumulation factor
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                # Scale loss and compute gradients
                scaler.scale(loss).backward()
                
                # Steps when accumulation is complete or at the end
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                    # Apply gradient clipping
                    if grad_clip_value > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                    
                    # Optimizer step and update scaler
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # More efficient
                    scheduler.step()  # Update LR scheduler every step for OneCycleLR
            else:
                # Standard training path without mixed precision
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
            
            # Track loss
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # Adjust for scaling
            
            # Update progress bar
            lr = optimizer.param_groups[0]['lr']
            batch_progress.set_postfix({
                "Loss": f"{loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f}",
                "LR": f"{lr:.6f}"
            })
            
            # Clear cache periodically to avoid OOM
            if (batch_idx + 1) % 10 == 0:
                clean_memory()
        
        # Calculate average loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # Clean memory before validation
        clean_memory()
        
        # Validation phase (simplified for memory efficiency)
        val_loss, val_accuracy = validate_model(model, valid_loader, epoch, logger)
        
        # Update LR scheduler
        # scheduler.step(val_loss)  # For ReduceLROnPlateau, not needed for OneCycleLR
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}, "
            f"LR: {lr:.6f}"
        )
        
        # Save checkpoint periodically to reduce I/O overhead
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss
            }, latest_model_path)
        
        # Save best model (always)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save best model with torch.save
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss
            }, best_model_path)
            
            logger.info(f"✓ Saved best model with val_loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            
            # Early stopping check
            if epochs_no_improve >= PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                early_stop = True
                break
        
        # Clear memory after each epoch
        clean_memory()
    
    # Final model saving
    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pt"))
    
    # Stop memory profiling
    tracemalloc.stop()
    
    # Clean up
    clean_memory()
    
    logger.info(f"Training completed after {epoch+1} epochs")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    train_model()
