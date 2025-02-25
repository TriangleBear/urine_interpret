import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from icecream import ic
from config import *
from models import UNetYOLO
from datasets import UrineStripDataset
from losses import dice_loss, focal_loss
from utils import compute_mean_std, dynamic_normalization, compute_class_weights
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from config import get_model_folder
import gc
import os

def train_unet_yolo(batch_size=1, accumulation_steps=32, patience=PATIENCE, pre_trained_weights=None):
    # Set environment variables for memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
    
    # Clear memory at start
    torch.cuda.empty_cache()
    gc.collect()

    # Reduce image size and batch size further
    max_size = 256  # Further reduced from 512 to 256
    
    # Create datasets with reduced size
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    val_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    test_dataset = UrineStripDataset(TEST_IMAGE_FOLDER, TEST_MASK_FOLDER)

    # Create data loaders with memory optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,  # Using batch_size=1
        shuffle=True, 
        num_workers=1,  # Reduced workers
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=1, 
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=1, 
        pin_memory=False
    )

    # Compute class weights using the training dataset
    class_weights = compute_class_weights(train_dataset)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Computed class weights: {class_weights}")

    # Model initialization with memory optimization
    model = UNetYOLO(3, NUM_CLASSES, dropout_prob=0.5).to(device)
    
    # Enable gradient checkpointing to save memory
    model.encoder.eval()  # Set encoder to eval mode
    model.decoder.train() # Only train decoder initially

    # Load pre-trained weights if specified
    if pre_trained_weights:
        model.load_state_dict(torch.load(pre_trained_weights, map_location=device), strict=False)
        ic(f"Successfully loaded pre-trained weights from {pre_trained_weights}")

    # Use a smaller learning rate
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
    scaler = GradScaler()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    train_losses = []
    val_losses = []
    val_accuracies = []

    # Initialize training loop variables
    best_loss = float('inf')
    early_stop_counter = 0
    model_folder = get_model_folder()
    model_filename = os.path.join(model_folder, "unet_model.pt")

    for epoch in range(NUM_EPOCHS):
        # Clear memory at the start of each epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        # First 5 epochs: train only decoder to save memory
        if epoch == 5:
            model.encoder.train()  # Start training encoder after 5 epochs
        
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)

        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}") as pbar:
            for i, (images, labels) in enumerate(train_loader):
                try:
                    # Move to GPU and free CPU memory
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    images = dynamic_normalization(images)
                    
                    # Resize images to reduce memory
                    if images.shape[2] > max_size or images.shape[3] > max_size:
                        scale_factor = max_size / max(images.shape[2], images.shape[3])
                        new_h, new_w = int(images.shape[2] * scale_factor), int(images.shape[3] * scale_factor)
                        images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    
                    # Use autocast for mixed precision
                    with autocast(device_type="cuda"):
                        outputs = model(images)
                        
                        # Global Average Pooling for classification
                        # Average over spatial dimensions to get [batch, classes]
                        pooled_outputs = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)
                        
                        # Use cross entropy loss for classification
                        loss = criterion(pooled_outputs, labels)
                    
                    # Scale loss and backward pass
                    loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    if (i + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                    
                    # Free up memory
                    del images, labels, outputs, pooled_outputs
                    torch.cuda.empty_cache()
                    
                    epoch_loss += loss.item() * accumulation_steps
                    pbar.update(1)
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error in batch {i}, skipping...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        # Validation with memory optimization
        val_loss = 0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                try:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    # Resize images to reduce memory
                    if images.shape[2] > max_size or images.shape[3] > max_size:
                        scale_factor = max_size / max(images.shape[2], images.shape[3])
                        new_h, new_w = int(images.shape[2] * scale_factor), int(images.shape[3] * scale_factor)
                        images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    
                    outputs = model(images)
                    
                    # Global Average Pooling for classification
                    pooled_outputs = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)
                    
                    loss_val = criterion(pooled_outputs, labels)
                    val_loss += loss_val.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(pooled_outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Free memory
                    del images, labels, outputs, pooled_outputs, predicted, loss_val
                    torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error in validation, skipping batch...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct / total if total > 0 else 0
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}: Train Loss {avg_loss:.4f}, Val Loss {avg_val_loss:.4f}, Val Accuracy {val_accuracy:.2f}%")
        
        # Adjust learning rate
        scheduler.step(epoch + avg_val_loss)

        # Save the best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_counter = 0
            # Use CPU tensors to save memory during model saving
            cpu_model = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_model, model_filename)
            print("Best model saved.")
            del cpu_model  # Free memory
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} epochs.") 

        # Save model checkpoint if divisible by 10
        if (epoch + 1) % 10 == 0:
            cpu_model = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_model, os.path.join(model_folder, f"unet_model_epoch_{epoch+1}.pth"))
            del cpu_model  # Free memory
        
        # Check early stopping
        if early_stop_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()
        gc.collect()
    
    # Add final test evaluation
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            try:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Resize images to reduce memory
                if images.shape[2] > max_size or images.shape[3] > max_size:
                    scale_factor = max_size / max(images.shape[2], images.shape[3])
                    new_h, new_w = int(images.shape[2] * scale_factor), int(images.shape[3] * scale_factor)
                    images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
                
                outputs = model(images)
                
                # Global Average Pooling for classification
                pooled_outputs = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)
                
                loss = criterion(pooled_outputs, labels)
                test_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(pooled_outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Free memory
                del images, labels, outputs, pooled_outputs, predicted, loss
                torch.cuda.empty_cache()
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in testing, skipping batch...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0
    print(f"\nTest Set Results:")
    print(f"Average Loss: {test_loss/len(test_loader):.4f}")
    print(f"Accuracy: {test_accuracy:.2f}%")
    
    # Clear final memory before returning
    torch.cuda.empty_cache()
    gc.collect()
    
    return model, train_losses, val_losses, val_accuracies, test_accuracy
