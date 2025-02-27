import torch
import torch.nn.functional as F
import torchvision.transforms as T
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import gc
import os
from ultralytics.nn.tasks import DetectionModel

def train_unet_yolo(batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS, patience=PATIENCE, pre_trained_weights=None):

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
    
    # Clear memory at start
    torch.cuda.empty_cache()
    gc.collect()

    # Reduce image size and batch size further
    max_size = 256  # Further reduced from 512 to 256
    
    # Create datasets with reduced size and data augmentation
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.RandomResizedCrop(max_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = T.Compose([
        T.Resize(max_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, transform=train_transform)
    val_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER, transform=val_transform)
    test_dataset = UrineStripDataset(TEST_IMAGE_FOLDER, TEST_MASK_FOLDER, transform=val_transform)

    # Create data loaders with memory optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,  # Set batch size to 1 consistently for memory management
        shuffle=True, 
        num_workers=0,  # Set workers to 0 to reduce memory overhead
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=0, 
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=0, 
        pin_memory=False
    )

    # Compute class weights using the training dataset
    class_weights = compute_class_weights(train_dataset)
    class_weights = class_weights.clone().detach().to(device)  # Updated to avoid warning

    print(f"Computed class weights: {class_weights}")

    # Ensure the criterion uses the correct number of classes
    if class_weights.shape[0] != NUM_CLASSES:
        raise ValueError(f"Expected class weights tensor of shape ({NUM_CLASSES},) but got {class_weights.shape}")
    criterion_ce = torch.nn.CrossEntropyLoss(weight=class_weights[:NUM_CLASSES], label_smoothing=0.1, reduction='mean')
    criterion_dice = dice_loss

    # Model initialization with memory optimization
    model = UNetYOLO(3, NUM_CLASSES, dropout_prob=0.5).to(device)
    
    # Enable gradient checkpointing to save memory
    model.unet.eval()  # Set UNet to eval mode

    # Only train decoder initially
    for layer in model.unet.up1, model.unet.up2, model.unet.up3, model.unet.up4:
        layer.train()

    # Load pre-trained weights if specified
    if pre_trained_weights:
        try:
            torch.serialization.add_safe_globals([DetectionModel])
            model.load_state_dict(torch.load(pre_trained_weights, map_location=device, weights_only=False), strict=False)
            ic(f"Successfully loaded pre-trained weights from {pre_trained_weights}")
        except Exception as e:
            print(f"Error loading pre-trained weights: {e}")
            print("Continuing with randomly initialized weights.")

    # Use a smaller learning rate
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


    train_losses = []
    val_losses = []
    val_accuracies = []

    # Initialize training loop variables
    best_loss = float('inf')
    early_stop_counter = 0
    model_folder = get_model_folder()
    model_filename = os.path.join(model_folder, "unet_model.pt")

    for epoch in range(NUM_EPOCHS):  # Start training loop

        # Clear memory at the start of each epoch to prevent OOM
        torch.cuda.empty_cache()
        gc.collect()
        
        # First 5 epochs: train only decoder to save memory and reduce GPU load
        if epoch == 5:
            for layer in model.unet.inc, model.unet.down1, model.unet.down2, model.unet.down3, model.unet.down4:
                layer.train()  # Start training encoder after 5 epochs
        
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)

        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}") as pbar:
            for i, (images, labels) in enumerate(train_loader):
                try:
                # Check if labels are empty and skip the batch if so
                    if labels.numel() == 0:
                        print(f"Skipping batch {i} due to empty labels.")
                        continue
                
                # Move to GPU and free CPU memory

                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    images = dynamic_normalization(images)

                    # Debugging: Print shapes of images and labels
                    print(f"Batch {i}: images shape: {images.shape}, labels shape: {labels.shape}")
                    
                    # Resize images to reduce memory
                    if images.shape[2] > max_size or images.shape[3] > max_size:
                        scale_factor = max_size / max(images.shape[2], images.shape[3])
                        new_h, new_w = int(images.shape[2] * scale_factor), int(images.shape[3] * scale_factor)
                        images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    
                    # Use autocast for mixed precision to further reduce memory usage
                    with autocast(device_type="cuda"):
                        outputs = model(images)
                        
                        # Global Average Pooling for classification
                        pooled_outputs = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)
                        
                        # Debugging: Print shapes of outputs and labels
                        print(f"Batch {i}: pooled_outputs shape: {pooled_outputs.shape}, labels shape: {labels.shape}")
                        
                        # Use combined loss for classification
                        loss_ce = criterion_ce(pooled_outputs, labels)  # Use pooled_outputs directly for loss
                        loss_dice = criterion_dice(outputs, labels)  # Use original outputs for dice loss
                        loss = loss_ce + loss_dice
                    
                    # Scale loss and backward pass
                    loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if (i + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        scaler.step(optimizer)
                        scaler.update()  # Ensure update is called after step
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                    
                    # Free up memory after each operation
                    del images, labels, outputs
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

        # Validation with memory optimization to handle OOM

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
                    
                    # Debugging: Print shapes of outputs and labels
                    print(f"Validation Batch: pooled_outputs shape: {pooled_outputs.shape}, labels shape: {labels.shape}")
                    
                    loss_val = criterion_ce(pooled_outputs, labels)  # Use pooled_outputs directly for loss
                    val_loss += loss_val.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(pooled_outputs, 1)  # Use pooled_outputs directly for accuracy
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Debugging: Print unique values in labels and predictions
                    print(f"Validation Batch: unique labels: {torch.unique(labels)}, unique predictions: {torch.unique(predicted)}")
                    
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
        
        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Save the best model based on validation loss to prevent overfitting
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
        
        # Check early stopping criteria
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
                
                loss = criterion_ce(pooled_outputs, labels)  # Use pooled_outputs directly for loss
                test_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(pooled_outputs, 1)  # Use pooled_outputs directly for accuracy
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Debugging: Print unique values in labels and predictions
                print(f"Test Batch: unique labels: {torch.unique(labels)}, unique predictions: {torch.unique(predicted)}")
                
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
