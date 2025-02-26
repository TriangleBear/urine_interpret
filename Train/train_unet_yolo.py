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
import gc
import os

# Configure CUDA for maximum performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def train_unet_yolo(batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS, patience=PATIENCE, pre_trained_weights=None):
    # Use safer CUDA memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
    
    # Clear memory at start
    torch.cuda.empty_cache()
    gc.collect()
    
    # Don't change default tensor type - this can cause issues
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Reduce image size and batch size further
    max_size = 256  # Further reduced from 512 to 256
    
    # Create datasets with reduced size
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    val_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    test_dataset = UrineStripDataset(TEST_IMAGE_FOLDER, TEST_MASK_FOLDER)

    # Create data loaders with CUDA optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2,  # Use multiple workers for loading
        pin_memory=True,  # Pin memory for faster host to device transfers
        persistent_workers=True  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Compute class weights using the training dataset
    class_weights = compute_class_weights(train_dataset)
    class_weights = class_weights.clone().detach().to(device)  # Updated to avoid warning

    print(f"Computed class weights: {class_weights}")
    print(f"Number of classes: {NUM_CLASSES}")

    # Model initialization with memory optimization
    model = UNetYOLO(3, NUM_CLASSES, dropout_prob=0.3).to(device)  # Reduced dropout
    
    # Test model with a small batch to ensure it works - with error handling
    try:
        # Use a very small input for initial test
        test_input = torch.zeros((1, 3, 64, 64), device=device)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"Model output shape: {test_output.shape}")
        output_channels = test_output.shape[1]
        
        # Clear test tensors explicitly
        del test_input, test_output
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error during model initialization: {e}")
        torch.cuda.empty_cache()
        raise e
    
    # Check if the model's output matches NUM_CLASSES
    if output_channels != NUM_CLASSES:
        print(f"Warning: Model outputs {output_channels} classes, but NUM_CLASSES is {NUM_CLASSES}")
        print("Adjusting CrossEntropyLoss to use no weights")
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    else:
        # Use CrossEntropyLoss for classification task
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.05,  # Reduced label smoothing
            reduction='mean'
        )
    
    # Train the full model from the beginning for better convergence
    model.train()  # Set the entire model to train mode
    
    # Use higher learning rate for better convergence
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Higher learning rate
    scaler = GradScaler()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)  # Adjusted scheduler

    train_losses = []
    val_losses = []
    val_accuracies = []

    # Initialize training loop variables
    best_loss = float('inf')
    early_stop_counter = 0
    model_folder = get_model_folder()
    model_filename = os.path.join(model_folder, "unet_model.pt")

    # Don't use CUDA graph warm-up as it's causing issues
    # Instead do a simple warm-up pass
    try:
        with torch.no_grad():
            _ = model(torch.zeros((1, 3, 64, 64), device=device))
        torch.cuda.synchronize()
    except Exception as e:
        print(f"Warning: Model warm-up failed: {e}")
        # Continue anyway
    
    for epoch in range(NUM_EPOCHS):  # Start training loop

        # Clear memory at the start of each epoch to prevent OOM
        torch.cuda.empty_cache()
        gc.collect()
        
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)

        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}") as pbar:
            for i, (images, labels) in enumerate(train_loader):
                try:
                    # Move to GPU and free CPU memory
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    # Apply normalization - ensure images are in range [0,1]
                    if torch.max(images) > 1.0:
                        images = images / 255.0
                    
                    # Use dynamic normalization
                    images = dynamic_normalization(images)
                    
                    # Resize images to reduce memory
                    if images.shape[2] > max_size or images.shape[3] > max_size:
                        scale_factor = max_size / max(images.shape[2], images.shape[3])
                        new_h, new_w = int(images.shape[2] * scale_factor), int(images.shape[3] * scale_factor)
                        images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    
                    # Print shapes and unique values in the first batch of the first epoch
                    if i == 0 and epoch == 0:
                        print(f"Images shape: {images.shape}")
                        print(f"Labels shape: {labels.shape}")
                        print(f"Unique labels: {torch.unique(labels)}")
                    
                    # Use autocast for mixed precision to fully utilize tensor cores
                    with autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(images)
                        
                        # Global Average Pooling for classification
                        pooled_outputs = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)
                        
                        # DEBUG: Print output for diagnosis
                        if i == 0 and epoch == 0:
                            print(f"Output shape: {outputs.shape}")
                            print(f"Pooled output shape: {pooled_outputs.shape}")
                            print(f"Pooled outputs: {pooled_outputs}")
                            
                        # Convert labels to long for CrossEntropyLoss
                        labels = labels.long()
                        
                        # Use cross entropy loss for classification
                        loss = criterion(pooled_outputs, labels)
                    
                    # Scale loss and backward pass
                    loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                    
                    # Only unscale when we're going to step
                    if (i + 1) % accumulation_steps == 0:
                        # Gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        
                        scaler.step(optimizer)
                        scaler.update()  # Ensure update is called after step
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                    
                    # Free up memory after each operation
                    del images, labels, outputs, pooled_outputs
                    torch.cuda.empty_cache()
                    
                    epoch_loss += loss.item() * accumulation_steps
                    pbar.update(1)
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error in batch {i}, skipping...")
                        # Force release of memory
                        for p in model.parameters():
                            if p.grad is not None:
                                del p.grad  # Remove gradients
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
        all_preds = []
        all_labels = []
        
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
                    
                    # Ensure labels are long tensors for loss calculation
                    labels = labels.long()
                    
                    loss_val = criterion(pooled_outputs, labels)
                    val_loss += loss_val.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(pooled_outputs, 1)
                    
                    # Collect for confusion matrix
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
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
        
        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct / max(1, total)  # Avoid division by zero
        val_accuracies.append(val_accuracy)
        
        # Print confusion matrix every 5 epochs to diagnose classification issues
        if epoch % 5 == 0 and len(all_preds) > 0:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
            print("\nConfusion Matrix:")
            print(cm)
            
            # Calculate and print class-wise accuracy
            print("\nClass-wise accuracy:")
            for cls in range(NUM_CLASSES):
                cls_total = np.sum(np.array(all_labels) == cls)
                if cls_total > 0:
                    cls_correct = np.sum((np.array(all_preds) == cls) & (np.array(all_labels) == cls))
                    print(f"Class {cls}: {100 * cls_correct / cls_total:.2f}% ({cls_correct}/{cls_total})")
                else:
                    print(f"Class {cls}: No samples")
        
        print(f"Epoch {epoch+1}: Train Loss {avg_loss:.4f}, Val Loss {avg_val_loss:.4f}, Val Accuracy {val_accuracy:.2f}%")
        
        # Adjust learning rate based on validation loss
        scheduler.step(epoch + avg_val_loss)

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
        
        # Explicitly synchronize CUDA operations at end of epoch
        torch.cuda.synchronize()
        
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
