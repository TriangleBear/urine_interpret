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
from utils import compute_mean_std, dynamic_normalization, compute_class_weights, post_process_segmentation
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import gc
import os
import matplotlib.pyplot as plt
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
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=0,
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

    # Compute class weights with special handling for missing classes
    print("Computing class weights...")
    class_weights = compute_class_weights(train_dataset, max_weight=50.0)
    class_weights = class_weights.clone().detach().to(device)
    
    print(f"Computed class weights: {class_weights}")

    # Standard multiclass approach using CrossEntropyLoss with class weights
    # Ensure the weight tensor is defined for all 12 classes
    print(f"Using class weights: {class_weights}")

    criterion_ce = torch.nn.CrossEntropyLoss(weight=class_weights, 
                                             label_smoothing=0.1, 
                                             reduction='mean', 
                                             ignore_index=NUM_CLASSES)  # Ignore empty labels during loss calculation
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

    # Use a smaller learning rate for better convergence with imbalanced data
    optimizer = AdamW(model.parameters(), lr=5e-6, weight_decay=1e-6)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


    train_losses = []
    val_losses = []
    val_accuracies = []

    # Initialize training loop variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    model_folder = get_model_folder()
    model_filename = os.path.join(model_folder, "unet_model.pt")
    best_model_path = os.path.join(model_folder, "unet_model_best.pt")
    metrics_folder = os.path.join(model_folder, "metrics")
    os.makedirs(metrics_folder, exist_ok=True)

    class_counts = {i: 0 for i in range(NUM_CLASSES + 1)}  # Include background class
    
    # Add this line to initialize the confusion matrix
    confmat = torch.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), device='cpu')

    # Add a function to save plots during training
    def save_epoch_plots(epoch, train_losses, val_losses, val_accuracies, class_correct, class_total, lr, metrics_folder):
        """Save plots for monitoring training progress at each epoch."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Training and validation loss
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Progress - Epoch {epoch+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Validation accuracy
        plt.subplot(2, 2, 2)
        plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'g-')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Per-class accuracy for key classes
        plt.subplot(2, 2, 3)
        key_classes = [0, 1, 5, 10]  # Example: focus on a few important classes
        class_names = ['Bilirubin', 'Blood', 'Nitrite', 'Strip']  # Corresponding names
        accuracies = []
        for cls in key_classes:
            if cls in class_total and class_total[cls] > 0:
                acc = 100 * class_correct[cls] / class_total[cls]
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        plt.bar(class_names, accuracies)
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.title('Per-class Accuracy')
        plt.ylim(0, 100)
        
        # Plot 4: Learning rate progression
        plt.subplot(2, 2, 4)
        plt.plot(range(1, epoch+2), lr, 'mo-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Progression')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = os.path.join(metrics_folder, f'epoch_{epoch+1:03d}_metrics.png')
        plt.savefig(plot_filename, dpi=100)
        plt.close()
        
        # Also save a "latest" version that always gets overwritten
        latest_plot = os.path.join(metrics_folder, 'latest_metrics.png')
        plt.figure(figsize=(15, 10))
        
        # Same plots as above
        # Plot 1
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Progress - Epoch {epoch+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2
        plt.subplot(2, 2, 2)
        plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'g-')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Plot 3
        plt.subplot(2, 2, 3)
        plt.bar(class_names, accuracies)
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.title('Per-class Accuracy')
        plt.ylim(0, 100)
        
        # Plot 4
        plt.subplot(2, 2, 4)
        plt.plot(range(1, epoch+2), lr, 'mo-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Progression')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(latest_plot, dpi=100)
        plt.close()

    # Track learning rate for plotting
    learning_rates = []

    for epoch in range(NUM_EPOCHS):  # Start training loop

        # Clear memory at the start of each epoch to prevent OOM
        torch.cuda.empty_cache()
        gc.collect()
        
        # First 5 epochs: train only decoder to save memory and reduce GPU load
        if epoch < 5:
            for layer in model.unet.inc, model.unet.down1, model.unet.down2, model.unet.down3, model.unet.down4:
                layer.eval()  # Freeze encoder layers
                
            # Create new weights that emphasize class 11 (strip)
            strip_focused_weights = class_weights.clone()
            # Boost the weight for class 11 (strip) by 5x
            strip_focused_weights[11] = strip_focused_weights[11] * 5.0
            # Scale down other classes
            for i in range(NUM_CLASSES):
                if i != 11:  # Skip strip class
                    strip_focused_weights[i] = strip_focused_weights[i] * 0.2
                    
            criterion_ce = torch.nn.CrossEntropyLoss(
                weight=strip_focused_weights[:NUM_CLASSES],  # Use all weights but focus on strip
                label_smoothing=0.1,
                reduction='mean',
                ignore_index=NUM_CLASSES
            )
        else:
            for layer in model.unet.inc, model.unet.down1, model.unet.down2, model.unet.down3, model.unet.down4:
                layer.train()  # Unfreeze encoder layers
                
            # Create new weights that emphasize classes 0-8, 10 (reagent pads)
            pad_focused_weights = class_weights.clone()
            # Boost the weight for reagent pad classes
            for i in list(range(9)) + [10]:  # Classes 0-8, 10
                pad_focused_weights[i] = pad_focused_weights[i] * 3.0
            # Scale down other classes
            for i in [9, 11]:  # Background and strip
                pad_focused_weights[i] = pad_focused_weights[i] * 0.3
                
            criterion_ce = torch.nn.CrossEntropyLoss(
                weight=pad_focused_weights[:NUM_CLASSES],  # Use all weights but focus on reagent pads
                label_smoothing=0.1,
                reduction='mean',
                ignore_index=NUM_CLASSES
            )
        
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)

        # Training loop - with correct handling of empty labels
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}") as pbar:
            for i, (images, labels, _) in enumerate(train_loader):
                try:
                    # Skip batches that consist entirely of empty labels
                    if torch.all(labels == NUM_CLASSES):
                        print(f"Skipping batch {i} - all empty labels")
                        pbar.update(1)
                        continue
                    
                    # For mixed batches, keep training but the loss function will
                    # ignore the empty labels thanks to ignore_index=NUM_CLASSES
                    
                    # Move to GPU and free CPU memory
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    images = dynamic_normalization(images)
                    
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
                        
                        # CrossEntropy loss (handled by ignore_index)
                        loss_ce = criterion_ce(pooled_outputs, labels)
                        
                        # Only compute dice and focal losses if there are non-background samples
                        background_mask = labels == NUM_CLASSES
                        if not torch.all(background_mask):
                            try:
                                # Wrap these in try-except - if they fail, fallback to CE loss only
                                with autocast(device_type="cuda"):  # Temporarily disable autocast for dice_loss
                                    loss_dice = criterion_dice(outputs, labels)
                                    loss_focal = focal_loss(outputs, labels, gamma=2.0)
                                
                                # Combined loss with balanced weights 
                                loss = loss_ce * 0.4 + loss_dice * 0.3 + loss_focal * 0.3
                            except RuntimeError as e:
                                print(f"Error in loss calculation: {e}")
                                print("Using CE loss only for this batch.")
                                loss = loss_ce
                        else:
                            # If all samples are background, use CE loss only
                            loss = loss_ce
                    
                    # Scale loss and backward pass
                    loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if scaler.get_scale() != 1.0:
                        scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    if (i + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        scaler.step(optimizer)
                        scaler.update()  # Ensure update is called after step
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                    
                        # Free up memory after processing

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

        # Validation with memory optimization to handle OOM
        val_loss = 0
        correct = 0
        total = 0
        model.eval()
        class_correct = {i: 0 for i in range(NUM_CLASSES)}
        class_total = {i: 0 for i in range(NUM_CLASSES)}
        
        # Validation with memory optimization - update to use layered post-processing
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc="Validation"):
                try:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    # Resize images to reduce memory
                    if images.shape[2] > max_size or images.shape[3] > max_size:
                        scale_factor = max_size / max(images.shape[2], images.shape[3])
                        new_h, new_w = int(images.shape[2] * scale_factor), int(images.shape[3] * scale_factor)
                        images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    
                    outputs = model(images)
                    
                    # Apply post-processing that respects class hierarchy
                    processed_outputs = post_process_segmentation(outputs, apply_layering=True)
                    
                    # Global Average Pooling for classification
                    pooled_outputs = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)
                    
                    loss_val = criterion_ce(pooled_outputs, labels)
                    val_loss += loss_val.item()
                    
                    # Calculate accuracy using hierarchical predictions
                    _, predicted = torch.max(pooled_outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Track per-class accuracy
                    for cls in range(NUM_CLASSES):
                        mask = (labels == cls)
                        if mask.sum().item() > 0:
                            class_correct[cls] += (predicted[mask] == cls).sum().item()
                            class_total[cls] += mask.sum().item()
                    
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
        
        # Calculate overall metrics
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct / total if total > 0 else 0
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}: Train Loss {avg_loss:.4f}, Val Loss {avg_val_loss:.4f}, Val Accuracy {val_accuracy:.2f}%")
        
        # Report per-class metrics
        for cls in range(NUM_CLASSES):
            if cls in class_counts and class_total[cls] > 0:
                cls_acc = 100 * class_correct[cls] / class_total[cls]
                print(f"  Class {cls} accuracy: {cls_acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")
        
        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Save plots for monitoring at each epoch
        save_epoch_plots(
            epoch, 
            train_losses, 
            val_losses, 
            val_accuracies, 
            class_correct,
            class_total,
            learning_rates,
            metrics_folder
        )
        
        print(f"Saved monitoring plots for epoch {epoch+1}")
        
        # Save model checkpoint periodically
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            checkpoint_path = os.path.join(model_folder, f"unet_model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Validation loss: {best_val_loss:.4f}")
            early_stop_counter = 0  # Reset counter when we find a better model
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")
            
        # Check for early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            # Save final model before stopping
            torch.save(model.state_dict(), model_filename)
            print(f"Saved final model to {model_filename}")
            break  # Exit the training loop
    
    # Save the final model if not stopped early
    if early_stop_counter < patience:
        torch.save(model.state_dict(), model_filename)
        print(f"Saved final model to {model_filename}")
        
    # If training completed, load the best model for evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path))

    test_total = 0
    test_loss = 0
    test_correct = 0
    
    # Test loop - update to unpack 3 values
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Testing"):  # Added _ to unpack class_distribution
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
    
    # Create a confusion matrix at the end of training to analyze performance
    print("\nGenerating final confusion matrix...")
    # Reset the confusion matrix before using it
    confmat.zero_()  # Clear any existing values
    
    # Confusion matrix generation - update to unpack 3 values
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="Confusion Matrix"):  # Added _ to unpack class_distribution
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            pooled_outputs = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)
            _, preds = torch.max(pooled_outputs, 1)
            for t, p in zip(labels, preds):
                confmat[t, p] += 1
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confmat)
    
    # Calculate per-class accuracy
    per_class_acc = confmat.diag().float() / confmat.sum(1).float() * 100
    for i in range(NUM_CLASSES):
        if i in class_counts:
            print(f"Class {i} accuracy: {per_class_acc[i]:.2f}%")
    
    # Clear final memory before returning
    torch.cuda.empty_cache()
    gc.collect()
    
    return model, train_losses, val_losses, val_accuracies, test_accuracy
