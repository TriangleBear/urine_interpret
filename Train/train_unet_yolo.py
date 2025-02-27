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

# Adding CLASS_NAMES definition at the module level (first line of code in the file)
CLASS_NAMES = {0: 'Bilirubin', 1: 'Blood', 2: 'Glucose', 3: 'Ketone', 4: 'Leukocytes', 5: 'Nitrite', 6: 'Protein', 7: 'SpGravity', 8: 'Urobilinogen', 9: 'pH', 10: 'strip'}

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

    # Check class distribution in train dataset
    train_labels = [label for _, label in train_dataset]
    label_counts = {i: train_labels.count(i) for i in range(NUM_CLASSES)}
    print(f"Training set class distribution:")
    for class_id, count in label_counts.items():
        class_name = CLASS_NAMES.get(class_id, f"Class-{class_id}")
        print(f"  Class {class_id} ({class_name}): {count} samples ({count/len(train_labels)*100:.2f}%)")
    
    # Debugging: Print the first few samples to verify labels
    for i in range(min(5, len(train_dataset))):
        image, label = train_dataset[i]
        print(f"Sample {i}: Label {label}")
    
    # Calculate class weights based on inverse frequency
    total_samples = len(train_dataset)
    class_weights = []
    for class_id in range(NUM_CLASSES):
        count = label_counts.get(class_id, 0)
        if count > 0:
            # Use stronger inverse frequency weighting to emphasize minority classes
            weight = total_samples / (count * NUM_CLASSES)
            # Cap weights to prevent numerical instability, but use higher cap for rare classes
            weight = min(weight, 25.0)  # Increased from 10.0 to 25.0
        else:
            weight = 15.0  # Default weight for classes with no samples
        class_weights.append(weight)
    
    # Further boost weights of non-strip classes (0-9) to fight class 10 dominance
    for class_id in range(NUM_CLASSES - 1):  # All classes except strip (class 10)
        class_weights[class_id] *= 2.0  # Double the weight of non-strip classes
    
    class_weights_tensor = torch.tensor(class_weights, device=device)
    print(f"Class weights: {class_weights_tensor}")

    # Model initialization with memory optimization
    model = UNetYOLO(3, NUM_CLASSES, dropout_prob=0.2).to(device)  # Reduce dropout further
    
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
        # Use class weights in loss function to address class imbalance
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights_tensor,  # Use calculated weights
            label_smoothing=0.1,  # Increased from 0.05 to 0.1 for better regularization
            reduction='mean'
        )
    
    # Train the entire model from the beginning
    model.train()
    
    # Use a higher learning rate for faster convergence
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scaler = GradScaler()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

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
    
    # Three-stage training strategy:
    # 1. Train on classes 0-9 only (reagent pads)
    # 2. Binary classification (strip vs non-strip)
    # 3. Full multi-class with all classes
    
    print("\n=== Stage 1: Reagent Pads Only Training (Classes 0-9) ===")
    
    # Create a filtered dataset containing only reagent pad classes (0-9)
    def filter_reagent_pad_classes(dataset):
        filtered_dataset = []
        for image, label in dataset:
            # Only include classes 0-9 (reagent pads)
            if label < 10:  
                filtered_dataset.append((image, label))
        return filtered_dataset
    
    reagent_train_dataset = filter_reagent_pad_classes(train_dataset)
    reagent_val_dataset = filter_reagent_pad_classes(val_dataset)
    
    # Check if we have enough samples
    if len(reagent_train_dataset) < 10:
        print("WARNING: Not enough reagent pad samples for Stage 1. Skipping to Stage 2.")
    else:
        # Create data loaders for reagent pad classes
        reagent_train_loader = DataLoader(
            reagent_train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        reagent_val_loader = DataLoader(
            reagent_val_dataset, 
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Create model for reagent pad classes only (10 output classes)
        reagent_model = UNetYOLO(3, 10, dropout_prob=0.2).to(device)
        
        # Class weights for reagent pad classes
        reagent_class_weights = class_weights[:10]  # Take only the first 10 class weights
        reagent_class_weights_tensor = torch.tensor(reagent_class_weights, device=device)
        
        # Loss function for reagent pad classes
        reagent_criterion = torch.nn.CrossEntropyLoss(
            weight=reagent_class_weights_tensor,
            label_smoothing=0.1,
            reduction='mean'
        )
        
        # Higher learning rate for reagent pad training
        reagent_optimizer = torch.optim.Adam(reagent_model.parameters(), lr=0.001)
        reagent_scheduler = CosineAnnealingWarmRestarts(reagent_optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        reagent_scaler = GradScaler()
        
        # Train reagent pad model for fewer epochs
        reagent_epochs = 15
        best_reagent_acc = 0.0
        reagent_model_path = os.path.join(model_folder, "reagent_model.pt")
        
        # Training loop for reagent pad classes
        for epoch in range(reagent_epochs):
            reagent_model.train()
            epoch_loss = 0
            reagent_optimizer.zero_grad(set_to_none=True)
            
            with tqdm(total=len(reagent_train_loader), desc=f"Reagent Training Epoch {epoch+1}") as pbar:
                for i, (images, labels) in enumerate(reagent_train_loader):
                    try:
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        
                        # Apply normalization
                        if torch.max(images) > 1.0:
                            images = images / 255.0
                        images = dynamic_normalization(images)
                        
                        # Resize images if needed
                        if images.shape[2] > max_size or images.shape[3] > max_size:
                            scale_factor = max_size / max(images.shape[2], images.shape[3])
                            new_h, new_w = int(images.shape[2] * scale_factor), int(images.shape[3] * scale_factor)
                            images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
                        
                        with autocast(device_type="cuda", dtype=torch.float16):
                            outputs = reagent_model(images)
                            labels = labels.long()
                            loss = reagent_criterion(outputs, labels)
                        
                        loss = loss / accumulation_steps
                        reagent_scaler.scale(loss).backward()
                        
                        if (i + 1) % accumulation_steps == 0:
                            reagent_scaler.unscale_(reagent_optimizer)
                            torch.nn.utils.clip_grad_norm_(reagent_model.parameters(), max_norm=0.5)
                            reagent_scaler.step(reagent_optimizer)
                            reagent_scaler.update()
                            reagent_optimizer.zero_grad(set_to_none=True)
                        
                        # Display batch accuracy
                        if i % 10 == 0:
                            _, predicted = torch.max(outputs, 1)
                            accuracy = (predicted == labels).float().mean() * 100
                            pbar.set_postfix({'batch_acc': f"{accuracy.item():.2f}%"})
                        
                        epoch_loss += loss.item() * accumulation_steps
                        pbar.update(1)
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"OOM error in batch {i}, skipping...")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
            
            avg_loss = epoch_loss / len(reagent_train_loader)
            print(f"Reagent Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
            
            # Validation for reagent pad classes
            reagent_model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(reagent_val_loader, desc="Reagent Validation"):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    outputs = reagent_model(images)
                    _, predicted = torch.max(outputs, 1)
                    
                    loss = reagent_criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            # Ensure we don't divide by zero
            if len(reagent_val_loader) > 0:
                val_accuracy = 100 * correct / total if total > 0 else 0
                avg_val_loss = val_loss / len(reagent_val_loader)
                print(f"Reagent Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
            else:
                val_accuracy = 0
                avg_val_loss = 0
                print("Reagent Validation: No samples available for validation.")
            
            reagent_scheduler.step()
            
            # Save the best reagent model
            if val_accuracy > best_reagent_acc:
                best_reagent_acc = val_accuracy
                torch.save(reagent_model.state_dict(), reagent_model_path)
                print(f"Reagent model saved with accuracy {val_accuracy:.2f}%")
    
    # Continue with Stage 2: Binary Classification (Strip vs Non-Strip)
    print("\n=== Stage 2: Binary Classification (Strip vs Non-Strip) ===")
    
    # Create a binary version of the dataset
    def create_binary_labels(dataset):
        binary_dataset = []
        for image, label in dataset:
            # Convert to binary: 0 for reagent pads (classes 0-9), 1 for strip (class 10)
            binary_label = 1 if label == 10 else 0
            binary_dataset.append((image, binary_label))
        return binary_dataset
    
    binary_train_dataset = create_binary_labels(train_dataset)
    binary_val_dataset = create_binary_labels(val_dataset)
    
    # Create new data loaders for binary classification
    binary_train_loader = DataLoader(
        binary_train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    binary_val_loader = DataLoader(
        binary_val_dataset, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create binary model (2 output classes: non-strip and strip)
    binary_model = UNetYOLO(3, 2, dropout_prob=0.2).to(device)
    
    # Simple CrossEntropyLoss for binary classification
    binary_criterion = torch.nn.CrossEntropyLoss()
    
    # Use higher learning rate for binary task
    binary_optimizer = torch.optim.Adam(binary_model.parameters(), lr=0.001)
    binary_scheduler = CosineAnnealingWarmRestarts(binary_optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    binary_scaler = GradScaler()
    
    # Train binary model for fewer epochs
    binary_epochs = 20
    best_binary_acc = 0.0
    binary_model_path = os.path.join(model_folder, "binary_model.pt")
    
    for epoch in range(binary_epochs):
        # Training
        binary_model.train()
        epoch_loss = 0
        binary_optimizer.zero_grad(set_to_none=True)
        
        with tqdm(total=len(binary_train_loader), desc=f"Binary Training Epoch {epoch+1}") as pbar:
            for i, (images, labels) in enumerate(binary_train_loader):
                try:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    # Apply normalization
                    if torch.max(images) > 1.0:
                        images = images / 255.0
                    images = dynamic_normalization(images)
                    
                    # Resize images if needed
                    if images.shape[2] > max_size or images.shape[3] > max_size:
                        scale_factor = max_size / max(images.shape[2], images.shape[3])
                        new_h, new_w = int(images.shape[2] * scale_factor), int(images.shape[3] * scale_factor)
                        images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    
                    with autocast(device_type="cuda", dtype=torch.float16):
                        outputs = binary_model(images)
                        labels = labels.long()
                        loss = binary_criterion(outputs, labels)
                    
                    loss = loss / accumulation_steps
                    binary_scaler.scale(loss).backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        binary_scaler.unscale_(binary_optimizer)
                        torch.nn.utils.clip_grad_norm_(binary_model.parameters(), max_norm=0.5)
                        binary_scaler.step(binary_optimizer)
                        binary_scaler.update()
                        binary_optimizer.zero_grad(set_to_none=True)
                    
                    # Display batch accuracy
                    if i % 10 == 0:
                        _, predicted = torch.max(outputs, 1)
                        accuracy = (predicted == labels).float().mean() * 100
                        pbar.set_postfix({'batch_acc': f"{accuracy.item():.2f}%"})
                    
                    epoch_loss += loss.item() * accumulation_steps
                    pbar.update(1)
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error in batch {i}, skipping...")
                        torch.cuda.empty_cache()
                    else:
                        raise e
        
        avg_loss = epoch_loss / len(binary_train_loader)
        print(f"Binary Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
        
        # Validation
        binary_model.eval()
        val_correct = 0
        val_total = 0
        strip_correct = 0
        strip_total = 0
        nonstrip_correct = 0
        nonstrip_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(binary_val_loader, desc="Binary Validation"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = binary_model(images)
                _, predicted = torch.max(outputs, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Separate accuracy for strip vs non-strip
                strip_mask = (labels == 1)
                nonstrip_mask = (labels == 0)
                
                strip_total += strip_mask.sum().item()
                nonstrip_total += nonstrip_mask.sum().item()
                
                strip_correct += (strip_mask & (predicted == 1)).sum().item()
                nonstrip_correct += (nonstrip_mask & (predicted == 0)).sum().item()
        
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
        strip_accuracy = 100 * strip_correct / strip_total if strip_total > 0 else 0
        nonstrip_accuracy = 100 * nonstrip_correct / nonstrip_total if nonstrip_total > 0 else 0
        
        print(f"Binary Validation Accuracy: {val_accuracy:.2f}%")
        print(f"  - Strip Class: {strip_accuracy:.2f}% ({strip_correct}/{strip_total})")
        print(f"  - Non-Strip Classes: {nonstrip_accuracy:.2f}% ({nonstrip_correct}/{nonstrip_total})")
        
        binary_scheduler.step()
        
        # Save best binary model
        if val_accuracy > best_binary_acc:
            best_binary_acc = val_accuracy
            torch.save(binary_model.state_dict(), binary_model_path)
            print(f"Binary model saved with accuracy {val_accuracy:.2f}%")
    
    print("\n=== Stage 3: Multi-class Classification (All Classes) ===")
    
    # Load the best binary model to initialize multi-class model
    model = UNetYOLO(3, NUM_CLASSES, dropout_prob=0.2).to(device)
    model_state_dict = model.state_dict()
    
    # Transfer weights from binary model
    binary_state_dict = torch.load(binary_model_path)
    
    # Also try to load weights from reagent model if available
    try:
        if os.path.exists(reagent_model_path):
            reagent_state_dict = torch.load(reagent_model_path)
            print("Transferring weights from reagent model...")
            
            # First transfer weights from reagent model (for classes 0-9 recognition)
            for name, param in reagent_state_dict.items():
                if name in model_state_dict and "classifier" not in name:
                    if param.shape == model_state_dict[name].shape:
                        model_state_dict[name].copy_(param)
                        print(f"Transferred from reagent model: {name}")
            
            # Then transfer remaining weights from binary model (strip recognition)
            for name, param in binary_state_dict.items():
                if name in model_state_dict and "classifier" not in name and "yolo_head" not in name:
                    if param.shape == model_state_dict[name].shape:
                        # Use weighted average for overlapping features
                        model_state_dict[name].copy_(0.5 * param + 0.5 * model_state_dict[name])
                        print(f"Merged from binary model: {name}")
        else:
            # If reagent model doesn't exist, just use binary model weights
            print("Reagent model not found, using binary model weights only...")
            for name, param in binary_state_dict.items():
                if name in model_state_dict and "classifier" not in name:
                    if param.shape == model_state_dict[name].shape:
                        model_state_dict[name].copy_(param)
                        print(f"Transferred from binary model: {name}")
    except Exception as e:
        print(f"Error transferring weights: {e}")
        print("Starting with fresh weights")
    
    model.load_state_dict(model_state_dict)
    
    # Use a slightly smaller learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    
    # Resume with normal training for all classes
    scaler = GradScaler()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Continue with the rest of the training as before
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
                        # Get outputs from model - include segmentation map
                        outputs, segmentation_maps = model(images, return_segmentation=True)
                        
                        # Debug info in first batch
                        if i == 0 and epoch == 0:
                            print(f"Output shape: {outputs.shape}")
                            print(f"Segmentation map shape: {segmentation_maps.shape}")
                            print(f"Labels shape: {labels.shape}")
                            print(f"Labels: {labels}")
                        
                        # Convert labels to long for CrossEntropyLoss
                        labels = labels.long()
                        
                        # Cross entropy loss for classification
                        loss = criterion(outputs, labels)
                        
                        # Monitor predicted classes during training
                        if i % 50 == 0:  # Every 50 batches
                            _, predicted = torch.max(outputs, 1)
                            accuracy = (predicted == labels).float().mean() * 100
                            pbar.set_postfix({'batch_acc': f"{accuracy.item():.2f}%"})
                    
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
                    del images, labels, outputs
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
        class_correct = torch.zeros(NUM_CLASSES, device=device)
        class_total = torch.zeros(NUM_CLASSES, device=device)
        
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
                    
                    # Get outputs directly - with segmentation maps
                    outputs, segmentation_maps = model(images, return_segmentation=True)
                    
                    # No pooling needed
                    labels = labels.long()
                    
                    loss_val = criterion(outputs, labels)
                    val_loss += loss_val.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Calculate per-class accuracy
                    for c in range(NUM_CLASSES):
                        class_mask = (labels == c)
                        class_total[c] += class_mask.sum().item()
                        class_correct[c] += (class_mask & (predicted == c)).sum().item()
                    
                    # Free memory
                    del images, labels, outputs, predicted, loss_val
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
        
        # Print per-class accuracy for better diagnostics
        print("\nPer-class validation accuracy:")
        for c in range(NUM_CLASSES):
            if class_total[c] > 0:
                class_acc = 100 * class_correct[c] / class_total[c]
                print(f"Class {c}: {class_acc:.2f}% ({int(class_correct[c])}/{int(class_total[c])})")
            else:
                print(f"Class {c}: No samples")
        
        print(f"Epoch {epoch+1}: Train Loss {avg_loss:.4f}, Val Loss {avg_val_loss:.4f}, Val Accuracy {val_accuracy:.2f}%")
        
        # Adjust learning rate based on validation loss
        scheduler.step(epoch + avg_val_loss)

        # After validation, detect if the model is stuck predicting one class
        predicted_classes = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                predicted_classes.extend(preds.cpu().numpy())
        
        unique_predictions = set(predicted_classes)
        if len(unique_predictions) <= 2 and epoch > 3:  # Changed from 1 to 2 and from 5 to 3
            print(f"WARNING: Model is stuck predicting only {unique_predictions}. Implementing recovery strategy.")
            
            # More aggressive recovery strategy
            # Re-initialize model if it's stuck
            del model
            torch.cuda.empty_cache()
            
            # Create new model with different initialization and lower dropout
            model = UNetYOLO(3, NUM_CLASSES, dropout_prob=0.1).to(device)
            
            # Use SGD with momentum and higher learning rate
            optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
            
            # Temporarily increase class weights even further for minority classes
            temp_class_weights = class_weights.copy()
            for class_id in range(NUM_CLASSES - 1):  # All classes except strip (class 10)
                temp_class_weights[class_id] *= 3.0  # Triple weights temporarily
            
            temp_class_weights_tensor = torch.tensor(temp_class_weights, device=device)
            criterion = torch.nn.CrossEntropyLoss(
                weight=temp_class_weights_tensor,
                label_smoothing=0.0,  # Remove label smoothing temporarily
                reduction='mean'
            )
            
            # Reset early stopping counter
            early_stop_counter = 0
            
            # Apply batch sampling to focus on minority classes in the next few epochs
            print("Applying focused training on minority classes...")

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
                
                # Get segmentation maps only
                segmentation_maps = model(images, segmentation_only=True)
                segmentation_maps = torch.argmax(segmentation_maps, dim=1).cpu().numpy()
                
                # Extract features for SVM classification
                features_list = []
                for i in range(images.size(0)):
                    image_np = images[i].permute(1, 2, 0).cpu().numpy()
                    segmentation_map = segmentation_maps[i]
                    features = extract_features_from_segmentation(image_np, segmentation_map)
                    features_list.append(features)
                
                features = np.array(features_list)
                features_scaled = scaler.transform(features)
                
                # Predict using SVM
                predictions = svm_model.predict(features_scaled)
                
                # Calculate accuracy
                test_total += labels.size(0)
                test_correct += (predictions == labels.cpu().numpy()).sum()
                
                # Free memory
                del images, labels, segmentation_maps, features, features_scaled, predictions
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
    print(f"Accuracy: {test_accuracy:.2f}%")
    
    # Clear final memory before returning
    torch.cuda.empty_cache()
    gc.collect()
    
    return model, train_losses, val_losses, val_accuracies, test_accuracy
