# train_unet.py
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from config import *
from models import UNet
from datasets import UrineStripDataset
from losses import dice_loss, focal_loss

def train_unet(batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS, patience=PATIENCE):
    # Dataset and DataLoader
    dataset = UrineStripDataset(IMAGE_FOLDER, MASK_FOLDER)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model and Optimizer
    model = UNet(3, NUM_CLASSES, dropout_prob=0.3).to(device)  # Increase dropout rate
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device=device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)  # Add learning rate scheduler

    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop
    best_loss = float('inf')
    early_stop_counter = 0
    model_filename = get_model_filename()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for i, (images, masks) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                focal_loss_value = focal_loss(outputs, masks)
                dice_loss_value = dice_loss(outputs, masks)
                loss = 0.3 * focal_loss_value + 0.7 * dice_loss_value  # Combine custom losses
                loss = loss.mean()  # Ensure the loss is a scalar
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            torch.cuda.empty_cache()  # Clear cache after each batch

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        val_loss = 0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images.to(device))
                dice_loss_value = dice_loss(outputs, masks.to(device))
                val_loss += dice_loss_value.mean().item()  # Ensure the loss is a scalar
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.to(device)  # Ensure predicted is on the same device as masks
                total += masks.numel()
                correct += (predicted == masks).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}: Train Loss {avg_loss:.4f}, Val Loss {avg_val_loss:.4f}, Val Accuracy {val_accuracy:.2f}%")
        
        # Adjust learning rate
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_filename)
            print("Model improved and saved.")
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} epochs.")
        
        # Check early stopping
        if early_stop_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break
    
    return model, train_losses, val_losses, val_accuracies