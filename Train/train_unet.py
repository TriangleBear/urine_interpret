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

def train_unet(batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS):
    # Dataset and DataLoader
    dataset = UrineStripDataset(IMAGE_FOLDER, MASK_FOLDER)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model and Optimizer
    model = UNet(3, NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device=device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for i, (images, masks) in enumerate(tqdm(train_loader)):
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

        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                outputs = model(images.to(device))
                val_loss += dice_loss(outputs, masks.to(device)).item()
        
        avg_loss = epoch_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss {avg_loss:.4f}, Val Loss {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), get_model_filename())
    
    return model