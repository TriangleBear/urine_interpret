# train_unet.py
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from config import *
from models import UNet
from datasets import UrineStripDataset
from losses import dice_loss, focal_loss

def train_unet():
    # Dataset and DataLoader
    dataset = UrineStripDataset(IMAGE_FOLDER, MASK_FOLDER)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model and Optimizer
    model = UNet(3, NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    # Training loop
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = 0.3*focal_loss(outputs, masks) + 0.7*dice_loss(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        
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