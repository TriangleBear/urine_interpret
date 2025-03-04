import torch
import numpy as np
import logging
from tqdm import tqdm  # Import tqdm for progress bars

from torch.utils.data import DataLoader
from datasets import UrineStripDataset
from models import UNetYOLO
from losses import dice_loss, focal_loss
from config import TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available

def train_model(num_epochs=10, batch_size=16, learning_rate=0.001):    
    logging.basicConfig(level=logging.INFO)  # Set up logging configuration

    # Load dataset
    dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer
    model = UNetYOLO(in_channels=3, out_channels=NUM_CLASSES).to(device)  # Move model to GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create a tqdm progress bar for epochs
    epoch_progress = tqdm(range(num_epochs), desc="Training epochs", position=0)
    
    for epoch in epoch_progress:
        model.train()
        epoch_loss = 0.0
        
        # Create a tqdm progress bar for batches
        batch_progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                               position=1, leave=False, total=len(dataloader))
        
        for images, targets, _ in batch_progress:
            images, targets = images.to(device), targets.to(device)  # Move to GPU
            try:
                outputs = model(images)
                loss = dice_loss(outputs, targets) + focal_loss(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update loss and progress bar description
                current_loss = loss.item()
                epoch_loss += current_loss
                batch_progress.set_postfix({"Loss": f"{current_loss:.4f}"})
                
            except Exception as e:
                print(f"Error during training at epoch {epoch+1}: {e}")
                continue  # Skip this batch and continue with the next
                
        # Update epoch progress bar with average loss
        avg_loss = epoch_loss / len(dataloader)
        epoch_progress.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')  # Log training progress
    
    # Save the trained model
    model_save_path = 'trained_model.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining complete! Model saved to {model_save_path}")
    
    return model
