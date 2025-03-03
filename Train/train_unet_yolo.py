import torch
import numpy as np
import logging  # Import logging library

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

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, targets, _ in dataloader:
            images, targets = images.to(device), targets.to(device)  # Move to GPU
            try:
                outputs = model(images)
                loss = dice_loss(outputs, targets) + focal_loss(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            except Exception as e:
                print(f"Error during training at epoch {epoch+1}: {e}")
                continue  # Skip this batch and continue with the next

        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}')  # Log training progress
