import torch.nn.functional as F
import torch

def dice_loss(outputs, targets, smooth=1e-6):
    outputs = F.softmax(outputs, dim=1)
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets_one_hot = F.one_hot(targets.long(), outputs.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (outputs * targets_one_hot).sum(dim=(2,3))
    union = outputs.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
    dice = 1 - (2 * intersection + smooth) / (union + smooth)
    return dice.mean()  # Ensure the loss is a scalar

def focal_loss(outputs, targets, alpha=0.25, gamma=2):
    # Ensure targets have the right shape before one-hot encoding
    if targets.dim() == 4:
        targets = targets.squeeze(1)  # Remove extra channel dim if exists
    elif targets.dim() == 2:  
        # If targets are (B, H * W), reshape it to (B, H, W)
        batch_size = outputs.shape[0]
        height, width = outputs.shape[2], outputs.shape[3]
        targets = targets.view(batch_size, height, width)
    elif targets.dim() == 3:
        targets = targets.unsqueeze(1)  # Convert (B, H, W) to (B, 1, H, W)
    elif targets.dim() == 1:
        # If targets are (B,), reshape it to (B, 1, 1, 1)
        targets = targets.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    # Ensure targets have 4 dimensions before one-hot encoding
    if targets.dim() != 4:
        raise ValueError(f"Expected targets to have 4 dimensions, but got {targets.dim()} dimensions")

    # Convert to one-hot encoding
    targets_one_hot = F.one_hot(targets.long(), num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
    
    # Compute Focal Loss
    bce_loss = F.binary_cross_entropy_with_logits(outputs, targets_one_hot, reduction='none')
    pt = torch.exp(-bce_loss)
    return (alpha * (1 - pt) ** gamma * bce_loss).mean()  # Ensure the loss is a scalar
