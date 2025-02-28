import torch.nn.functional as F
import torch
import numpy as np
import cv2
import gc

def fix_target_indices(targets, num_classes):
    """
    Fix target indices to ensure they are within the valid range.
    
    Args:
        targets: Tensor containing class indices
        num_classes: Number of valid classes
    
    Returns:
        Tensor with indices clamped to valid range
    """
    if targets.max() >= num_classes:
    # print(f"Fixing invalid target indices: max value {targets.max().item()} >= {num_classes}")

        # Option 1: Clamp to the last valid class
        targets = torch.clamp(targets, 0, num_classes-1)
        
        # Option 2: Use a mapping approach for better debugging
        value_counts = torch.bincount(targets.flatten())
    # print(f"Value counts after clamping: {value_counts}")

    
    return targets

def dice_loss(outputs, targets, smooth=1e-6, max_size=512):
    # Resize outputs to save memory if necessary
    original_size = outputs.shape[2:]
    if outputs.shape[2] > max_size or outputs.shape[3] > max_size:
        scale_factor = max_size / max(outputs.shape[2], outputs.shape[3])
        new_h, new_w = int(outputs.shape[2] * scale_factor), int(outputs.shape[3] * scale_factor)
        outputs = F.interpolate(outputs, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    outputs = F.softmax(outputs, dim=1)
    
    # Handle target dimensions
    if targets.dim() == 5:
        targets = targets.squeeze(1)
    elif targets.dim() == 2:  
        batch_size = outputs.shape[0]
        height, width = outputs.shape[2], outputs.shape[3]
        targets = targets.view(batch_size, height, width)
    elif targets.dim() == 3:
        targets = targets.unsqueeze(1)
    elif targets.dim() == 1:
        targets = targets.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    
    # Resize targets to match outputs size
    if targets.shape[2:] != outputs.shape[2:]:
        targets = F.interpolate(targets.float(), size=outputs.shape[2:], mode='nearest').long()
    
    # Fix any invalid target indices
    targets = fix_target_indices(targets, outputs.shape[1])
    
    # Convert to one-hot encoding
    targets = targets.squeeze(1)  # Remove channel dim for one-hot
    targets_one_hot = F.one_hot(targets.long(), num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
    
    # Compute dice loss in batches to save memory
    batch_size = outputs.shape[0]
    dice_sum = 0
    
    for i in range(batch_size):
        # Process one sample at a time
        output_i = outputs[i:i+1]
        target_i = targets_one_hot[i:i+1]
        
        intersection = (output_i * target_i).sum(dim=(2,3))
        union = output_i.sum(dim=(2,3)) + target_i.sum(dim=(2,3))
        dice_i = 1 - (2 * intersection + smooth) / (union + smooth)
        dice_sum += dice_i.mean()
        
        # Clean up to save memory
        del output_i, target_i, intersection, union, dice_i
    
    # Clean up memory
    del outputs, targets, targets_one_hot
    torch.cuda.empty_cache()
    gc.collect()
    
    return dice_sum / batch_size

def focal_loss(outputs, targets, alpha=0.25, gamma=2, max_size=512):
    """
    Compute focal loss for multi-class segmentation with support for class indices.
    
    Args:
        outputs: Model predictions [B, C, H, W]
        targets: Ground truth labels [B] or [B, H, W]
        alpha: Weighting factor for rare classes
        gamma: Focusing parameter
        max_size: Maximum size for resizing to save memory
    """
    # Resize outputs to save memory if necessary
    original_size = outputs.shape[2:]
    if outputs.shape[2] > max_size or outputs.shape[3] > max_size:
        scale_factor = max_size / max(outputs.shape[2], outputs.shape[3])
        new_h, new_w = int(outputs.shape[2] * scale_factor), int(outputs.shape[3] * scale_factor)
        outputs = F.interpolate(outputs, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Get dimensions
    batch_size = outputs.shape[0]
    num_classes = outputs.shape[1]
    height, width = outputs.shape[2], outputs.shape[3]
    
    # Handle target dimensions - support both segmentation masks and class indices
    if targets.dim() == 1:  # [B] - class indices
        # Convert class indices to one-hot encoded format
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()  # [B, C]
        
        # Compute binary cross entropy loss
        outputs_pooled = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)  # [B, C]
        bce_loss = F.binary_cross_entropy_with_logits(outputs_pooled, targets_one_hot, reduction='none')
        
    else:  # Handle segmentation masks [B, H, W] or [B, 1, H, W]
        # Handle 3D or 4D targets
        if targets.dim() == 4:  # [B, 1, H, W]
            targets = targets.squeeze(1)  # Convert to [B, H, W]
        
        # Resize targets to match outputs size if needed
        if targets.shape[1:] != outputs.shape[2:]:
            targets = F.interpolate(targets.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').squeeze(1).long()
        
        # Fix any invalid target indices
        targets = fix_target_indices(targets, num_classes)
        
        # Convert to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Compute binary cross entropy loss per pixel
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets_one_hot, reduction='none')
        bce_loss = bce_loss.mean(dim=(2, 3))  # Average over spatial dimensions
    
    # Apply focal loss weighting
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    
    # Average over batch and classes
    focal_loss = focal_loss.mean()
    
    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()
    
    return focal_loss
