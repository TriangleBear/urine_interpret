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
    # Resize outputs to save memory if necessary
    original_size = outputs.shape[2:]
    if outputs.shape[2] > max_size or outputs.shape[3] > max_size:
        scale_factor = max_size / max(outputs.shape[2], outputs.shape[3])
        new_h, new_w = int(outputs.shape[2] * scale_factor), int(outputs.shape[3] * scale_factor)
        outputs = F.interpolate(outputs, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
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
    
    # Debug info
    # print(f"Targets shape before processing: {targets.shape}")

    
    # Handle polygon-shaped bounding boxes for class 10
    if targets.dim() == 4 and targets.shape[1] == 1 and targets[0, 0, 0, 0] == 10:
        # Process polygon-shaped bounding boxes
        batch_size = outputs.shape[0]
        height, width = outputs.shape[2], outputs.shape[3]
        polygon_targets = targets.view(batch_size, -1, 2)
        new_targets = torch.zeros((batch_size, 1, height, width), dtype=torch.long, device=targets.device)
        
        for i in range(batch_size):
            polygon_points = polygon_targets[i].cpu().detach().numpy().reshape(-1, 2)
            polygon_points[:, 0] *= width
            polygon_points[:, 1] *= height
            polygon_points = polygon_points.astype(np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_points], 10)
            new_targets[i, 0] = torch.from_numpy(mask).to(targets.device)
        
        targets = new_targets
    
    # Debug info
    # print(f"Targets shape after processing: {targets.shape}")

    
    # Resize targets to match outputs size
    if targets.shape[2:] != outputs.shape[2:]:
        targets = F.interpolate(targets.float(), size=outputs.shape[2:], mode='nearest').long()
    
    # Debug info
    # print(f"Targets shape before one-hot: {targets.shape}")

    
    # Fix any invalid target indices
    targets = fix_target_indices(targets, outputs.shape[1])
    
    # Convert to one-hot encoding more efficiently
    targets = targets.squeeze(1)  # Remove channel dim for one-hot
    targets_one_hot = F.one_hot(targets.long(), num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
    
    # Debug info
    # print(f"Outputs shape: {outputs.shape}, Targets one-hot shape: {targets_one_hot.shape}")

    
    # Compute Focal Loss in batches to save memory
    batch_size = outputs.shape[0]
    loss_sum = 0
    
    for i in range(batch_size):
        # Process one sample at a time
        output_i = outputs[i:i+1]
        target_i = targets_one_hot[i:i+1]
        
        bce_loss = F.binary_cross_entropy_with_logits(output_i, target_i, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss_i = (alpha * (1 - pt) ** gamma * bce_loss).mean()
        loss_sum += focal_loss_i
        
        # Clean up to save memory
        del output_i, target_i, bce_loss, pt, focal_loss_i
    
    # Clean up memory
    del outputs, targets, targets_one_hot
    torch.cuda.empty_cache()
    gc.collect()
    
    return loss_sum / batch_size
