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
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    if targets.dim() == 2:
        targets = targets.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    print(f"After processing: targets.shape = {targets.shape}")
    targets_one_hot = F.one_hot(targets.long(), num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
    bce_loss = F.binary_cross_entropy_with_logits(outputs, targets_one_hot, reduction='none')
    pt = torch.exp(-bce_loss)
    return (alpha * (1 - pt) ** gamma * bce_loss).mean()  # Ensure the loss is a scalar