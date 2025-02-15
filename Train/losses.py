import torch.nn.functional as F
import torch

def dice_loss(outputs, targets, smooth=1e-6):
    outputs = F.softmax(outputs, dim=1)
    targets = targets.squeeze(1)
    targets_one_hot = F.one_hot(targets.long(), outputs.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (outputs * targets_one_hot).sum(dim=(2,3))
    union = outputs.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
    return 1 - (2 * intersection + smooth) / (union + smooth)

def focal_loss(outputs, targets, alpha=0.25, gamma=2):
    targets = targets.squeeze(1)
    targets_one_hot = F.one_hot(targets, outputs.shape[1]).permute(0, 3, 1, 2).float()
    bce_loss = F.binary_cross_entropy_with_logits(outputs, targets_one_hot, reduction='none')
    pt = torch.exp(-bce_loss)
    return (alpha * (1 - pt) ** gamma * bce_loss).mean()