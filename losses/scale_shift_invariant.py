import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4):
        super().__init__()
        self.alpha = alpha
        self.scales = scales
        
    def forward(self, pred, target):
        """
        pred: predicted depth (B, 1, H, W)
        target: ground truth depth (B, 1, H, W)
        """
        # Ensure float tensors and make contiguous
        pred = pred.float().contiguous()
        target = target.float().contiguous()
        
        # Create valid mask
        mask = (target > 0).float()
        
        total_loss = 0
        weight = 1.0
        current_pred = pred
        current_target = target
        current_mask = mask
        
        for scale in range(self.scales):
            # Apply mask
            pred_masked = current_pred * current_mask
            target_masked = current_target * current_mask
            
            # Compute log difference
            diff = torch.log(pred_masked + 1e-6) - torch.log(target_masked + 1e-6)
            diff = diff * current_mask
            
            # Count valid pixels
            num_valid = torch.sum(current_mask) + 1e-8
            
            # Compute scale and shift invariant loss
            diff_sum = torch.sum(diff) / num_valid
            diff_sqr_sum = torch.sum(diff ** 2) / num_valid
            loss = diff_sqr_sum - self.alpha * (diff_sum ** 2)
            
            total_loss += weight * loss
            weight /= 2.0
            
            # Downsample for next scale using interpolation instead of pooling
            if scale < self.scales - 1:
                size = (current_pred.shape[2] // 2, current_pred.shape[3] // 2)
                current_pred = F.interpolate(current_pred, size=size, mode='bilinear', align_corners=True)
                current_target = F.interpolate(current_target, size=size, mode='bilinear', align_corners=True)
                current_mask = F.interpolate(current_mask, size=size, mode='nearest')
        
        return total_loss
