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

class MultiScaleScaleShiftInvariantLoss(nn.Module):
    """Multi-scale version of scale-shift invariant loss"""
    def __init__(self, alpha=0.5, scales=4, weights=None):
        super().__init__()
        self.scales = scales
        self.alpha = alpha
        
        # Initialize weights for different scales
        if weights is None:
            # Default weights decrease by factor of 2 for each scale
            self.weights = [1.0 / (2 ** i) for i in range(scales)]
            # Normalize weights
            self.weights = [w / sum(self.weights) for w in self.weights]
        else:
            assert len(weights) == scales, "Number of weights must match number of scales"
            self.weights = weights
    
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
        current_pred = pred
        current_target = target
        current_mask = mask
        
        # Compute loss at each scale
        for scale, weight in enumerate(self.weights):
            # Apply mask
            pred_masked = current_pred * current_mask
            target_masked = current_target * current_mask
            
            # Compute log difference
            diff = torch.log(pred_masked + 1e-6) - torch.log(target_masked + 1e-6)
            diff = diff * current_mask
            
            # Count valid pixels
            num_valid = torch.sum(current_mask) + 1e-8
            
            # Compute scale and shift invariant loss at current scale
            diff_sum = torch.sum(diff) / num_valid
            diff_sqr_sum = torch.sum(diff ** 2) / num_valid
            loss = diff_sqr_sum - self.alpha * (diff_sum ** 2)
            
            # Add weighted loss for this scale
            total_loss += weight * loss
            
            # Downsample for next scale
            if scale < self.scales - 1:
                size = (current_pred.shape[2] // 2, current_pred.shape[3] // 2)
                current_pred = F.interpolate(current_pred, size=size, mode='bilinear', align_corners=True)
                current_target = F.interpolate(current_target, size=size, mode='bilinear', align_corners=True)
                current_mask = F.interpolate(current_mask, size=size, mode='nearest')
        
        return total_loss
