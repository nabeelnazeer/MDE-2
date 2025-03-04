import torch
import torch.nn as nn
from .scale_shift_invariant import ScaleShiftInvariantLoss
from .gradient_matching import GradientLoss

class CombinedLoss(nn.Module):
    def __init__(self, ssi_weight=1.0, gradient_weight=0.5, ssi_alpha=0.5, ssi_scales=4):
        super().__init__()
        self.ssi_loss = ScaleShiftInvariantLoss(alpha=ssi_alpha, scales=ssi_scales)
        self.gradient_loss = GradientLoss()
        self.ssi_weight = ssi_weight
        self.gradient_weight = gradient_weight
    
    def forward(self, pred, target):
        # Ensure inputs are float32 and contiguous
        pred = pred.float().contiguous()
        target = target.float().contiguous()
        
        # Compute individual losses with error handling
        try:
            loss_ssi = self.ssi_loss(pred, target)
            loss_gradient = self.gradient_loss(pred, target)
            
            # Check for NaN losses
            if torch.isnan(loss_ssi):
                loss_ssi = torch.tensor(0.0, device=pred.device)
            if torch.isnan(loss_gradient):
                loss_gradient = torch.tensor(0.0, device=pred.device)
            
            # Combine losses
            total_loss = self.ssi_weight * loss_ssi + self.gradient_weight * loss_gradient
            
            return {
                'loss': total_loss,
                'ssi': loss_ssi,
                'gradient': loss_gradient
            }
            
        except RuntimeError as e:
            print(f"Error computing loss: {str(e)}")
            return {
                'loss': torch.tensor(0.0, device=pred.device),
                'ssi': torch.tensor(0.0, device=pred.device),
                'gradient': torch.tensor(0.0, device=pred.device)
            }
