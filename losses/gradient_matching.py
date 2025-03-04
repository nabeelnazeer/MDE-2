import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientLoss(nn.Module):
    """
    Gradient matching loss that compares image gradients between prediction and target
    """
    def __init__(self):
        super().__init__()
        # Initialize Sobel filters on CPU first
        self.sobel_x_filter = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3)
        
        self.sobel_y_filter = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3)
        
        # Select appropriate device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
            
        self.device = device
        self.sobel_x_filter = self.sobel_x_filter.to(device)
        self.sobel_y_filter = self.sobel_y_filter.to(device)
    
    def to(self, device):
        self.device = device
        self.sobel_x_filter = self.sobel_x_filter.to(device)
        self.sobel_y_filter = self.sobel_y_filter.to(device)
        return super().to(device)
    
    def forward(self, pred, target):
        # Ensure inputs are float and contiguous
        pred = pred.float().contiguous()
        target = target.float().contiguous()
        
        # Create valid mask
        mask = (target > 0).float()
        
        # Compute gradients for prediction
        pred_grad_x = F.conv2d(
            pred, 
            self.sobel_x_filter.expand(pred.size(1), -1, -1, -1), 
            padding=1, 
            groups=pred.size(1)
        )
        pred_grad_y = F.conv2d(
            pred, 
            self.sobel_y_filter.expand(pred.size(1), -1, -1, -1), 
            padding=1, 
            groups=pred.size(1)
        )
        
        # Compute gradients for target
        target_grad_x = F.conv2d(
            target, 
            self.sobel_x_filter.expand(target.size(1), -1, -1, -1), 
            padding=1, 
            groups=target.size(1)
        )
        target_grad_y = F.conv2d(
            target, 
            self.sobel_y_filter.expand(target.size(1), -1, -1, -1), 
            padding=1, 
            groups=target.size(1)
        )
        
        # Apply mask
        pred_grad_x = pred_grad_x * mask
        pred_grad_y = pred_grad_y * mask
        target_grad_x = target_grad_x * mask
        target_grad_y = target_grad_y * mask
        
        # Compute loss
        loss = torch.mean(torch.abs(pred_grad_x - target_grad_x) + 
                         torch.abs(pred_grad_y - target_grad_y))
        
        return loss

class MultiScaleGradientMatchingLoss(nn.Module):
    """
    Multi-scale gradient matching loss
    """
    def __init__(self, scales=4, weights=None):
        super(MultiScaleGradientMatchingLoss, self).__init__()
        self.scales = scales
        self.gradient_loss = GradientLoss()
        
        if weights is None:
            # Default weights that give more importance to finer scales
            self.weights = [1.0 / (2 ** i) for i in range(scales)]
            # Normalize weights
            self.weights = [w / sum(self.weights) for w in self.weights]
        else:
            assert len(weights) == scales, "Number of weights must match number of scales"
            self.weights = weights
            
    def forward(self, pred_features, target, mask=None):
        """
        pred_features: list of feature maps at different scales from the student model
        target: ground truth depth map
        mask: optional mask for valid pixels
        """
        total_loss = 0.0
        
        for i, (pred_feature, weight) in enumerate(zip(pred_features, self.weights)):
            # Get prediction at this scale
            if pred_feature.shape[2:] != target.shape[2:]:
                # Resize to match target
                pred_feature = F.interpolate(
                    pred_feature, size=target.shape[2:], 
                    mode='bilinear', align_corners=False
                )
            
            # Calculate gradient matching loss at this scale
            loss = self.gradient_loss(pred_feature, target, mask)
            total_loss += weight * loss
            
        return total_loss
