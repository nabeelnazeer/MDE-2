import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

class DynoV2Teacher(nn.Module):
    """Teacher model based on DINOv2"""
    def __init__(self, pretrained=True, freeze_encoder=False, arch='vit_base'):
        super().__init__()
        
        # Map architecture names to DINOv2 model names
        arch_map = {
            'vit_small': 'dinov2_vits14',
            'vit_base': 'dinov2_vitb14',
            'vit_large': 'dinov2_vitl14',
            'vit_giant': 'dinov2_vitg14'
        }
        
        # Get correct model name
        model_name = arch_map.get(arch)
        if model_name is None:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Load pretrained DINOv2 encoder
        self.encoder = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # Get encoder dimensions
        encoder_dims = {
            'vit_small': 384,
            'vit_base': 768,
            'vit_large': 1024,
            'vit_giant': 1536
        }
        encoder_dim = encoder_dims[arch]
        
        # Create depth head with proper upsampling
        self.depth_head = nn.Sequential(
            nn.Conv2d(encoder_dim, encoder_dim // 2, 3, padding=1),
            nn.BatchNorm2d(encoder_dim // 2),
            nn.GELU(),
            nn.Conv2d(encoder_dim // 2, encoder_dim // 4, 3, padding=1),
            nn.BatchNorm2d(encoder_dim // 4),
            nn.GELU(),
            nn.Conv2d(encoder_dim // 4, 1, 1)
        )
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Add depth range parameters
        self.min_depth = 1e-3
        self.max_depth = 80.0
    
    def forward(self, x):
        # Verify input dimensions
        B, C, H, W = x.shape
        assert H == W == 224, f"Input size must be 224x224, got {H}x{W}"
        
        # Get features from encoder
        features_dict = self.encoder.forward_features(x)
        features = features_dict['x_norm_patchtokens']  # Shape: [B, N, D]
        
        # Calculate proper dimensions
        B, N, D = features.shape
        P = int(math.sqrt(N))  # Should be 16 for 224x224 input with patch_size=14
        
        # Reshape features to spatial form - use reshape instead of view
        features = features.permute(0, 2, 1).contiguous()  # [B, D, N]
        features = features.reshape(B, D, P, P)  # [B, D, 16, 16]
        
        # Process through depth head and normalize
        depth = self.depth_head(features)
        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=True)
        
        # Apply sigmoid to get normalized depth in [0, 1]
        depth = torch.sigmoid(depth)
        
        return depth
    
    def compute_loss(self, pred, target):
        """
        Compute loss using normalized depth values
        pred, target: depth values in [0, 1] range
        """
        valid_mask = (target > 0).detach()
        
        # Add small epsilon to avoid log(0)
        eps = 1e-6
        pred = pred.clamp(eps, 1.0)
        target = target.clamp(eps, 1.0)
        
        diff = torch.log(pred) - torch.log(target)
        diff = diff * valid_mask
        
        num_valid = valid_mask.sum() + eps
        
        loss = (diff ** 2).sum() / num_valid - \
               0.5 * (diff.sum() ** 2) / (num_valid ** 2)
        
        return loss
