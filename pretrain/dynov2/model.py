"""
DynoV2 model implementation based on Vision Transformer (ViT) architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VisionTransformer

class DynoV2(nn.Module):
    """
    DynoV2 model with Vision Transformer backbone
    """
    def __init__(self, 
                 img_size=224,
                 patch_size=16, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12,
                 mlp_ratio=4,
                 out_dim=65536, 
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 use_cls_token=True,
                 freeze_backbone=False):
        super().__init__()
        
        # Vision Transformer backbone
        self.backbone = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            num_layers=depth,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=embed_dim * mlp_ratio,
            num_classes=0  # No classification head
        )
        
        # Extract features from the class token
        self.use_cls_token = use_cls_token
        
        # Projector network (MLP) for DINO
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            norm_layer(2048),
            act_layer(),
            nn.Linear(2048, 2048),
            norm_layer(2048),
            act_layer(),
            nn.Linear(2048, out_dim)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        
        # Apply projector to get embedding
        proj = self.projector(features)
        
        # Normalize embedding
        proj = F.normalize(proj, dim=-1)
        
        return proj
    
    def get_intermediate_layers(self, x, n=1):
        """Return intermediate transformer layers"""
        return self.backbone.get_intermediate_layers(x, n)
    
    def get_last_selfattention(self, x):
        """Get attention maps from the last layer"""
        return self.backbone.get_last_selfattention(x)
    
class DynoV2Encoder(nn.Module):
    """
    DynoV2 encoder for depth estimation
    """
    def __init__(self, pretrained_weights=None, freeze=True):
        super().__init__()
        
        # Create DynoV2 model
        self.model = DynoV2()
        
        # Load pretrained weights if provided
        if pretrained_weights is not None:
            if pretrained_weights.endswith('.pth'):
                state_dict = torch.load(pretrained_weights, map_location='cpu')
                if 'student' in state_dict:
                    # If we have the full checkpoint with student/teacher
                    state_dict = state_dict['student']
                elif 'model' in state_dict:
                    # If we have model weights only
                    state_dict = state_dict['model']
                
                # Load the state dict
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights from {pretrained_weights}: {msg}")
            else:
                print(f"Pretrained weights file not found: {pretrained_weights}")
        
        # Remove projector as we don't need it for depth estimation
        self.model.projector = nn.Identity()
        
        # Freeze backbone if specified
        if freeze:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            
    def forward(self, x):
        return self.model(x)
    
    def get_intermediate_layers(self, x, n=1):
        return self.model.get_intermediate_layers(x, n)
