import torch

def normalize_depth(depth, min_depth=1e-3, max_depth=80.0):
    """Normalize depth values to [0, 1] range"""
    depth = torch.clamp(depth, min_depth, max_depth)
    return (depth - min_depth) / (max_depth - min_depth)

def denormalize_depth(depth_norm, min_depth=1e-3, max_depth=80.0):
    """Convert normalized depth back to metric depth"""
    return depth_norm * (max_depth - min_depth) + min_depth

def compute_scale_shift(pred, target, valid_mask=None):
    """Compute optimal scale and shift for depth prediction"""
    if valid_mask is None:
        valid_mask = target > 0
    
    pred_masked = pred[valid_mask]
    target_masked = target[valid_mask]
    
    # Compute scale and shift
    scale = (target_masked * pred_masked).sum() / (pred_masked * pred_masked).sum()
    shift = target_masked.mean() - scale * pred_masked.mean()
    
    return scale, shift
