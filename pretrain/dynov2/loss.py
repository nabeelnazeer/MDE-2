"""
Implementation of the DINOv2 loss function for self-supervised learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class DINOLoss(nn.Module):
    """
    DINO loss for self-supervised learning as described in
    "Emerging Properties in Self-Supervised Vision Transformers"
    """
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        
    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        
        Args:
            student_output: Tensor of shape (batch_size, out_dim)
            teacher_output: Tensor of shape (batch_size, out_dim)
        """
        student_temp = self.student_temp
        teacher_temp = self.teacher_temp
        
        # Get batch size
        batch_size = student_output.shape[0]
        
        # Teacher sharpening
        teacher_out = F.softmax(teacher_output / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()  # Stop gradient for teacher
        
        # Student sharpening and log
        student_out = F.log_softmax(student_output / student_temp, dim=-1)
        
        # Cross entropy loss
        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()
        
        # Update center for teacher
        self.update_center(teacher_output)
        
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output with exponential moving average.
        
        Args:
            teacher_output: Tensor of shape (batch_size, out_dim)
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        
        # Gather centers from all GPUs if using distributed training
        if dist.is_initialized():
            batch_center_all = [torch.zeros_like(batch_center) for _ in range(dist.get_world_size())]
            dist.all_gather(batch_center_all, batch_center)
            batch_center = torch.cat(batch_center_all, dim=0).mean(dim=0, keepdim=True)
        
        # Update center with momentum
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
