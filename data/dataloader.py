import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class DepthDataset(Dataset):
    """Dataset for loading depth estimation data"""
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        
        # Ensure exact 224x224 input size for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Slightly larger size
            transforms.CenterCrop(224),     # Then crop to exact size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
        # Add depth normalization parameters
        self.min_depth = 1e-3
        self.max_depth = 80.0
        
        # Get file lists
        split_dir = os.path.join(data_dir, split)
        self.image_dir = os.path.join(split_dir, 'images')
        self.depth_dir = os.path.join(split_dir, 'depths')
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(self.image_dir) 
                            if f.endswith(('.jpg', '.png'))])
    
    def __len__(self):
        return len(self.images)
    
    def normalize_depth(self, depth):
        """Normalize depth values to [0, 1] range"""
        depth = torch.clamp(depth, self.min_depth, self.max_depth)
        return (depth - self.min_depth) / (self.max_depth - self.min_depth)
    
    def denormalize_depth(self, depth):
        """Convert normalized depth back to metric depth"""
        return depth * (self.max_depth - self.min_depth) + self.min_depth
    
    def __getitem__(self, idx):
        # Get image path
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Get corresponding depth path
        depth_name = os.path.splitext(img_name)[0] + '.png'
        depth_path = os.path.join(self.depth_dir, depth_name)
        
        # Load image and depth
        image = Image.open(img_path).convert('RGB')
        depth = Image.open(depth_path)
        
        # Apply transforms
        image = self.transform(image)
        depth = self.depth_transform(depth)
        
        # Normalize depth
        depth = self.normalize_depth(depth)
        
        return image, depth

def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    """Create dataloaders for train, validation and test sets"""
    train_dataset = DepthDataset(data_dir, split='train')
    val_dataset = DepthDataset(data_dir, split='val')
    test_dataset = DepthDataset(data_dir, split='test')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
