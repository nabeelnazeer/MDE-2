"""
Data augmentations for DynoV2 pretraining.
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class DataAugmentationDINO:
    """
    Data augmentation strategy for DINO pretraining.
    This creates two global views and several local views of an image.
    """
    def __init__(self, 
                global_crops_scale=(0.4, 1.0),
                local_crops_scale=(0.05, 0.4),
                local_crops_number=8,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                img_size=224,
                local_crops_size=96):
        
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # Transformations for global crops
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            normalize,
        ])
        
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            normalize,
        ])
        
        # Transformations for local crops
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            normalize,
        ])
        
    def __call__(self, image):
        """
        Apply augmentation pipeline.
        
        Args:
            image: PIL Image to transform
            
        Returns:
            List of transformed images (2 global views + local views)
        """
        crops = []
        # Global crops
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        
        # Local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
            
        return crops
