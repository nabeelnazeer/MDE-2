import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from models.teacher import DynoV2Teacher
from data.dataloader import DepthDataset
from utils import load_model

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on sample images')
    parser.add_argument('--model_path', type=str, 
                        default='checkpoints/teacher_pretrain/teacher_best.pth',
                        help='Path to model weights')
    parser.add_argument('--samples_dir', type=str, 
                        default='samples',
                        help='Directory containing sample images')
    parser.add_argument('--output_dir', type=str, 
                        default='outputs/inference_results',
                        help='Directory to save results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and load weights
    model = DynoV2Teacher(pretrained=False).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])
    
    # Process each image in the samples directory
    sample_files = [f for f in os.listdir(args.samples_dir) 
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(sample_files)} sample images")
    
    with torch.no_grad():
        for sample_file in sample_files:
            # Load and preprocess image
            img_path = os.path.join(args.samples_dir, sample_file)
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Run inference
            depth_pred = model(input_tensor)
            
            # Convert prediction to visualization
            depth_vis = depth_pred[0, 0].cpu().numpy()
            
            # Normalize for visualization
            depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())
            
            # Create visualization
            plt.figure(figsize=(12, 5))
            
            plt.subplot(121)
            plt.imshow(image)
            plt.title('Input Image')
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(depth_vis, cmap='magma')
            plt.title('Predicted Depth')
            plt.axis('off')
            
            plt.colorbar(label='Normalized Depth')
            
            # Save result
            output_path = os.path.join(args.output_dir, f'depth_{os.path.splitext(sample_file)[0]}.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Processed {sample_file}")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == '__main__':
    main()
