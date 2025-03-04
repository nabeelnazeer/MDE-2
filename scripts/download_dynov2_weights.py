"""
Script to download pretrained DINOv2 weights from the official repository.
These weights will be used as initialization for the DynoV2 teacher model.
"""
import os
import argparse
import requests
import torch
from tqdm import tqdm

# URLs for pretrained DINOv2 models
DINOV2_URLS = {
    'vit_small': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
    'vit_base': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
    'vit_large': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
    'vit_giant': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth',
}

def parse_args():
    parser = argparse.ArgumentParser(description='Download pretrained DINOv2 weights')
    parser.add_argument('--model', type=str, default='vit_base',
                        choices=list(DINOV2_URLS.keys()),
                        help='Model architecture to download')
    parser.add_argument('--output_dir', type=str, default='./pretrained_weights',
                        help='Directory to save weights')
    return parser.parse_args()

def download_file(url, destination):
    """Download a file with a progress bar"""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(destination):
        print(f"File already exists at {destination}. Skipping download.")
        return
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get file size from headers
    file_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=file_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            progress_bar.update(size)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get URL for selected model
    url = DINOV2_URLS[args.model]
    filename = f"dinov2_{args.model}_pretrain.pth"
    destination = os.path.join(args.output_dir, filename)
    
    print(f"Downloading {args.model} weights from {url}...")
    download_file(url, destination)
    
    # Verify the downloaded file
    try:
        checkpoint = torch.load(destination, map_location='cpu')
        print(f"Successfully downloaded and loaded weights: {filename}")
        
        # Print some info about the model
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                print("Checkpoint contains 'state_dict' key")
            elif 'model' in checkpoint:
                print("Checkpoint contains 'model' key")
                
            for key in checkpoint.keys():
                print(f"Key: {key}")
    except Exception as e:
        print(f"Error verifying weights: {e}")
        print("The download may be incomplete or corrupted.")

if __name__ == '__main__':
    main()
