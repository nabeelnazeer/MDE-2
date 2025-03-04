import os
import zipfile
import requests
from tqdm import tqdm
import shutil
from pathlib import Path

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

def prepare_dataset(zip_path, output_dir):
    """Extract and organize dataset"""
    print(f"Extracting dataset to {output_dir}...")
    
    # Create temporary extraction directory
    temp_dir = os.path.join(os.path.dirname(output_dir), "temp_extract")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find extracted directories
    rgb_dir = os.path.join(temp_dir, "nyu_depth_v2_sample", "rgb")
    depth_dir = os.path.join(temp_dir, "nyu_depth_v2_sample", "depth")
    
    # Create dataset directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "depths"), exist_ok=True)
    
    # Get all filenames
    files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
    
    # Split into train, val, test
    train_end = int(len(files) * 0.7)
    val_end = int(len(files) * 0.85)
    
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    # Copy files
    for split, file_list in splits.items():
        print(f"Processing {split} split: {len(file_list)} files")
        for i, filename in enumerate(tqdm(file_list)):
            # Get RGB image
            src_rgb = os.path.join(rgb_dir, filename)
            dst_rgb = os.path.join(output_dir, split, "images", f"{i:05d}.jpg")
            shutil.copy(src_rgb, dst_rgb)
            
            # Get depth map
            depth_filename = filename.replace('.jpg', '.png')
            src_depth = os.path.join(depth_dir, depth_filename)
            dst_depth = os.path.join(output_dir, split, "depths", f"{i:05d}.png")
            shutil.copy(src_depth, dst_depth)
    
    # Clean up
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    print("Dataset preparation complete!")
    print(f"Train: {len(train_files)} samples")
    print(f"Validation: {len(val_files)} samples")
    print(f"Test: {len(test_files)} samples")

def main():
    # URL for the NYU Depth V2 sample dataset
    url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    
    # For simplicity, let's use a smaller sample that's easier to work with
    url = "https://drive.google.com/uc?id=1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw"  # Example URL, update with a valid one
    
    # Define paths
    project_dir = Path(__file__).resolve().parent.parent
    download_dir = os.path.join(project_dir, "downloads")
    os.makedirs(download_dir, exist_ok=True)
    
    output_dir = os.path.join(project_dir, "data")
    zip_path = os.path.join(download_dir, "nyu_depth_sample.zip")
    
    # Download dataset
    print(f"Downloading NYU Depth V2 sample dataset...")
    download_file(url, zip_path)
    
    # Prepare dataset
    prepare_dataset(zip_path, output_dir)

if __name__ == "__main__":
    main()
