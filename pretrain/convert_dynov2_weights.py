"""
Script to convert pretrained DynoV2 weights to the format needed for the teacher model.
"""
import os
import torch
import argparse
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))
from models.teacher import DynoV2Teacher
from pretrain.dynov2.model import DynoV2Encoder

def parse_args():
    parser = argparse.ArgumentParser(description='Convert DynoV2 weights to teacher model format')
    parser.add_argument('--input', type=str, required=True, help='Path to pretrained DynoV2 weights')
    parser.add_argument('--output', type=str, default='./dynov2_teacher.pth', help='Output path for teacher weights')
    return parser.parse_args()

def convert_weights(input_path, output_path):
    """Convert DynoV2 weights to teacher model format"""
    print(f"Loading pretrained weights from {input_path}")
    
    # Load pretrained weights
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'student' in checkpoint:
        state_dict = checkpoint['student']
        print("Loaded weights from student model")
    elif 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
        print("Loaded weights from teacher model")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model'] 
        print("Loaded weights from model key")
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = checkpoint
        print("Loaded weights directly from checkpoint")
    
    # Create teacher model
    teacher_model = DynoV2Teacher(pretrained=False)
    
    # Try to load weights
    try:
        # Create a temporary encoder to handle conversion
        temp_encoder = DynoV2Encoder()
        # Load weights into temporary encoder
        temp_encoder.model.load_state_dict(state_dict, strict=False)
        
        # Now get backbone weights from temp encoder and load into teacher
        backbone_state_dict = {}
        for k, v in temp_encoder.model.backbone.state_dict().items():
            backbone_state_dict[f"encoder.{k}"] = v
            
        # Load backbone weights
        missing_keys, unexpected_keys = teacher_model.load_state_dict(backbone_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        print("Successfully converted backbone weights")
        
    except Exception as e:
        print(f"Error during weight conversion: {e}")
        print("Trying direct loading...")
        
        # Try direct loading with some key mapping
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone'):
                # Map backbone keys to encoder keys
                new_key = k.replace('backbone', 'encoder')
                new_state_dict[new_key] = v
            elif k.startswith('encoder'):
                # Keep encoder keys as is
                new_state_dict[k] = v
            else:
                # Other keys (might need more specific handling)
                new_state_dict[f"encoder.{k}"] = v
                
        # Load the converted state dict
        missing_keys, unexpected_keys = teacher_model.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    
    # Save the converted model
    print(f"Saving converted weights to {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(teacher_model.state_dict(), output_path)
    print("Conversion completed.")

def main():
    args = parse_args()
    convert_weights(args.input, args.output)

if __name__ == '__main__':
    main()