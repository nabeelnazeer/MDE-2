import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from models.teacher import DynoV2Teacher
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain DynoV2 on ImageNet')
    parser.add_argument('--imagenet_path', type=str, required=True,
                        help='Path to ImageNet dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/imagenet_pretrain',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=90,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = DynoV2Teacher(pretrained=False).to(device)
    
    # ImageNet transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create ImageNet dataset and dataloader
    train_dataset = datasets.ImageNet(
        args.imagenet_path,
        split='train',
        transform=train_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    # Create optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Training loop
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'logs'))
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss (use model's internal loss)
            loss = model.compute_loss(outputs)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        avg_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'final_model.pth'))
    writer.close()

if __name__ == '__main__':
    main()
