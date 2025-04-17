from typing import List, Literal
from pathlib import Path

import torch
import numpy as np
import PIL.Image as Image
import argparse
import os

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

from dataset import DepthEstimationDataset, create_depth_dataloaders
from evaluate import evaluate as eval_depth_maps
from utils import RunningAverageDict, RunningAverage

# Optional wandb import with handling
try:
    import wandb
except ImportError:
    wandb = None

MODEL_CONFIG = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_name: str):
    model_config = MODEL_CONFIG[model_name]
    model_path = f'checkpoints/depth_anything_v2_{model_name}.pth'
    depth_anything = DepthAnythingV2(**model_config)
    depth_anything.load_state_dict(torch.load(model_path))
    depth_anything = depth_anything.to(DEVICE)
    return depth_anything

def train_step(model, train_dataloader, optimizer, criterion):
    model.train()
    avg_loss = RunningAverage()
    
    for imgs, depth_maps in train_dataloader:
        imgs = imgs.to(DEVICE)
        depth_maps = depth_maps.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, depth_maps)
        loss.backward()
        optimizer.step()
        
        avg_loss.update(loss.item())
    
    return avg_loss.get_value()

def eval_step(model, val_dataloader, criterion):
    model.eval()
    metrics = RunningAverageDict()
    avg_loss = RunningAverage()
    
    with torch.no_grad():
        for imgs, depth_maps in val_dataloader:
            imgs = imgs.to(DEVICE)
            depth_maps = depth_maps.to(DEVICE)
            
            out = model(imgs)
            loss = criterion(out, depth_maps)
            avg_loss.update(loss.item())
            
            # Process each image in the batch
            for i in range(imgs.size(0)):
                batch_metrics = eval_depth_maps(out[i].unsqueeze(0).cpu(), depth_maps[i].unsqueeze(0).cpu())
                metrics.update(batch_metrics)
    
    metrics_dict = metrics.get_value()
    metrics_dict["val_loss"] = avg_loss.get_value()
    return metrics_dict

def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int,
        backbone_lr: float,
        dpt_head_lr: float,
        use_wandb: bool,
        experiment_name: str = None,
        save_dir: str = "saved_models"
):
    # Set up different parameter groups with different learning rates
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    # Set up optimizer with parameter groups
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': dpt_head_lr}
    ])
    
    # Define loss function
    criterion = torch.nn.L1Loss()
    
    # Set up wandb if enabled
    if use_wandb and wandb is not None:
        wandb.init(project="depth_anything_v2_training", name=experiment_name)
        wandb.config.update({
            "epochs": epochs,
            "backbone_lr": backbone_lr,
            "dpt_head_lr": dpt_head_lr,
            "train_batch_size": train_dataloader.batch_size,
            "val_batch_size": val_dataloader.batch_size,
        })
    
    # Create directory to save models if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training step
        train_loss = train_step(model, train_dataloader, optimizer, criterion)
        
        # Validation step
        val_metrics = eval_step(model, val_dataloader, criterion)
        val_loss = val_metrics['val_loss']
        
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics: {val_metrics}")
        
        # Log to wandb if enabled
        if use_wandb and wandb is not None:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **val_metrics
            }
            wandb.log(log_dict)
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(save_dir, f"best_model_{experiment_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
            
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}_{experiment_name}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Close wandb run if it was used
    if use_wandb and wandb is not None:
        wandb.finish()

def main(
        model_name: Literal['vits', 'vitb', 'vitl', 'vitg'],
        dataset_root_path: str,
        sport_name: str = None,
        seed: int = 42,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        epochs: int = 30,
        backbone_lr: float = 1e-5,
        dpt_head_lr: float = 1e-4,
        use_wandb: bool = False,
        experiment_name: str = None
):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Default experiment name if not provided
    if experiment_name is None:
        experiment_name = f"depth_anything_v2_{model_name}"
    
    # Load model
    model = load_model(model_name)
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_depth_dataloaders(
        dataset_root_path, 
        sport_name, 
        train_batch_size,
        val_batch_size,
        seed
    )
    
    # Train model
    train(
        model,
        train_dataloader,
        val_dataloader,
        epochs,
        backbone_lr,
        dpt_head_lr,
        use_wandb,
        experiment_name
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Depth Anything V2')
    parser.add_argument('--model', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'], default='vits',
                        help='Model size to use')
    parser.add_argument('--dataset-path', dest='dataset_path', type=str, required=True,
                        help='Path to the dataset root directory')
    parser.add_argument('--sport-name', dest='sport_name', type=str, default=None,
                        help='Optional sport name filter for dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--train-batch-size', dest='train_batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val-batch-size', dest='val_batch_size', type=int, default=8,
                        help='Batch size for validation')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--backbone-lr', dest='backbone_lr' type=float, default=1e-5,
                        help='Learning rate for backbone parameters')
    parser.add_argument('--head-lr', dest='head_lr', type=float, default=1e-4,
                        help='Learning rate for DPT head parameters')
    parser.add_argument('--use-wandb', dest='use_wandb' action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, default=None,
                        help='Name for the experiment run')
    
    args = parser.parse_args()
    
    main(
        model_name=args.model,
        dataset_root_path=args.dataset_path,
        sport_name=args.sport_name,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        backbone_lr=args.backbone_lr,
        dpt_head_lr=args.head_lr,
        use_wandb=args.use_wandb,
        experiment_name=args.experiment_name
    )
