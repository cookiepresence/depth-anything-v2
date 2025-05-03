from typing import List, Literal
from pathlib import Path

import torch
import numpy as np
import PIL.Image as Image
import argparse
import os
import tqdm
import peft

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

from dataset import DepthEstimationDataset, create_depth_dataloaders, CutMix
from evaluate import evaluate as eval_depth_maps
from utils import RunningAverageDict, RunningAverage

try:
    import wandb
except ImportError:
    wandb = None

MODEL_CONFIG = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vits_r': {'encoder': 'vits_r', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def scale_and_shift(pred, target, mask):

    batch_size = pred.shape[0]

    h, w = pred.shape[1], pred.shape[2]
    pred_flat = pred.view(batch_size, -1)  # B x (H*W)
    target_flat = target.view(batch_size, -1)  # B x (H*W)
    mask_flat = mask.view(batch_size, -1)  # B x (H*W)

    if torch.isnan(pred_flat).any() or torch.isnan(target_flat).any():
        print("Warning: NaN values found in input tensors")
        exit(0)

    mask_flat = mask_flat.bool().float()
    valid_pixels = mask_flat.sum(dim=1)
    # if (valid_pixels < 10).any():
    #     print(f"Warning: Very few valid pixels in some samples: {valid_pixels}")

    a_00 = torch.sum(mask_flat * pred_flat * pred_flat, dim=1)  # B
    a_01 = torch.sum(mask_flat * pred_flat, dim=1)  # B
    a_11 = torch.sum(mask_flat, dim=1)  # B

    b_0 = torch.sum(mask_flat * pred_flat * target_flat, dim=1)  # B
    b_1 = torch.sum(mask_flat * target_flat, dim=1)  # B

    det = a_00 * a_11 - a_01 * a_01  # B

    scale = torch.ones(batch_size, device=pred.device)
    shift = torch.zeros(batch_size, device=pred.device)

    eps = 1e-8 * torch.max(a_00) * torch.max(a_11)  # Dynamic threshold
    valid = (det > eps) & (a_11 > 0)  # Additional check for a_11 positivity
    
    # Apply safeguards for numerical stability
    # if valid.sum() < batch_size:
    #     print(f"Warning: {batch_size - valid.sum()} samples have unstable determinants")

    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]


    if torch.isnan(scale).any() or torch.isnan(shift).any():
        print("NaNs found in scale or shift!")
        exit(0)

    scaled_pred = scale.view(batch_size, 1, 1) * pred + shift.view(batch_size, 1, 1)

    if torch.isnan(scaled_pred).any():
        print("NaNs found in final output!")
        exit(0)

    return scaled_pred

def global_normalization(depth_maps):

    batch_size = depth_maps.shape[0]
    normalized_maps = torch.zeros_like(depth_maps)

    depth_maps_flat = depth_maps.reshape(batch_size, -1)

    median = torch.median(depth_maps_flat, dim=1, keepdim=True).values

    abs_diff = torch.abs(depth_maps_flat - median)
    mad = torch.mean(abs_diff, dim=1, keepdim=True)
    mad = torch.clamp(mad, min=1e-8)

    normalized_maps_flat = (depth_maps_flat - median) / mad
    normalized_maps = normalized_maps_flat.reshape(depth_maps.shape)

    return normalized_maps

def SILogLoss(pred, target, mask=None, variance_focus=0.85):
    """
    Compute SILog loss between predicted and target depth maps.

    Args:
        pred (Tensor): Predicted depth (B x H x W or B x 1 x H x W).
        target (Tensor): Ground truth depth (same shape as pred).
        mask (Tensor, optional): Binary mask to include valid pixels only.

    Returns:
        Tensor: SILog loss value.
    """
    if mask is None:
        mask = (target > 0).detach()
    
    mask[:, 870:1016, 1570:1829] = 0

    # scaled_pred = scale_and_shift(pred, target, mask)

    # assert not torch.isnan(scaled_pred).any()

    # pred = torch.where(mask, pred, 0)
    pred = pred[mask]
    # target = torch.where(mask, target, 0)
    target = target[mask]

    # norm_student_depth = global_normalization(pred)
    # norm_teacher_depth = global_normalization(target)

    # loss = torch.mean(torch.abs(norm_student_depth - norm_teacher_depth))

    # return loss

    log_diff = torch.log(pred + 1e-8) - torch.log(target + 1e-8)
    mean_log_diff_squared = torch.mean(log_diff ** 2)
    mean_log_diff = torch.mean(log_diff)

    silog_loss = mean_log_diff_squared - variance_focus * (mean_log_diff ** 2)
    return silog_loss

def GradientMatchingLoss(pred, target, mask=None):
    assert pred.shape == target.shape, "pred and target must have the same shape"

    if mask is None:
        mask = (target > 0).detach()
    
    mask[:, 870:1016, 1570:1829] = 0

    # scaled_pred = scale_and_shift(pred, target, mask)

    # assert not torch.isnan(scaled_pred).any()

    pred = torch.where(mask, pred, 0)
    target = torch.where(mask, target, 0)

    N = torch.sum(mask)
    log_d_diff = torch.log(pred + 1e-8) - torch.log(target + 1e-8)

    v_grad = torch.abs(log_d_diff[...,:-2,:] - log_d_diff[..., 2:, :])
    h_grad = torch.abs(log_d_diff[..., :, :-2] - log_d_diff[..., :, 2:])

    return (torch.sum(h_grad) + torch.sum(v_grad)) / N

def loss_criterion(preds, target, mask=None, variance_focus=0.85, alpha=3.0):
    return SILogLoss(preds, target, mask, variance_focus) + alpha * GradientMatchingLoss(preds, target, mask)
    # return SILogLoss(preds, target, mask, variance_focus)

def load_model(model_name: str, use_registers: bool = False, model_weights: str = None):
    model_name_r = model_name + '_r' if use_registers else model_name
    model_config = MODEL_CONFIG[model_name_r]
    model_path = f'checkpoints/depth_anything_v2_{model_name}.pth'
    depth_anything = DepthAnythingV2(**model_config)
    if model_weights is not None:
        depth_anything.load_state_dict(torch.load(model_weights, weights_only=True))
    else:
        depth_anything.load_state_dict(torch.load(model_path, weights_only=True), strict=False)

    if use_registers:
        depth_anything.pretrained.load_state_dict(torch.load(f'checkpoints/dinov2-with-registers-{model_name}.pt', map_location=DEVICE, weights_only=True))
    
    depth_anything = depth_anything.to(DEVICE)
    return depth_anything

def train_step(model, train_dataloader, optimizer, scheduler, criterion, use_masking = False, use_cutmix: bool = True):
    model.train()
    train_loss = RunningAverageDict()
    batch_train_loss = []

    if use_cutmix:
        cutmix = CutMix(beta=1, patch_size=14)

    for imgs, depth_maps, metadata in tqdm.tqdm(train_dataloader):
        if torch.isnan(imgs).any():
            print("nan detected after colorjitter for images")
            exit(0)
        if torch.isnan(depth_maps).any():
            print("nan detected after colorjitter for depth maps")
            exit(0)
        imgs = imgs.to(DEVICE)
        depth_maps = depth_maps.to(DEVICE).squeeze(1)

        # apply cutmix
        if use_cutmix:
            imgs, depth_maps, _, _ = cutmix(imgs, depth_maps)

        optimizer.zero_grad()
        out = model(imgs)
        if torch.isnan(out).any():
            print("nan values out")
            exit(0)
        if use_masking:
            out_copy = out.clone().detach()
            depth_maps_copy = depth_maps.clone().detach()
            log_diff = torch.abs(torch.log(out_copy+1e-8) - torch.log(depth_maps_copy + 1e-8))
            pixel_loss = log_diff
            percentile = 70
            masking_threshold = torch.quantile(pixel_loss.flatten(), percentile/100.0)
            mask = (pixel_loss < masking_threshold).clone().detach()
            loss = criterion(out, depth_maps, mask)
        else:
            loss = criterion(out, depth_maps)

        # print(loss.item())
        loss.backward()
        batch_train_loss.append(loss.item())
        optimizer.step()

        train_loss.update({"loss": loss.item()})

    scheduler.step()
    return train_loss.get_value(), batch_train_loss

def eval_step(model, val_dataloader, criterion, sport_name):
    model.eval()
    metrics = RunningAverageDict()

    with torch.no_grad():
        for imgs, depth_maps, metadata in tqdm.tqdm(val_dataloader):
            imgs = imgs.to(DEVICE)
            depth_maps = depth_maps.to(DEVICE).squeeze(1)

            # output: [B, 1, E, E]
            out = model(imgs)
            # out = (out - out.min()) / (out.max() - out.min())
            loss = criterion(out, depth_maps)
            metrics.update({'val_loss': loss.item()})

            # Process each image in the batch
            for i in range(imgs.size(0)):
                batch_metrics = eval_depth_maps(out[i], depth_maps[i], sport_name=sport_name, device=DEVICE, mask_need=True)
                metrics.update(batch_metrics)
    metrics_dict = metrics.get_value()
    return metrics_dict

def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int,
        backbone_lr: float,
        dpt_head_lr: float,
        weight_decay: float,
        use_wandb: bool,
        sport_name: str = None,
        experiment_name: str = None,
        use_cutmix: bool = True,
        use_lora: bool = False,
        min_lr_factor: float = 1e-2,
        save_dir: str = "saved_models"
):
    # Set up different parameter groups with different learning rates
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'pretrained' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Set up optimizer with parameter groups
    if backbone_lr == 0.0:
        optimizer = torch.optim.Adam([
            {'params': head_params, 'lr': dpt_head_lr}
        ], weight_decay = weight_decay)
        # ensure that we freeze backbone params correctly
        for p in backbone_params:
            p.requires_grad_(False)
    else:
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': dpt_head_lr}
        ], weight_decay = weight_decay)

    min_lr = dpt_head_lr * min_lr_factor
    # num_training_steps = epochs * len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,
        eta_min=min_lr  # This will be scaled for each param group
    )

    # Define loss function
    # criterion = torch.nn.L1Loss()
    # criterion = SILogLoss()
    criterion = loss_criterion

    # Set up wandb if enabled
    if use_wandb and wandb is not None:
        wandb.init(project="depth_anything_v2_finetuning", name=experiment_name)
        wandb.config.update({
            "epochs": epochs,
            "backbone_lr": backbone_lr,
            "dpt_head_lr": dpt_head_lr,
            "train_batch_size": train_dataloader.batch_size,
            "val_batch_size": val_dataloader.batch_size,
            "min_lr_factor": min_lr_factor,
        })

    # Create directory to save models if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_val_loss = float('inf')

    # # Validation step
    val_metrics = eval_step(model, val_dataloader, criterion, sport_name)
    val_loss = val_metrics['val_loss']

    # Print metrics
    print(f"Epoch 0/{epochs}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Metrics:")
    for k, v in val_metrics.items():
        print(f"\t{k}: {v*1e3:.4f}")

    switch_every_n_epochs = 3
    for epoch in range(epochs):
        if args.use_masking:
            use_dynamic_mask = ((epoch) % (switch_every_n_epochs+1) == switch_every_n_epochs)
            print(f"Dynamic mask: {use_dynamic_mask}")
        # Training step
            train_loss, batch_train_loss = train_step(model, train_dataloader, optimizer, scheduler, criterion, use_dynamic_mask, use_cutmix=use_cutmix)
        else:
            train_loss, batch_train_loss = train_step(model, train_dataloader, optimizer, scheduler, criterion, use_cutmix=use_cutmix)
        train_loss = train_loss['loss']

        # Validation step
        val_metrics = eval_step(model, val_dataloader, criterion, sport_name)
        val_loss = val_metrics['val_loss']

        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics:")
        for k, v in val_metrics.items():
            print(f"\t{k}: {v*1e3:.4f}")

        # Log to wandb if enabled
        if use_wandb and wandb is not None:
            for loss in batch_train_loss:
                wandb.log({
                    "epoch": epoch + 1,
                    "batch_train_loss": loss
                })
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
            # checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}_{experiment_name}.pth")
            # torch.save({
            #     'epoch': epoch + 1,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'train_loss': train_loss,
            #     'val_loss': val_loss,
            # }, checkpoint_path)
            # print(f"Saved checkpoint to {checkpoint_path}")
            pass

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
        weight_decay: float = 1e-3,
        use_wandb: bool = False,
        experiment_name: str = None,
        use_registers: bool = False,
        model_weights: str = None,
        use_cutmix: bool = True,
        use_lora: bool = False,
        lora_rank: int = 2,
        lora_alpha: int = 16,
        lora_modules: List[str] = ["qkv"],
):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Default experiment name if not provided
    if experiment_name is None:
        experiment_name = f"depth_anything_v2_{model_name}"

    # Load model
    model = load_model(model_name, use_registers, model_weights)
    if use_lora:
        print(f"Lora modules: {lora_modules[0]}")
        lora_config = peft.LoraConfig(
            r=lora_rank,
            target_modules=lora_modules[0],
            lora_alpha=lora_alpha,
            lora_dropout=0.05
        )
        peft.get_peft_model(model, lora_config)

    # Create dataloaders
    train_dataloader, val_dataloader = create_depth_dataloaders(
        root_dir=dataset_root_path,
        sport_name=sport_name,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        seed=seed
    )

    # Train model
    train(
        model,
        train_dataloader,
        val_dataloader,
        epochs,
        backbone_lr,
        dpt_head_lr,
        weight_decay,
        use_wandb,
        sport_name,
        experiment_name,
        use_cutmix,
        use_lora,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Depth Anything V2')
    parser.add_argument('--model', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'], default='vits',
                        help='Model size to use')
    parser.add_argument('--dataset-path', dest='dataset_path', type=Path, required=True,
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
    parser.add_argument('--backbone-lr', dest='backbone_lr', type=float, default=1e-6,
                        help='Learning rate for backbone parameters')
    parser.add_argument('--head-lr', dest='head_lr', type=float, default=1e-5,
                        help='Learning rate for DPT head parameters')
    parser.add_argument('--weight-decay', type=float, default=5e-6,
                        help='Weight decay for model')
    parser.add_argument('--use-wandb', dest='use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, default=None,
                        help='Name for the experiment run')
    parser.add_argument('--use-registers', action='store_true', help='use dino backbone with registers')
    parser.add_argument('--model-weights', type=str, help = 'model weights to use and begin training from', required=False)
    parser.add_argument('--use-masking', action='store_true', help="use alternate masking while finetuning")
    parser.add_argument('--use-cutmix', action='store_true', help='use cutmix while training to improve performance')

    parser.add_argument('--use-lora', action='store_true', help='use lora while training to speed things up?')
    parser.add_argument('--lora-rank', type=int, help='what is the rank of updates?')
    parser.add_argument('--lora-alpha', type=int, help='scaling factor for lora updates')
    parser.add_argument('--lora-modules', type=str, nargs='+', default=[], action='append', help='what modules to apply lora on?')

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
        weight_decay=args.weight_decay,
        use_wandb=args.use_wandb,
        experiment_name=args.experiment_name,
        use_registers=args.use_registers,
        model_weights = args.model_weights,
        use_cutmix = args.use_cutmix,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_modules=args.lora_modules,
    )
