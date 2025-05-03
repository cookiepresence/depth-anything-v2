import torch
import numpy as np
import argparse
import os
import tqdm
import torch.nn.functional as F # Import for resizing
from pathlib import Path
from typing import Literal, List

# Assuming these are available in your project
from depth_anything_v2.dpt import DepthAnythingV2
from dataset import DepthEstimationDataset, create_depth_dataloaders
from evaluate import evaluate as eval_depth_maps # Assuming eval_depth_maps is in evaluate.py
from utils import RunningAverageDict # Assuming RunningAverageDict is in utils.py

# Import peft for LoRA support
try:
    import peft
except ImportError:
    peft = None
    print("PEFT library not found. LoRA functionality will not be available.")


# Model configuration as defined in your training script
MODEL_CONFIG = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vits_r': {'encoder': 'vits_r', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_for_eval(
    model_name: str,
    model_weights_path: str,
    use_registers: bool = False,
    use_lora: bool = False,
    lora_rank: int = None,
    lora_alpha: int = None,
    lora_modules: List[str] = None
):
    """
    Loads the model architecture and state dictionary from a trained weights file,
    including support for loading LoRA weights.

    Args:
        model_name (str): The name of the model architecture ('vits', 'vitb', 'vitl', 'vitg').
        model_weights_path (str): Path to the trained model weights file (.pth).
        use_registers (bool): Whether the model uses registers (for DinoV2 backbone).
        use_lora (bool): Whether to load LoRA weights.
        lora_rank (int, optional): The rank of the LoRA updates. Required if use_lora is True.
        lora_alpha (int, optional): The scaling factor for LoRA updates. Required if use_lora is True.
        lora_modules (List[str], optional): List of target module names for LoRA. Required if use_lora is True.

    Returns:
        torch.nn.Module: The loaded model with trained weights (potentially with LoRA).
    """
    model_name_r = model_name + '_r' if use_registers else model_name
    if model_name_r not in MODEL_CONFIG:
        raise ValueError(f"Model name '{model_name_r}' not found in MODEL_CONFIG")

    model_config = MODEL_CONFIG[model_name_r]

    # Load the base model architecture
    depth_anything = DepthAnythingV2(**model_config)

    # If using registers, load the specific pretrained backbone weights (if applicable)
    # Note: This assumes the DinoV2 weights are separate from the finetuned weights.
    # If your model_weights_path contains the full model including backbone, this might not be needed.
    if use_registers:
         try:
             # Attempt to load the DinoV2 backbone weights with registers
             dino_weights_path = f'checkpoints/dinov2-with-registers-{model_name}.pt'
             if os.path.exists(dino_weights_path):
                 print(f"Loading DinoV2 backbone with registers from {dino_weights_path}")
                 depth_anything.pretrained.load_state_dict(torch.load(dino_weights_path, map_location=DEVICE, weights_only=True))
             else:
                  print(f"Warning: DinoV2 weights with registers not found at {dino_weights_path}. Backbone will use default initialization or weights from model_weights_path.")
         except Exception as e:
             print(f"Error loading DinoV2 weights with registers: {e}")


    # Load the state dictionary
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at: {model_weights_path}")

    print(f"Loading model weights from {model_weights_path}")
    state_dict = torch.load(model_weights_path, map_location=DEVICE, weights_only=True)

    if use_lora:
        if peft is None:
            raise ImportError("PEFT library is required for LoRA support, but it's not installed.")
        if lora_rank is None or lora_alpha is None or not lora_modules:
             raise ValueError("LoRA rank, alpha, and modules must be specified when use_lora is True.")

        print(f"Applying LoRA config: rank={lora_rank}, alpha={lora_alpha}, modules={lora_modules}")
        lora_config = peft.LoraConfig(
            r=lora_rank,
            target_modules=lora_modules, # target_modules expects a list of strings
            lora_alpha=lora_alpha,
            lora_dropout=0.05 # Assuming a default dropout for evaluation consistency
        )
        # Wrap the base model with PEFT
        model_with_lora = peft.get_peft_model(depth_anything, lora_config)

        # Load the LoRA weights into the PEFT model
        try:
            # PEFT state dicts usually contain only the LoRA weights
            model_with_lora.base_model.model.load_state_dict(state_dict)
            print("Successfully loaded LoRA weights.")
        except Exception as e:
            print(f"Error loading LoRA state dict: {e}")
            # Attempt to load with strict=False if there are minor mismatches
            try:
                 model_with_lora.load_state_dict(state_dict, strict=False)
                 print("Loaded LoRA weights with strict=False.")
            except Exception as e_strict:
                 print(f"Failed to load LoRA weights even with strict=False: {e_strict}")
                 print("Please ensure the LoRA config and weights match the model.")
                 raise e_strict # Re-raise the error if loading fails

        model = model_with_lora # Use the PEFT wrapped model

    else:
        # Load the full model state dictionary if not using LoRA
        try:
            depth_anything.load_state_dict(state_dict)
            print("Successfully loaded full model weights.")
        except RuntimeError as e:
            print(f"Error loading state dict: {e}")
            print("Attempting to load with strict=False (may indicate mismatch in model architecture/keys)")
            depth_anything.load_state_dict(state_dict, strict=False)
        model = depth_anything # Use the base model


    model = model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    return model

def create_test_dataloader(
    root_dir: Path,
    sport_name: str,
    batch_size: int = 8,
    crop_size: int = 518,
    num_workers: int = 8,
    seed: int = 42
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for the test dataset.

    Args:
        root_dir (Path): Root directory of the test data (e.g., dataset_root / "Test").
        sport_name (str): Sport name filter (can be None).
        batch_size (int): Batch size for the dataloader.
        crop_size (int): Size to crop the shorter side to (should match training).
        num_workers (int): Number of workers for the dataloader.
        seed (int): Random seed.

    Returns:
        torch.utils.data.DataLoader: Test dataloader.
    """
    test_dataset = DepthEstimationDataset(
        root_dir=root_dir,
        sport_name=sport_name,
        crop_size=crop_size,
        apply_augmentations=False
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # Do not shuffle test data
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    return test_loader


def evaluate(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    sport_name: str = None,
    resolution_scales: List[float] = [1.0], # Added resolution scales argument
    base_crop_size: int = 518 # Added base crop size for scaling calculation
):
    """
    Evaluates the model on the test dataset with optional test-time resolution scaling.

    Args:
        model (torch.nn.Module): The trained model.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
        sport_name (str, optional): Sport name filter. Defaults to None.
        resolution_scales (List[float]): List of scaling factors to apply to the base resolution.
        base_crop_size (int): The base resolution (shorter side) used during training/dataset loading.

    Returns:
        dict: Dictionary of average evaluation metrics.
    """
    model.eval() # Ensure model is in evaluation mode
    metrics = RunningAverageDict()

    print("Starting evaluation with resolution scales:", resolution_scales)
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for imgs, depth_maps, metadata in tqdm.tqdm(test_dataloader, desc="Evaluating"):
            # imgs and depth_maps are already at base_crop_size resolution from the dataloader
            imgs = imgs.to(DEVICE)
            depth_maps = depth_maps.to(DEVICE).squeeze(1) # Assuming depth_maps are Bx1xHxW

            ensemble_preds = []

            for scale in resolution_scales:
                if scale == 1.0:
                    # Use the original image resolution from the dataloader
                    scaled_imgs = imgs
                else:
                    # Calculate target size and resize input images
                    scaled_h = int(base_crop_size * scale)
                    scaled_w = int(base_crop_size * scale)

                    # Ensure dimensions are multiples of 14 (or model's patch size)
                    scaled_h = (scaled_h // 14) * 14
                    scaled_w = (scaled_w // 14) * 14

                    # Resize input images
                    scaled_imgs = F.interpolate(
                        imgs,
                        size=(scaled_h, scaled_w),
                        mode='bilinear',
                        align_corners=False
                    )

                # Get model prediction at the current scale
                scaled_out = model(scaled_imgs)

                # Resize the prediction back to the original depth map resolution (base_crop_size)
                # The model output 'scaled_out' is expected to be BxHxW after the DPT head
                resized_pred = F.interpolate(
                    scaled_out.unsqueeze(1), # Add channel dimension for interpolate
                    size=depth_maps.shape[-2:], # Target size is the original depth map size (HxW)
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1) # Remove the added channel dimension

                ensemble_preds.append(resized_pred)

            # Combine predictions from different scales (e.g., average)
            if len(ensemble_preds) > 0:
                combined_pred = torch.mean(torch.stack(ensemble_preds, dim=0), dim=0)
            else:
                # Should not happen if resolution_scales is not empty, but as a safeguard
                combined_pred = torch.zeros_like(depth_maps)


            # Process each image in the batch for metrics using the combined prediction
            for i in range(imgs.size(0)):
                # eval_depth_maps is assumed to take single image tensors (HxW)
                # and return a dictionary of metrics for that image.
                batch_metrics = eval_depth_maps(combined_pred[i], depth_maps[i], sport_name=sport_name, device=DEVICE, mask_need=True)
                metrics.update(batch_metrics)

    metrics_dict = metrics.get_value()
    return metrics_dict

def main():
    parser = argparse.ArgumentParser(description='Evaluate Depth Anything V2')
    parser.add_argument('--model', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'], default='vits',
                        help='Model size used for training')
    parser.add_argument('--dataset-path', dest='dataset_path', type=Path, required=True,
                        help='Path to the dataset root directory (should contain a "Test" subfolder)')
    parser.add_argument('--model-weights', type=str, required=True,
                        help='Path to the trained model weights file (.pth). This should be the LoRA weights if --use-lora is set.')
    parser.add_argument('--sport-name', dest='sport_name', type=str, default=None,
                        help='Optional sport name filter for dataset')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--use-registers', action='store_true',
                        help='Specify if the trained model used a DinoV2 backbone with registers')
    # Add crop_size argument to ensure consistency with training/dataset loading
    parser.add_argument('--crop-size', type=int, default=518,
                        help='Size to crop the shorter side of images to (should match training)')

    # LoRA arguments
    parser.add_argument('--use-lora', action='store_true',
                        help='Enable LoRA model loading. Requires --lora-rank, --lora-alpha, and --lora-modules.')
    parser.add_argument('--lora-rank', type=int, default=None,
                        help='The rank of the LoRA updates. Required if --use-lora is set.')
    parser.add_argument('--lora-alpha', type=int, default=None,
                        help='The scaling factor for LoRA updates. Required if --use-lora is set.')
    parser.add_argument('--lora-modules', type=str, nargs='+', default=None,
                        help='List of target module names for LoRA (e.g., "qkv" "mlp"). Required if --use-lora is set.')

    # Resolution scaling arguments
    parser.add_argument('--resolution-scales', type=float, nargs='+', default=[1.0],
                        help='List of resolution scaling factors to apply during test-time (e.g., 1.0 2.0 4.0).')


    args = parser.parse_args()

    # Load the trained model (with or without LoRA)
    model = load_model_for_eval(
        model_name=args.model,
        model_weights_path=args.model_weights,
        use_registers=args.use_registers,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_modules=args.lora_modules
    )

    # Create the test dataloader
    test_dataloader = create_test_dataloader(
        root_dir=args.dataset_path / "Test", # Point to the Test subdirectory
        sport_name=args.sport_name,
        batch_size=args.batch_size,
        crop_size=args.crop_size # Use the specified crop size
    )

    # Perform evaluation
    evaluation_metrics = evaluate(
        model=model,
        test_dataloader=test_dataloader,
        sport_name=args.sport_name,
        resolution_scales=args.resolution_scales, # Pass the resolution scales
        base_crop_size=args.crop_size # Pass the base crop size
    )

    # Print final evaluation metrics
    print("\n--- Evaluation Results ---")
    for metric_name, metric_value in evaluation_metrics.items():
        # Assuming some metrics like RMSE, AbsRel, etc. are typically reported as 1e3 * value
        # Adjust formatting based on the actual metrics returned by eval_depth_maps
        if 'loss' in metric_name.lower():
             print(f"{metric_name}: {metric_value:.4f}")
        else:
             print(f"{metric_name}: {metric_value*1e3:.4f} (x1e3)")
    print("--------------------------")

if __name__ == '__main__':
    main()
