import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as TF
from PIL import Image
import cv2

from depth_anything_v2.util.transform import Resize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DepthEstimationDataset(Dataset):
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        sport_name: str,
        transform=None,
        target_transform=None,
        crop_size: int = 518,
        apply_augmentations: bool = False  # Flag to control augmentations
    ):
        """        
        Args:
            root_dir: Root directory containing game folders
            sport_name: Sport name to include in metadata
            transform: Optional transforms to apply to the source images
            target_transform: Optional transforms to apply to the depth images
            crop_size: Size to crop shorter side to (default: 518 for VIT)
            apply_augmentations: Whether to apply training augmentations
        """
        self.root_dir = Path(root_dir)
        self.sport_name = sport_name
        self.transform = transform
        self.target_transform = target_transform
        self.crop_size = crop_size
        
        # Collect all image paths and metadata
        self.samples = self._collect_samples()
        self.apply_augmentations=apply_augmentations
        
        # Define augmentations for training
        if self.apply_augmentations:
            # pass
            self.augmentations = v2.Compose([ 
                # Random cropping with padding
                # v2.RandomResizedCrop(
                #     size=(crop_size, crop_size)
                # ),
                # Strong color jitter
                v2.ColorJitter(
                    brightness=0.1,
                    hue=0.1
                ),
                # v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                # v2.RandomAutocontrast(p=0.3),
                # v2.RandomEqualize(p=0.2),
                # # Random Gaussian blur
                v2.GaussianBlur(
                    kernel_size=(5, 5),
                    sigma=(0.1, 0.1)
                )
                # Additional color distortions
                # Random grayscale to simulate challenging lighting
                # v2.RandomGrayscale(p=0.1),
                # Random perspectives to simulate different viewpoints
                # v2.RandomPerspective(distortion_scale=0.2, p=0.3),
            ])

        self.resize_transform = v2.Compose([
            Resize(
                width=self.crop_size,
                height=self.crop_size,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.1)
        ])

    def _collect_samples(self) -> List[Dict]:
        """Collect all valid samples with their paths and metadata."""
        samples = []
        # Iterating through game folders
        for game_folder in sorted(self.root_dir.glob("game_*")):
            game_number = int(game_folder.name.split("_")[1])
            
            # Load metadata from JSON file
            json_file = game_folder / f"{game_folder.name}.json"
            if not json_file.exists():
                continue
                
            with open(json_file, "r") as f:
                json_data = json.load(f)
            
            # Handle both single game and multiple game JSON formats
            if isinstance(json_data, list):
                video_metadatas = json_data
            else:
                video_metadatas = [json_data]
            
            # Processing each video in the game folder
            for video_idx, video_metadata in enumerate(video_metadatas, 1):
                video_folder = game_folder / f"video_{video_idx}"
                
                if not video_folder.exists():
                    continue
                    
                # Getting color and depth_r folders
                color_folder = video_folder / "color"
                depth_r_folder = video_folder / "depth_r"
                
                if not color_folder.exists() or not depth_r_folder.exists():
                    continue
                
                num_frames = int(video_metadata.get("Number of frames", 0))
                
                for frame_path in sorted(color_folder.glob("*.png")):
                    frame_number = int(frame_path.stem)
                    depth_path = depth_r_folder / f"{frame_number}.png"
                    
                    if depth_path.exists():
                        samples.append({
                            "color_path": str(frame_path),
                            "depth_path": str(depth_path),
                            "game_number": game_number,
                            "video_number": video_idx,
                            "frame_number": frame_number,
                            "sport_name": self.sport_name,
                            "total_frames": num_frames
                        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load a single item by index.
        
        Returns:
            Dict containing:
                - image: RGB image tensor
                - depth: Normalized depth tensor
                - metadata: Dict with game_number, video_number, frame_number, 
                  sport_name, and total_frames
        """
        sample_info = self.samples[idx]
        
        # Load color image
        color_img = cv2.imread(sample_info["color_path"], cv2.IMREAD_COLOR)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img = color_img / 255.0

        depth_img = cv2.imread(sample_info["depth_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
    
        # img = Image.open(sample_info["color_path"]).convert("RGB")
        # color_img = np.array(img) / 255.0
        # img.close()

        # # Load depth image (16-bit)
        # depth = Image.open(sample_info["depth_path"])
        # depth_img = np.array(depth, dtype=np.float32)
        # depth.close()
        # # print(f"closing {idx}")

        sample = self.resize_transform({"image": color_img, "depth": depth_img})
        color_img = sample['image']
        depth_img = sample['depth']
        if self.transform:
            color_img = self.transform(color_img)
        else:
            color_img = v2.Compose([
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])(color_img)
            color_img = color_img.to(torch.float)
        
        if self.apply_augmentations:
            augmented = self.augmentations({"image": color_img})
            color_img = augmented["image"]
        
        depth_normalized = self._normalize_depth(depth_img)
        
        
        if self.target_transform:
            depth_tensor = self.target_transform(depth_normalized)
        else:
            depth_tensor = depth_normalized
        
        metadata = {
            "game_number": sample_info["game_number"],
            "video_number": sample_info["video_number"],
            "frame_number": sample_info["frame_number"],
            "sport_name": sample_info["sport_name"],
            "total_frames": sample_info["total_frames"]
        }
        return color_img, depth_tensor, metadata

    def _normalize_depth(self, depth_array):
        """
        Normalize the 16-bit depth map to [0, 1] range.
        Returns:
            Normalized depth array as float32 in range [0, 1]
        """
        # Handle edge case of empty depth
        depth_array = depth_array.to(torch.float)
        if depth_array.max() == depth_array.min():
            return torch.zeros_like(depth_array)

        mask = (depth_array > 0) & (depth_array < 65536)
        # return inv_depth

        # Normalize to [0, 1]
        depth_array = 1 / depth_array
        depth_min = depth_array[mask].min()
        depth_max = depth_array[mask].max()
        normalized = (depth_array - depth_min) / (depth_max - depth_min)
        return normalized

def create_depth_dataloaders(
    root_dir: Union[str, Path],
    sport_name: str,
    train_batch_size: int = 16,
    val_batch_size: int = 16,
    crop_size: int = 518,
    num_workers: int = 8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        root_dir: Root directory with the data
        sport_name: Sport name to include in metadata
        train_batch_size: Batch size for training dataloader
        val_batch_size: Batch size for validation dataloader
        crop_size: Size to crop the shorter side to
        num_workers: Number of workers for dataloaders
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
=    train_dataset = DepthEstimationDataset(
        root_dir=root_dir / "Train",
        sport_name=sport_name,
        crop_size=crop_size,
        apply_augmentations=True  =
    )

    val_dataset = DepthEstimationDataset(
        root_dir=root_dir / "Validation",
        sport_name=sport_name,
        crop_size=crop_size,
        apply_augmentations=False  
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    torch.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    
    return train_loader, val_loader

class CutMix(torch.nn.Module):
    """
    CutMix augmentation module that can be used independently of class numbers.
    This implementation focuses on the image mixing, allowing it to be used for
    both classification and dense prediction tasks like depth estimation.

    Args:
        beta (float): Parameter for beta distribution. Default: 1.0
        patch_size (int): Size of patches to use for granular mixing. Default: None
                          If specified, cuts will be aligned to patch boundaries.
    """
    def __init__(self, beta=1.0, patch_size=None):
        super().__init__()
        self.beta = beta
        self.patch_size = patch_size

    def _rand_bbox(self, img_shape, lam):
        """
        Generate random bounding box for CutMix

        Args:
            img_shape (tuple): Shape of the image (B, C, H, W)
            lam (float): Lambda parameter (controls the size of the box)

        Returns:
            tuple: (x1, y1, x2, y2) coordinates of the box
        """
        W = img_shape[3]
        H = img_shape[2]

        cut_rat = np.sqrt(1. - lam)

        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        if self.patch_size is not None:
            # Round to nearest patch boundary
            x1 = (x1 // self.patch_size) * self.patch_size
            y1 = (y1 // self.patch_size) * self.patch_size
            # Ensure x2 and y2 are also on patch boundaries
            x2 = min(((x2 + self.patch_size - 1) // self.patch_size) * self.patch_size, W)
            y2 = min(((y2 + self.patch_size - 1) // self.patch_size) * self.patch_size, H)

        return x1, y1, x2, y2

    def forward(self, img, target=None):
        """
        Applying CutMix to a batch of images and optionally targets (like depth maps)

        Args:
            img (torch.Tensor): Batch of images (B, C, H, W)
            target (torch.Tensor, optional): Batch of targets like depth maps (B, C_t, H, W)
                                            or (B, H, W) for single-channel targets

        Returns:
            tuple: If target is provided:
                  (mixed images, mixed targets, lambda, source indices)
                  If target is None:
                  (mixed images, lambda, source indices)
        """
        batch_size = img.size(0)

        if batch_size <= 1:
            if target is not None:
                return img, target, 1.0, torch.arange(batch_size)
            return img, 1.0, torch.arange(batch_size)

        lam = np.random.beta(self.beta, self.beta)

        rand_index = torch.randperm(batch_size).to(img.device)

        x1, y1, x2, y2 = self._rand_bbox(img.shape, lam)

        mixed_img = img.clone()

        # Applying cutmix
        mixed_img[:, :, y1:y2, x1:x2] = img[rand_index, :, y1:y2, x1:x2]

        mixed_target = None
        if target is not None:
            mixed_target = target.clone()
            mixed_target[:, y1:y2, x1:x2] = target[rand_index, y1:y2, x1:x2]

        lam = 1 - ((x2 - x1) * (y2 - y1) / (img.size(2) * img.size(3)))

        if target is not None:
            return mixed_img, mixed_target, lam, rand_index
        return mixed_img, lam, rand_index
