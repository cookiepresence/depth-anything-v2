import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.v2 as v2
from PIL import Image
import cv2

from depth_anything_v2.util.transform import Resize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DepthEstimationDataset(Dataset):
    """
    PyTorch Dataset for monocular depth estimation that loads images from a folder structure
    along with metadata.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        sport_name: str,
        transform=None,
        target_transform=None,
        crop_size: int = 518,
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing game folders
            sport_name: Sport name to include in metadata
            transform: Optional transforms to apply to the source images
            target_transform: Optional transforms to apply to the depth images
            crop_size: Size to crop shorter side to (default: 518 for VIT)
        """
        self.root_dir = Path(root_dir)
        self.sport_name = sport_name
        self.transform = transform
        self.target_transform = target_transform
        self.crop_size = crop_size
        
        # Collect all image paths and metadata
        self.samples = self._collect_samples()
        
    def _collect_samples(self) -> List[Dict]:
        """Collect all valid samples with their paths and metadata."""
        samples = []
        # Iterate through game folders
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
            
            # Process each video in the game folder
            for video_idx, video_metadata in enumerate(video_metadatas, 1):
                video_folder = game_folder / f"video_{video_idx}"
                
                if not video_folder.exists():
                    continue
                    
                # Get color and depth_r folders
                color_folder = video_folder / "color"
                depth_r_folder = video_folder / "depth_r"
                
                if not color_folder.exists() or not depth_r_folder.exists():
                    continue
                
                # Get number of frames from metadata
                num_frames = int(video_metadata.get("Number of frames", 0))
                
                # Collect all valid frame pairs
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
        """Return the total number of samples."""
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
        color_img = np.array(Image.open(sample_info["color_path"]).convert("RGB")) / 255.0
        
        # Load depth image (16-bit)
        depth_img = np.array(Image.open(sample_info["depth_path"]))
        
        # Center crop to target size
        resize_transform = v2.Compose([
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
        ])
        sample = resize_transform({"image": color_img, "depth": depth_img})
        color_img = sample['image']
        depth_img = sample['depth']
        
        # Normalize depth (16-bit depth to normalized float)
        depth_normalized = self._normalize_depth(depth_img)
        
        # Apply transformations if provided
        if self.transform:
            color_img = self.transform(color_img)
        else:
            color_img = v2.Compose([
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])(color_img)
            color_img = color_img.to(torch.float)
        
        if self.target_transform:
            depth_tensor = self.target_transform(depth_normalized)
        else:
            # Default: convert normalized depth to tensor
            depth_tensor = depth_normalized
        
        # Extract metadata
        metadata = {
            "game_number": sample_info["game_number"],
            "video_number": sample_info["video_number"],
            "frame_number": sample_info["frame_number"],
            "sport_name": sample_info["sport_name"],
            "total_frames": sample_info["total_frames"]
        }
        return color_img, depth_tensor, metadata

    def _normalize_depth(self, depth_array: np.ndarray) -> np.ndarray:
        """
        Normalize the 16-bit depth map to [0, 1] range.
        
        Args:
            depth_array: Raw depth array (16-bit)
            
        Returns:
            Normalized depth array as float32 in range [0, 1]
        """
        # Handle edge case of empty depth
        depth_array = depth_array.to(torch.float)
        if depth_array.max() == depth_array.min():
            return torch.zeros_like(depth_array)

        mask = (depth_array > 0) & (depth_array < 65536)

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
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        root_dir: Root directory with the data
        sport_name: Sport name to include in metadata
        batch_size: Batch size for dataloaders
        crop_size: Size to crop the shorter side to
        train_val_split: Fraction of data to use for training
        num_workers: Number of workers for dataloaders
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create the dataset
    train_dataset = DepthEstimationDataset(
        root_dir=root_dir / "Train",
        sport_name=sport_name,
        crop_size=crop_size
    )

    val_dataset = DepthEstimationDataset(
        root_dir=root_dir / "Validation",
        sport_name=sport_name,
        crop_size=crop_size
    )

    print(train_dataset[0])
    print(val_dataset[0])

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader
