import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--img_folder', type=str, required=True, help='Path to folder containing images')
    parser.add_argument('--gt_file', type=str)
    parser.add_argument('--gt_folder', type=str, required=True, help='Path to folder containing ground truth depth maps')

    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred_only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--finetune_name', type=str, help='Name of finetuned model (required if not using pretrained)')

    parser.add_argument('--use_registers', action='store_true', help='use dino with registers?')

    parser.add_argument('--extension', type=str, default='.png', help='File extension of images to process')

    args = parser.parse_args()

    if not args.pretrained and args.finetune_name is None:
        parser.error("--finetune-name is required when --pretrained is not set.")

    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vits_r': {'encoder': 'vits_r', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = args.encoder
    if args.use_registers:
        encoder = args.encoder + '_r'
    
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    if args.pretrained:
        print(f"Using pretrained model for interence.")
        depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location=DEVICE, weights_only=True), strict=False)
    else:
        print(f"Using finetuned model for inference")
        depth_anything.load_state_dict(torch.load(f'saved_models/{args.finetune_name}', weights_only=True, map_location=DEVICE), strict=False)

    if args.use_registers:
        depth_anything.pretrained.load_state_dict(torch.load(f'checkpoints/dinov2-with-registers-{args.encoder}.pt', map_location=DEVICE, weights_only=True))

    depth_anything = depth_anything.to(DEVICE).eval()

    if args.img_path is None:
        image_files = sorted(glob.glob(os.path.join(args.img_folder, f'*{args.extension}')))
        gt_files = [os.path.join(args.gt_folder, os.path.basename(f)) for f in image_files]
    
    else :
        image_files = [args.img_path]
        gt_files = [args.gt_file]

    # if os.path.isfile(args.img_path):
    #     if args.img_path.endswith('txt'):
    #         with open(args.img_path, 'r') as f:
    #             filenames = f.read().splitlines()
    #     else:
    #         filenames = [args.img_path]
    # else:
    #     filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    print(f"Found {len(image_files)} images to process")
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, (filename, filename_gt) in enumerate(tqdm(zip(image_files, gt_files))):

        if k>100:
            break

        if not os.path.exists(filename_gt):
            print(f"Warning: ground truth file not found: {filename_gt}, skippnig")
            continue        
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"Warning: Could not read image file: {filename}, skipping")
            continue
            
        depth_gt = cv2.imread(filename_gt)
        if depth_gt is None:
            print(f"Warning: Could not read ground truth file: {filename_gt}, skipping")
            continue

        # print(depth_gt.shape)
        
        depth = depth_anything.infer_image(raw_image, args.input_size)

        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        output_filename = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth, split_region, depth_gt])

            cv2.imwrite(output_filename, combined_result)
            
            # if args.finetune_name is not None:
            #     cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_finetune_masked' + '.png'), combined_result)
            # else:
            #     cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
