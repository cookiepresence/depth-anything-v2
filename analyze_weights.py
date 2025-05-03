import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Literal

# Define MODEL_CONFIG and DEVICE
# Set DEVICE based on your environment, 'cuda' if a GPU is available, 'cpu' otherwise
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
print(f"Using device: {DEVICE}")

MODEL_CONFIG = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vits_r': {'encoder': 'vits_r', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

from depth_anything_v2.dpt import DepthAnythingV2

def load_model(model_name: str, use_registers: bool = False, model_weights: str = None):
    """
    Loads the DepthAnythingV2 model with specified weights.
    """
    model_name_r = model_name + '_r' if use_registers else model_name
    if model_name_r not in MODEL_CONFIG:
        raise ValueError(f"Model name {model_name_r} not found in MODEL_CONFIG. Available models: {list(MODEL_CONFIG.keys())}")

    model_config = MODEL_CONFIG[model_name_r]
    # Instantiate the model using the potentially imported or dummy class
    depth_anything = DepthAnythingV2(**model_config)

    if model_weights is not None:
        model_weights_path = Path(model_weights)
        if not model_weights_path.exists():
            raise FileNotFoundError(f"Model weights file not found at: {model_weights_path}")
        print(f"Loading state dict from {model_weights_path}")
        # Load state dict, mapping to the determined device
        state_dict = torch.load(model_weights_path, weights_only=True, map_location=DEVICE)

        # Load state dict, handling potential mismatches with strict=False
        try:
            depth_anything.load_state_dict(state_dict, strict=True)
            print("State dict loaded with strict=True.")
        except RuntimeError as e:
            print(f"Strict loading failed: {e}. Trying with strict=False.")
            try:
                depth_anything.load_state_dict(state_dict, strict=False)
                print("State dict loaded with strict=False.")
            except Exception as e_false:
                print(f"Loading with strict=False also failed: {e_false}")
                print("Continuing analysis, but some weights might not be loaded correctly.")


    depth_anything = depth_anything.to(DEVICE)
    return depth_anything

def analyze_weight_updates(base_model: torch.nn.Module, finetuned_model: torch.nn.Module):
    """
    Analyzes the weight updates between base and finetuned models.
    """
    print("Analyzing weight updates...")

    weight_updates_analysis = {}

    # Ensure parameters are on the same device before calculating difference
    base_params = dict(base_model.named_parameters())
    finetuned_params = dict(finetuned_model.named_parameters())

    analyzed_layers_count = 0

    for name, base_param in base_params.items():
        if name in finetuned_params:
            finetuned_param = finetuned_params[name]

            if base_param.shape == finetuned_param.shape:
                analyzed_layers_count += 1
                # Move parameters to CPU for analysis if on GPU, to avoid CUDA memory issues
                # and ensure compatibility with numpy for singular values analysis later
                update = finetuned_param.data.cpu() - base_param.data.cpu()

                # Calculate the rank of the update matrix
                # For tensors with more than 2 dimensions, flatten the last two for rank calculation
                if update.dim() > 1: # Consider parameters with more than one dimension (weights)
                    if update.dim() > 2:
                        original_shape = update.shape
                        # Flatten all leading dimensions into one, keeping the last dimension
                        update_matrix = update.view(-1, update.shape[-1])
                    else: # Already 2D
                        update_matrix = update

                    if update_matrix.numel() > 0: # Ensure matrix is not empty
                        try:
                            # Calculate rank
                            rank = torch.linalg.matrix_rank(update_matrix).item()
                        except Exception as e:
                            rank = f"Error calculating rank: {e}"

                        # Perform SVD and analyze singular values for low-rank approximability
                        try:
                             # Perform SVD on CPU tensor. full_matrices=False is more efficient.
                             U, S, Vh = torch.linalg.svd(update_matrix, full_matrices=False)
                             singular_values = S.numpy() # Already on CPU

                             # Analyze singular values: check for rapid decay
                             singular_values_info = {
                                 'count': len(singular_values),
                                 'sum': np.sum(singular_values),
                                 'first_5': singular_values[:min(5, len(singular_values))].tolist(), # First few singular values
                                 # Example decay metric: ratio of the first to the fifth singular value
                                 'decay_ratio_1_to_5': (singular_values[0] / singular_values[4]) if len(singular_values) >= 5 and singular_values[4] != 0 else 'N/A'
                             }
                        except torch.linalg.LinAlgError:
                            singular_values_info = "SVD did not converge"
                        except Exception as e:
                            singular_values_info = f"Error during SVD: {e}"

                    else:
                        rank = 0
                        singular_values_info = "Matrix is empty, SVD not applicable"

                else: # 1D tensor (bias) - rank is 1 if not all zeros, SVD not typically applied
                    rank = 1 if torch.any(update != 0) else 0
                    singular_values_info = 'N/A (1D tensor)'


                weight_updates_analysis[name] = {
                    'shape': tuple(update.shape),
                    'rank': rank,
                    'singular_values_info': singular_values_info
                }
            else:
                weight_updates_analysis[name] = {
                    'shape': tuple(base_param.shape), # Report base shape for mismatch
                    'rank': 'Shape mismatch',
                    'singular_values_info': 'Shape mismatch'
                }
        else:
             weight_updates_analysis[name] = {
                'shape': tuple(base_param.shape),
                'rank': 'Missing in finetuned model',
                'singular_values_info': 'Missing in finetuned model'
            }

    # Print the analysis results
    print(f"\n--- Analysis Results for {analyzed_layers_count} Layers ---")
    for name, analysis in weight_updates_analysis.items():
        print(f"Layer: {name}")
        print(f"  Shape: {analysis['shape']}")
        print(f"  Rank: {analysis['rank']}")
        if analysis['singular_values_info'] not in ['N/A (not a matrix)', 'Shape mismatch', 'Missing in finetuned model', 'N/A (1D tensor)']:
            print(f"  Singular Values Info:")
            if isinstance(analysis['singular_values_info'], dict):
                for k, v in analysis['singular_values_info'].items():
                    print(f"    {k}: {v}")
            else:
                 print(f"    {analysis['singular_values_info']}")
        print("-" * 20)

    return weight_updates_analysis

def print_summary(weight_updates_analysis):
    """
    Prints a summary of the weight update analysis.
    """
    print("\n--- Summary ---")
    low_rank_updates_count = 0
    low_rank_approximable_count = 0
    total_layers_analyzed_for_rank = 0
    total_layers_analyzed_for_svd = 0

    for name, analysis in weight_updates_analysis.items():
        if isinstance(analysis['rank'], int): # Check if rank was calculated and is an integer
            total_layers_analyzed_for_rank += 1
            # Heuristic: Consider low rank if rank is significantly smaller than the smallest dimension
            # For flattened matrices, min_dim is the smaller of the original dimensions.
            # Using the last dimension of the original shape is a common heuristic for weight matrices.
            original_shape = analysis['shape']
            if len(original_shape) > 1:
                 min_dim = min(original_shape) # Consider the minimum of the original shape dimensions
                 if analysis['rank'] > 0 and analysis['rank'] < min_dim / 10: # Example heuristic threshold (e.g., rank < 10% of min dimension)
                     low_rank_updates_count += 1

        if isinstance(analysis['singular_values_info'], dict): # Check if SVD was performed
             total_layers_analyzed_for_svd += 1
             # Heuristic: Consider low-rank approximable if singular values decay rapidly
             if 'decay_ratio_1_to_5' in analysis['singular_values_info']:
                  decay_ratio = analysis['singular_values_info']['decay_ratio_1_to_5']
                  # Example heuristic threshold for rapid decay (e.g., the first singular value is at least 10 times larger than the fifth)
                  if isinstance(decay_ratio, float) and decay_ratio > 10:
                      low_rank_approximable_count += 1

    print(f"Total layers with calculable rank: {total_layers_analyzed_for_rank}")
    print(f"Total layers with SVD performed: {total_layers_analyzed_for_svd}")
    print(f"Layers with potentially low-rank updates (rank < 10% of min dimension): {low_rank_updates_count}")
    print(f"Layers with potentially low-rank approximable updates (1st SV / 5th SV > 10): {low_rank_approximable_count}")

    if total_layers_analyzed_for_rank > 0:
        print(f"% of layers with potentially low-rank updates: {(low_rank_updates_count / total_layers_analyzed_for_rank) * 100:.2f}%")
    else:
        print("No layers analyzed for low-rank updates.")

    if total_layers_analyzed_for_svd > 0:
        print(f"% of layers with potentially low-rank approximable updates: {(low_rank_approximable_count / total_layers_analyzed_for_svd) * 100:.2f}%")
    else:
        print("No layers analyzed for low-rank approximability.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze weight updates between base and finetuned models.')
    parser.add_argument('--base-model-path', type=str, required=True,
                        help='Path to the base model weights file (.pth)')
    parser.add_argument('--finetuned-model-path', type=str, required=True,
                        help='Path to the finetuned model weights file (.pth)')
    parser.add_argument('--model-name', type=str, choices=list(MODEL_CONFIG.keys()), required=True,
                        help='Name of the model architecture (e.g., vits, vitb, vitl, vitg)')
    # Add an argument for using registers if applicable to your models
    parser.add_argument('--use-registers', action='store_true',
                        help='Specify if the models use the registers version (e.g., vits_r)')


    args = parser.parse_args()

    try:
        # Load the base and finetuned models using paths from arguments
        base_model = load_model(args.model_name, use_registers=args.use_registers, model_weights=args.base_model_path)
        finetuned_model = load_model(args.model_name, use_registers=args.use_registers, model_weights=args.finetuned_model_path)

        # Perform the analysis
        analysis_results = analyze_weight_updates(base_model, finetuned_model)

        # Print the summary
        print_summary(analysis_results)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
