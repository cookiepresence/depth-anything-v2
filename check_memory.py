from depth_anything_v2.dpt import DepthAnythingV2
import torch

def get_model_memory_MB(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (param_size + buffer_size) / (1024 ** 2)
    print(f"Model memory (parameters + buffers): {total_size:.2f} MB")

# Example:
# model = MyModel()

MODEL_CONFIG = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vits_r': {'encoder': 'vits_r', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

model_config = MODEL_CONFIG['vitl']

depth_anything = DepthAnythingV2(**model_config)
depth_anything.load_state_dict(torch.load("saved_models/quantized_model.pth", weights_only=True), strict=False)
print(f"Model size in memory: {get_model_size(model):.2f} MB")
