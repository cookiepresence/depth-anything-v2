import time
import torch
import argparse
from models import get_model  # Import the function that returns the model
from utils import load_dataset, poison_dataset
from hqq.core.quantize import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def measure_inference_time(model, input_tensor, num_iterations=1000):
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    # Warm-up runs
    for _ in range(10):
        _ = model(input_tensor)

    # Measure inference time over multiple iterations
    start_time = time.time()
    for _ in range(num_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()  # Ensure GPU operations are complete
        _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_iterations
    return avg_inference_time

def measure_memory_usage(model, input_tensor):
    torch.cuda.empty_cache()  # Clear cache before measuring
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    torch.cuda.reset_peak_memory_stats(device)  # Reset peak memory tracking
    _ = model(input_tensor)  # Run one inference
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # In MB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # In MB
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Peak memory

    return allocated, reserved, peak

def quantize_linear_layers(module, config, device):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, HQQLinear(
                getattr(module, name),
                quant_config=config,
                compute_dtype=torch.float,
                device=device,
                initialize=True,
                del_orig=True
            ))
        else:
            quantize_linear_layers(child, config, device)

def get_quantized_model(model_instance, model_path_name, quantization_level="4bit"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance.load_state_dict(torch.load(model_path_name, map_location=device))
    model_instance.to(device)
    model_instance.eval()
    
    quantization_bits = {"2bit": 2, "4bit": 4, "8bit": 8}
    nbits = quantization_bits.get(quantization_level, 4)
    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=32)
    
    quantize_linear_layers(model_instance, config=quant_config, device=device)
    return model_instance

def main():
    parser = argparse.ArgumentParser(description="Quantize ViT model using bitsandbytes")
    parser.add_argument("--n_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--n_attention_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--forward_mul", type=int, default=4, help="Forward expansion multiplier")
    parser.add_argument("--image_size", type=int, default=32, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size for ViT")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--model-path", dest='model_path', type=str, default="cifar10_vit.pth", help="Path to trained model")
    parser.add_argument("--output-path", dest='output_path', type=str, default="vit_quantized_cifar10.pth", help="Path to save quantized model")
    parser.add_argument("--quantization-level", type=str, choices=["2bit", "4bit", "8bit"], default="4bit", help="Quantization level")
    args = parser.parse_args()
    
    num_classes = 100  # CIFAR-10 has 100 classes
    
    model = get_model("vit", num_classes).to(device)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    print("Before quantization:")
    print(f"Inference Time: {measure_inference_time(model, input_tensor, 1000):.6f} seconds")
    allocated, reserved, peak = measure_memory_usage(model, input_tensor)
    print(f"Memory Allocated: {allocated:.2f} MB, Memory Reserved: {reserved:.2f} MB, Peak Memory: {peak:.2f} MB")
    
    # Quantize the model
    model = get_quantized_model(model, args.model_path, args.quantization_level)
    
    print("\nAfter quantization:")
    print(f"Inference Time: {measure_inference_time(model, input_tensor, 1000):.6f} seconds")
    allocated, reserved, peak = measure_memory_usage(model, input_tensor)
    print(f"Memory Allocated: {allocated:.2f} MB, Memory Reserved: {reserved:.2f} MB, Peak Memory: {peak:.2f} MB")
    
    # Save the quantized model
    torch.save(model.state_dict(), args.output_path)
    print(f"\nQuantized model saved as {args.output_path}")

if __name__ == "__main__":
    main()