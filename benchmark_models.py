import sys
import os
import time
import torch
import torch.nn.functional as F

try:
    from utils import load_config
    from modelinfer import SAMRoadplus
except ImportError:
    print("Could not import SAMRoadplus. Make sure you are in the samroadplus directory.")
    sys.exit(1)

def get_config_path(backbone_name):
    # Map backbone name to its config file
    config_dir = os.path.abspath('./config')
    mapping = {
        'resnet50': 'toponet_resnet50_512_globalscale.yaml',
        'dinov2': 'toponet_dinov2_512_globalscale.yaml',
        'dinov3': 'toponet_dinov3_512_globalscale.yaml', 
        'radio': 'toponet_radio_512_globalscale.yaml',
        'sam': 'toponet_vitb_512_globalscale.yaml',
        'sam2': 'toponet_sam2_512_globalscale.yaml'
    }
    
    file_name = mapping.get(backbone_name)
    if not file_name:
        return None
    path = os.path.join(config_dir, file_name)
    return path if os.path.exists(path) else None

def benchmark_model(backbone_name, device="cuda"):
    config_path = get_config_path(backbone_name)
    if not config_path:
        print(f"Skipping {backbone_name}: Config {config_path} not found.")
        return

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Failed to load config for {backbone_name}: {e}")
        return

    print(f"--- Benchmarking {backbone_name.upper()} ---")
    
    # Initialize model
    try:
        model = SAMRoadplus(config)
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"Failed to initialize {backbone_name}: {e}")
        return
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.image_encoder.parameters()) if hasattr(model, 'image_encoder') else 0
    print(f"Total Params: {total_params / 1e6:.2f} M (Backbone: {backbone_params / 1e6:.2f} M)")

    # Prepare dummy input
    dummy_input = torch.randn(1, 512, 512, 3, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model.infer_masks_and_img_features(dummy_input)

    # Benchmark Latency & Memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    num_runs = 50
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_event.record()
            _ = model.infer_masks_and_img_features(dummy_input)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
            
    avg_latency = sum(times) / len(times)
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    print(f"Inference Latency: {avg_latency:.2f} ms")
    print(f"Peak Inference VRAM: {peak_mem_mb:.2f} MB")
    
    # Try to calculate MACs using thop if available
    macs_str = "N/A"
    try:
        import thop
        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m.infer_masks_and_img_features(x)
        
        macs, _ = thop.profile(Wrapper(model), inputs=(dummy_input,), verbose=False)
        macs_str = f"{macs / 1e9:.2f} G"
    except ImportError:
        macs_str = "(thop not installed)"
    except Exception as e:
        macs_str = f"Error computing FLOPs: {e}"
        
    print(f"Compute (MACs): {macs_str}")
    print("------------------------------------------\n")
    
    # Free memory
    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    models_to_test = ['resnet50', 'dinov2', 'dinov3', 'radio', 'sam', 'sam2']
    for m in models_to_test:
        benchmark_model(m)
