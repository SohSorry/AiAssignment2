"""
Quick script to check GPU/CUDA availability for PyTorch
"""

import torch
import sys

print("="*70)
print("GPU/CUDA Configuration Check for PyTorch")
print("="*70)

# PyTorch version
print(f"\n1. PyTorch Version: {torch.__version__}")

# CUDA availability
print(f"\n2. CUDA Available: {torch.cuda.is_available()}")

# GPU availability
print(f"\n3. GPU Availability:")
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"   ✓ {num_gpus} GPU(s) detected:")
    for i in range(num_gpus):
        print(f"     - GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"       Compute Capability: {torch.cuda.get_device_capability(i)}")
        print(f"       Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("   ✗ No GPU detected - PyTorch will use CPU")

# CUDA version (if available)
print(f"\n4. CUDA Version:")
if torch.cuda.is_available():
    print(f"   PyTorch CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"   cuDNN Enabled: {torch.backends.cudnn.enabled}")
else:
    print("   PyTorch is NOT built with CUDA support")
    print("   → Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

# Test GPU computation
print(f"\n5. GPU Computation Test:")
try:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device)
        c = torch.matmul(a, b)
        print("   ✓ GPU computation successful!")
        print(f"   Result:\n{c.cpu().numpy()}")
        print(f"   Tensor is on: {c.device}")
    else:
        print("   ✗ No GPU available for testing")
except Exception as e:
    print(f"   ✗ GPU computation failed: {e}")
    print("   → GPU may not be available or configured properly")

# Memory info (if GPU available)
if torch.cuda.is_available():
    print(f"\n6. GPU Memory Info:")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i} ({torch.cuda.get_device_name(i)}):")
        print(f"   - Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"   - Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        print(f"   - Max Allocated: {torch.cuda.max_memory_allocated(i) / 1024**2:.2f} MB")

# Additional backend info
print(f"\n7. Backend Information:")
print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
print(f"   cuDNN Deterministic: {torch.backends.cudnn.deterministic}")

print("\n" + "="*70)
print("Summary:")
if torch.cuda.is_available():
    print("✓ Your system is ready to use GPU acceleration with PyTorch!")
    print("✓ Training will automatically use the GPU")
    print(f"✓ Default GPU: {torch.cuda.get_device_name(0)}")
else:
    print("✗ No GPU detected")
    print("→ Install PyTorch with CUDA support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("→ Check NVIDIA drivers and CUDA installation")
    print("→ Or proceed with CPU (slower training)")
print("="*70)
