# PyTorch GPU Setup Guide

Your project has been successfully converted from TensorFlow to PyTorch with GPU support!

## What Changed

All Python files have been updated to use PyTorch instead of TensorFlow:
- ✅ `train.py` - Now uses PyTorch DataLoaders and GPU training
- ✅ `model_baseline_A.py` - Converted to PyTorch nn.Module with GPU support
- ✅ `preprocess.py` - Now creates PyTorch DataLoaders
- ✅ `evaluate.py` - Updated for PyTorch model evaluation
- ✅ `check_gpu.py` - Now checks PyTorch CUDA availability
- ✅ `requirements.txt` - Updated with PyTorch dependencies

## Installation Steps

### 1. Install PyTorch with CUDA Support

For CUDA 11.8 (most common):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU only (not recommended for training):
```powershell
pip install torch torchvision torchaudio
```

### 2. Install Other Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Verify GPU Setup

Run the GPU check script:
```powershell
python check_gpu.py
```

You should see:
- ✓ PyTorch version
- ✓ CUDA available: True
- ✓ GPU detected with name and memory info
- ✓ Successful GPU computation test

## Key Changes in Usage

### Training
The model will now automatically use GPU if available:
```powershell
python train.py --epochs 20 --batch_size 8
```

### Model Files
- Model checkpoints are now saved as `.pth` files (PyTorch format) instead of `.h5` (TensorFlow/Keras)
- Previous TensorFlow checkpoints are incompatible and need retraining

### GPU Memory Management
PyTorch handles GPU memory differently:
- Memory is allocated dynamically
- Use `torch.cuda.empty_cache()` if you run into memory issues
- Monitor GPU usage with `nvidia-smi` command

## Benefits of PyTorch

1. **Better GPU Utilization** - PyTorch generally has better GPU performance
2. **More Flexible** - Easier to debug and customize
3. **Industry Standard** - Widely used in research and production
4. **Better Multi-GPU Support** - If you have multiple GPUs
5. **Native Python** - More pythonic and intuitive

## Troubleshooting

### No GPU Detected
1. Check NVIDIA driver: `nvidia-smi`
2. Verify CUDA installation
3. Reinstall PyTorch with correct CUDA version

### Out of Memory Errors
1. Reduce batch size: `--batch_size 4`
2. Clear cache: Add `torch.cuda.empty_cache()` calls
3. Use mixed precision training (FP16)

### Import Errors
Make sure all dependencies are installed:
```powershell
pip install -r requirements.txt
```

## Next Steps

1. Run `check_gpu.py` to verify GPU setup
2. Start training with `python train.py`
3. Monitor GPU usage with `nvidia-smi` in another terminal
4. Evaluate model with `python evaluate.py --model_weights results/best_model_*.pth`

Enjoy faster training with PyTorch GPU acceleration! 🚀
