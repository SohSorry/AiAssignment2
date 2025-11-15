"""
Quick test script to verify installation and GPU availability
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("="*60)
    print("Testing Package Imports...")
    print("="*60)
    
    packages = {
        'tensorflow': 'TensorFlow',
        'transformers': 'Transformers',
        'librosa': 'Librosa',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n⚠ Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All packages imported successfully!")
    return True

def test_gpu():
    """Test GPU availability"""
    print("\n" + "="*60)
    print("Testing GPU Availability...")
    print("="*60)
    
    try:
        import tensorflow as tf
        
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n✓ GPU Available: {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"    Memory growth enabled: ✓")
                except RuntimeError as e:
                    print(f"    Memory growth: {e}")
        else:
            print("\n⚠ No GPU detected. Training will use CPU (slower).")
            print("To use GPU, ensure:")
            print("  1. NVIDIA GPU with CUDA support is installed")
            print("  2. CUDA toolkit and cuDNN are installed")
            print("  3. TensorFlow GPU version is installed")
        
        # Test GPU computation
        if gpus:
            print("\nTesting GPU computation...")
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
            print(f"✓ GPU computation successful: {c.numpy()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing GPU: {e}")
        return False

def test_wav2vec2():
    """Test Wav2Vec2 model loading"""
    print("\n" + "="*60)
    print("Testing Wav2Vec2 Model...")
    print("="*60)
    
    try:
        from transformers import Wav2Vec2Processor, TFWav2Vec2Model
        
        print("Loading Wav2Vec2 processor...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        print("✓ Processor loaded")
        
        print("Loading Wav2Vec2 model...")
        model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        print("✓ Model loaded")
        
        # Test forward pass
        import numpy as np
        print("\nTesting forward pass...")
        dummy_input = np.random.randn(1, 16000).astype(np.float32)
        inputs = processor(dummy_input, sampling_rate=16000, return_tensors="tf")
        outputs = model(inputs.input_values)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {inputs.input_values.shape}")
        print(f"  Output shape: {outputs.last_hidden_state.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Wav2Vec2: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    print("\n" + "="*60)
    print("Testing Data Loading...")
    print("="*60)
    
    import os
    data_path = "SAND_Challenge_task1_dataset/task1/training"
    
    if not os.path.exists(data_path):
        print(f"⚠ Dataset not found at: {data_path}")
        print("Please ensure the dataset is in the correct location.")
        return False
    
    print(f"✓ Dataset found at: {data_path}")
    
    # Count files
    categories = ['phonationA', 'phonationE', 'phonationI', 'phonationO', 
                  'phonationU', 'rhythmKA', 'rhythmPA', 'rhythmTA']
    
    total_files = 0
    for cat in categories:
        cat_path = os.path.join(data_path, cat)
        if os.path.exists(cat_path):
            files = [f for f in os.listdir(cat_path) if f.endswith('.wav')]
            print(f"  {cat}: {len(files)} files")
            total_files += len(files)
    
    print(f"\n✓ Total audio files: {total_files}")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("BASELINE PIPELINE - INSTALLATION TEST")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Package Imports", test_imports()))
    
    # Test GPU
    results.append(("GPU Availability", test_gpu()))
    
    # Test Wav2Vec2
    results.append(("Wav2Vec2 Model", test_wav2vec2()))
    
    # Test data loading
    results.append(("Data Loading", test_data_loading()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now start training:")
        print("  python train.py --epochs 20 --batch_size 8")
    else:
        print("\n" + "="*60)
        print("⚠ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before training.")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
