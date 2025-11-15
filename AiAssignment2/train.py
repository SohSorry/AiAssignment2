"""
Training script for baseline models
Handles data loading, preprocessing, model training, and checkpoint saving
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import numpy as np

from data_loader import AudioDataLoader
from preprocess import AudioPreprocessor
from model_baseline_A import Wav2Vec2Baseline

def setup_gpu():
    """Configure GPU settings"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU Available: {torch.cuda.device_count()} GPU(s)")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        device = torch.device('cpu')
        print("⚠ No GPU found. Using CPU.")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    return device

def main():
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--data_path', type=str, 
                       default='SAND_Challenge_task1_dataset/task1/training',
                       help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--dry_run', action='store_true',
                       help='Dry run - test setup without training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max_duration', type=float, default=5.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Setup GPU
    print("="*60)
    print("Setting up GPU...")
    print("="*60)
    device = setup_gpu()
    
    # Load data
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    loader = AudioDataLoader(args.data_path, sample_rate=16000)
    audio_data, labels, file_paths = loader.load_dataset(max_duration=args.max_duration)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        audio_data, labels, test_size=0.2, val_size=0.1
    )
    
    # Preprocess data
    print("\n" + "="*60)
    print("Preprocessing audio...")
    print("="*60)
    preprocessor = AudioPreprocessor(model_name="facebook/wav2vec2-base")
    
    # Create PyTorch DataLoaders
    train_dataset, train_loader = preprocessor.create_dataloader(
        X_train, y_train, batch_size=args.batch_size, shuffle=True
    )
    val_dataset, val_loader = preprocessor.create_dataloader(
        X_val, y_val, batch_size=args.batch_size, shuffle=False
    )
    test_dataset, test_loader = preprocessor.create_dataloader(
        X_test, y_test, batch_size=args.batch_size, shuffle=False
    )
    
    # Get input shape
    sample_features = preprocessor.process_audio(X_train[0])
    input_shape = sample_features.shape
    
    # Build model
    print("\n" + "="*60)
    print("Building model...")
    print("="*60)
    model = Wav2Vec2Baseline(
        num_classes=len(loader.get_class_names()),
        learning_rate=args.learning_rate
    )
    model.build_model(input_shape)
    model.to(device)
    
    # Setup checkpoint path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(args.results_dir, f'best_model_{timestamp}.pth')
    log_path = os.path.join(args.results_dir, f'training_log_{timestamp}.csv')
    
    # Dry run check
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - Testing file generation")
        print("="*60)
        
        # Test creating output files
        import csv
        test_results = {
            'model': 'Wav2Vec2_Baseline_A',
            'hyperparameters': model.get_hyperparameters(),
            'training_args': vars(args),
            'test_loss': 0.0,
            'test_accuracy': 0.0,
            'best_val_accuracy': 0.0,
            'timestamp': timestamp,
            'note': 'DRY RUN - No actual training performed'
        }
        
        # Save test results JSON
        test_results_file = os.path.join(args.results_dir, f'dryrun_results_{timestamp}.json')
        with open(test_results_file, 'w') as f:
            json.dump(test_results, f, indent=4)
        print(f"✓ Test results JSON created: {test_results_file}")
        
        # Save test CSV log
        test_log_file = os.path.join(args.results_dir, f'dryrun_log_{timestamp}.csv')
        with open(test_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'val_f1', 'lr'])
            writer.writerow([0, 0.0, 0.0, 0.0, 0.0, 0.0, args.learning_rate])
        print(f"✓ Test CSV log created: {test_log_file}")
        
        # Test model save
        test_checkpoint = os.path.join(args.results_dir, f'dryrun_model_{timestamp}.pth')
        model.save_model(test_checkpoint)
        print(f"✓ Test model checkpoint created: {test_checkpoint}")
        
        print("\n" + "="*60)
        print("✓ All setup completed successfully!")
        print("✓ Data loaded and preprocessed")
        print("✓ Model built and ready")
        print("✓ Device configured")
        print("✓ Output files can be generated")
        print("="*60)
        print("\nTo run actual training, remove --dry_run flag")
        return
    
    # Train model
    print("\n" + "="*60)
    print("Training model...")
    print("="*60)
    
    history = model.fit_model(
        train_loader, 
        val_loader, 
        epochs=args.epochs,
        device=device,
        checkpoint_path=checkpoint_path,
        log_path=log_path
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    test_loss, test_accuracy = model.evaluate(test_loader, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save results
    results = {
        'model': 'Wav2Vec2_Baseline_A',
        'hyperparameters': model.get_hyperparameters(),
        'training_args': vars(args),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'best_val_accuracy': float(max(history['val_accuracy'])),
        'timestamp': timestamp
    }
    
    results_file = os.path.join(args.results_dir, f'results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {results_file}")
    print("\nTraining completed successfully! ✓")

if __name__ == '__main__':
    main()
