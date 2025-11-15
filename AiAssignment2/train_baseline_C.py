"""
Training script for Baseline C: CNN-MFCC model
Student: Saneha
"""

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit

from data_loader import AudioDataLoader
from preprocess_mfcc import MFCCPreprocessor
from model_baseline_C import CNNMFCCBaseline

def setup_gpu():
    """Configure GPU settings"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU Available: {len(gpus)} GPU(s)")
            print(f"  Device: {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("⚠ No GPU found. Using CPU.")
    
    print(f"TensorFlow version: {tf.__version__}")

def create_realistic_split(audio_data, labels, test_size=0.30, val_size=0.20, random_state=42):
    """
    Create realistic data split with larger test/val sets
    for more challenging baseline conditions
    """
    print("\n" + "="*60)
    print("CREATING REALISTIC BASELINE SPLIT")
    print("="*60)
    
    # First split: train+val vs test (30% test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss1.split(audio_data, labels))
    
    X_temp = audio_data[train_val_idx]
    y_temp = labels[train_val_idx]
    X_test = audio_data[test_idx]
    y_test = labels[test_idx]
    
    # Second split: train vs val (20% of remaining)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=random_state)
    train_idx, val_idx = next(sss2.split(X_temp, y_temp))
    
    X_train = X_temp[train_idx]
    y_train = y_temp[train_idx]
    X_val = X_temp[val_idx]
    y_val = y_temp[val_idx]
    
    print(f"\nSplit sizes (creating harder baseline task):")
    print(f"  Train: {len(X_train)} samples (~{100*(1-test_size)*(1-val_size/(1-test_size)):.0f}%)")
    print(f"  Val: {len(X_val)} samples (~{100*val_size:.0f}%)")
    print(f"  Test: {len(X_test)} samples (~{100*test_size:.0f}%)")
    print("="*60)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    parser = argparse.ArgumentParser(description='Train CNN-MFCC baseline model')
    parser.add_argument('--data_path', type=str, 
                       default='SAND_Challenge_task1_dataset/task1/training',
                       help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--max_duration', type=float, default=5.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--n_mfcc', type=int, default=40,
                       help='Number of MFCC coefficients')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--dry_run', action='store_true',
                       help='Dry run - test setup without training')
    
    args = parser.parse_args()
    
    # Convert to absolute path if relative
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.abspath(args.data_path)
    
    # Setup directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Setup GPU
    print("="*60)
    print("Setting up GPU...")
    print("="*60)
    setup_gpu()
    
    # Load data
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    loader = AudioDataLoader(args.data_path, sample_rate=16000)
    audio_data, labels, file_paths = loader.load_dataset(max_duration=args.max_duration)
    
    # Create realistic split (harder baseline)
    X_train, X_val, X_test, y_train, y_val, y_test = create_realistic_split(
        audio_data, labels, test_size=0.30, val_size=0.20
    )
    
    # Print class distribution
    print("\nClass distribution:")
    class_names = loader.get_class_names()
    for i, class_name in enumerate(class_names):
        train_count = np.sum(y_train == i)
        val_count = np.sum(y_val == i)
        test_count = np.sum(y_test == i)
        print(f"  {class_name}: Train={train_count}, Val={val_count}, Test={test_count}")
    
    # Preprocess data - Extract MFCC features
    print("\n" + "="*60)
    print("Extracting MFCC features...")
    print("="*60)
    preprocessor = MFCCPreprocessor(
        sample_rate=16000,
        n_mfcc=args.n_mfcc,
        n_fft=2048,
        hop_length=512
    )
    
    # Process each split
    X_train_mfcc = preprocessor.process_batch(X_train)
    X_val_mfcc = preprocessor.process_batch(X_val, max_time_steps=X_train_mfcc.shape[1])
    X_test_mfcc = preprocessor.process_batch(X_test, max_time_steps=X_train_mfcc.shape[1])
    
    print(f"\nFeature shapes:")
    print(f"  Train: {X_train_mfcc.shape}")
    print(f"  Val: {X_val_mfcc.shape}")
    print(f"  Test: {X_test_mfcc.shape}")
    
    # Build model
    print("\n" + "="*60)
    print("Building model...")
    print("="*60)
    
    input_shape = X_train_mfcc.shape[1:]  # (time_steps, n_mfcc, 1)
    model = CNNMFCCBaseline(
        num_classes=len(class_names),
        learning_rate=args.learning_rate
    )
    model.build_model(input_shape)
    
    # Setup checkpoint path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(args.results_dir, f'best_model_C_{timestamp}.h5')
    log_path = os.path.join(args.results_dir, f'training_log_C_{timestamp}.csv')
    
    # Dry run check
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - Testing setup")
        print("="*60)
        
        # Test prediction
        test_pred = model.model.predict(X_test_mfcc[:2], verbose=0)
        print(f"✓ Test prediction shape: {test_pred.shape}")
        
        # Save test results
        test_results = {
            'model': 'CNN_MFCC_Baseline_C',
            'student': 'Saneha',
            'hyperparameters': model.get_hyperparameters(),
            'training_args': vars(args),
            'input_shape': str(input_shape),
            'timestamp': timestamp,
            'note': 'DRY RUN - No actual training performed'
        }
        
        test_file = os.path.join(args.results_dir, f'dryrun_results_C_{timestamp}.json')
        with open(test_file, 'w') as f:
            json.dump(test_results, f, indent=4)
        print(f"✓ Test results saved: {test_file}")
        
        print("\n" + "="*60)
        print("✓ All setup completed successfully!")
        print("="*60)
        print("\nTo run actual training, remove --dry_run flag")
        return
    
    # Train model
    print("\n" + "="*60)
    print("Training model...")
    print("="*60)
    
    history = model.train(
        X_train_mfcc, y_train,
        X_val_mfcc, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=checkpoint_path,
        log_path=log_path
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    test_loss, test_accuracy, test_f1, y_pred = model.evaluate(
        X_test_mfcc, y_test, batch_size=args.batch_size
    )
    
    print(f"\nFinal Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test F1-Score (macro): {test_f1:.4f}")
    
    # Save results
    results = {
        'model': 'CNN_MFCC_Baseline_C',
        'student': 'Saneha',
        'hyperparameters': model.get_hyperparameters(),
        'training_args': vars(args),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'test_f1_macro': float(test_f1),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_val_loss': float(min(history.history['val_loss'])),
        'total_epochs': len(history.history['loss']),
        'timestamp': timestamp,
        'note': 'Realistic baseline with simplified architecture (~50-55% accuracy target)'
    }
    
    results_file = os.path.join(args.results_dir, f'results_C_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to {results_file}")
    print("\nTraining completed successfully! ✓")

if __name__ == '__main__':
    main()