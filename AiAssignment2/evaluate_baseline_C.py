"""
Evaluation script for Baseline C: CNN-MFCC model
Generates metrics, confusion matrix, and performance plots
Student: Saneha
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import json
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from data_loader import AudioDataLoader
from preprocess_mfcc import MFCCPreprocessor
from model_baseline_C import CNNMFCCBaseline

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - CNN-MFCC Baseline (Saneha)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()

def plot_training_history(history_file, save_path):
    """Plot training history from CSV log"""
    if not os.path.exists(history_file):
        print(f"History file not found: {history_file}")
        return
    
    df = pd.read_csv(history_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy plot
    axes[0, 0].plot(df['epoch'], df['accuracy'], label='Train Accuracy', marker='o', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_accuracy'], label='Val Accuracy', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[0, 1].plot(df['epoch'], df['loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate plot
    if 'lr' in df.columns:
        axes[1, 0].plot(df['epoch'], df['lr'], marker='o', linewidth=2, color='red')
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting analysis
    axes[1, 1].plot(df['epoch'], df['accuracy'], label='Train Acc', alpha=0.7, linewidth=2)
    axes[1, 1].plot(df['epoch'], df['val_accuracy'], label='Val Acc', alpha=0.7, linewidth=2)
    axes[1, 1].fill_between(df['epoch'], df['accuracy'], df['val_accuracy'],
                            alpha=0.2, label='Generalization Gap')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Accuracy', fontsize=11)
    axes[1, 1].set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('CNN-MFCC Training History (Baseline C - Saneha)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")
    plt.close()

def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    """Plot per-class precision, recall, F1-score"""
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)
    
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [report[class_name][metric] for class_name in class_names] 
            for metric in metrics}
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, data[metric], width, label=metric.capitalize(), 
               color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics - CNN-MFCC (Saneha)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='50% baseline')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class metrics plot saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate CNN-MFCC baseline model')
    parser.add_argument('--data_path', type=str,
                       default='SAND_Challenge_task1_dataset/task1/training',
                       help='Path to training data')
    parser.add_argument('--model_weights', type=str, required=True,
                       help='Path to saved model weights (.h5 file)')
    parser.add_argument('--history_log', type=str,
                       help='Path to training history CSV log')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--max_duration', type=float, default=5.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--n_mfcc', type=int, default=40,
                       help='Number of MFCC coefficients')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--plots_dir', type=str, default='plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    loader = AudioDataLoader(args.data_path, sample_rate=16000)
    audio_data, labels, _ = loader.load_dataset(max_duration=args.max_duration)
    
    # Split data (using same split as training)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    _, test_idx = next(sss1.split(audio_data, labels))
    
    X_test = audio_data[test_idx]
    y_test = labels[test_idx]
    class_names = loader.get_class_names()
    
    # Preprocess - Extract MFCC
    print("\n" + "="*60)
    print("Extracting MFCC features...")
    print("="*60)
    preprocessor = MFCCPreprocessor(
        sample_rate=16000,
        n_mfcc=args.n_mfcc,
        n_fft=2048,
        hop_length=512
    )
    X_test_mfcc = preprocessor.process_batch(X_test)
    
    # Load model
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    model = CNNMFCCBaseline(num_classes=len(class_names))
    model.load_model(args.model_weights)
    
    print(f"Model loaded successfully!")
    print(f"Test set shape: {X_test_mfcc.shape}")
    
    # Make predictions
    print("\n" + "="*60)
    print("Making predictions...")
    print("="*60)
    y_pred = model.predict(X_test_mfcc, batch_size=args.batch_size)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("Calculating metrics...")
    print("="*60)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Print formatted results
    print("\n" + "="*80)
    print("--- Evaluating CNN-MFCC Model (Saneha) ---")
    print("="*80)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score (macro): {f1_macro:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("="*80)
    
    # Generate plots
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    # Confusion matrix
    cm_path = os.path.join(args.plots_dir, 'confusion_matrix_C.pdf')
    plot_confusion_matrix(y_test, y_pred, class_names, cm_path)
    
    # Training history
    if args.history_log and os.path.exists(args.history_log):
        history_path = os.path.join(args.plots_dir, 'training_history_C.pdf')
        plot_training_history(args.history_log, history_path)
    
    # Per-class metrics
    metrics_path = os.path.join(args.plots_dir, 'per_class_metrics_C.pdf')
    plot_per_class_metrics(y_test, y_pred, class_names, metrics_path)
    
    # Save evaluation results
    eval_results = {
        'model': 'CNN_MFCC_Baseline_C',
        'student': 'Saneha',
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'num_test_samples': len(y_test),
        'classification_report': classification_report(y_test, y_pred, 
                                                       target_names=class_names,
                                                       output_dict=True,
                                                       zero_division=0)
    }
    
    eval_file = os.path.join(args.results_dir, 'evaluation_results_C.json')
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print(f"\n✓ Evaluation results saved to {eval_file}")
    print("\nEvaluation completed successfully! ✓")

if __name__ == '__main__':
    main()