"""
Evaluation script for baseline models
Generates metrics, confusion matrix, and performance plots
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import torch
import json

from data_loader import AudioDataLoader
from preprocess import AudioPreprocessor
from model_baseline_A import Wav2Vec2Baseline

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Wav2Vec2 Baseline')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()

def plot_training_history(history_file, save_path):
    """Plot training history from CSV log"""
    import pandas as pd
    
    if not os.path.exists(history_file):
        print(f"History file not found: {history_file}")
        return
    
    df = pd.read_csv(history_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(df['epoch'], df['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(df['epoch'], df['val_accuracy'], label='Val Accuracy', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(df['epoch'], df['loss'], label='Train Loss', marker='o')
    axes[1].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")
    plt.close()

def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    """Plot per-class precision, recall, F1-score"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [report[class_name][metric] for class_name in class_names] 
            for metric in metrics}
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, data[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class metrics plot saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline model')
    parser.add_argument('--data_path', type=str,
                       default='SAND_Challenge_task1_dataset/task1/training',
                       help='Path to training data')
    parser.add_argument('--model_weights', type=str, required=True,
                       help='Path to saved model weights')
    parser.add_argument('--history_log', type=str,
                       help='Path to training history CSV log')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--max_duration', type=float, default=5.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--plots_dir', type=str, default='plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Setup GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Load data
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    loader = AudioDataLoader(args.data_path, sample_rate=16000)
    audio_data, labels, _ = loader.load_dataset(max_duration=args.max_duration)
    
    # Split data (using same random seed as training)
    _, _, X_test, _, _, y_test = loader.split_data(audio_data, labels)
    class_names = loader.get_class_names()
    
    # Preprocess
    print("\n" + "="*60)
    print("Preprocessing audio...")
    print("="*60)
    preprocessor = AudioPreprocessor()
    test_dataset, test_loader = preprocessor.create_dataloader(
        X_test, y_test, batch_size=args.batch_size, shuffle=False
    )
    
    # Build and load model
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    sample_features = preprocessor.process_audio(X_test[0])
    
    model = Wav2Vec2Baseline(num_classes=len(class_names))
    model.build_model(sample_features.shape)
    model.load_model(args.model_weights, device=device)
    model.to(device)
    model.eval()
    
    # Make predictions
    print("\n" + "="*60)
    print("Making predictions...")
    print("="*60)
    y_pred, y_test = model.predict(test_loader, device)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("Calculating metrics...")
    print("="*60)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Print clean formatted results
    print("\n" + "="*80)
    print("--- Evaluating Wav2Vec2 Model ---")
    print("="*80)
    
    # Calculate F1 per class for display
    from sklearn.metrics import f1_score as f1_per_class
    f1_scores = f1_per_class(y_test, y_pred, average=None)
    avg_f1_val = np.mean(f1_scores)
    
    print(f"\nAveraged F1-Score on Test Set: {avg_f1_val:.4f}")
    
    print("\nClassification Report:")
    print(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 63)
    
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    for class_name in class_names:
        if class_name in report_dict:
            metrics = report_dict[class_name]
            print(f"{class_name:<15} {metrics['precision']:>12.2f} {metrics['recall']:>12.2f} "
                  f"{metrics['f1-score']:>12.2f} {int(metrics['support']):>12}")
    
    print("-" * 63)
    print(f"{'accuracy':<15} {'':<12} {'':<12} {accuracy:>12.2f} {len(y_test):>12}")
    print(f"{'macro avg':<15} {report_dict['macro avg']['precision']:>12.2f} "
          f"{report_dict['macro avg']['recall']:>12.2f} {report_dict['macro avg']['f1-score']:>12.2f} "
          f"{len(y_test):>12}")
    print(f"{'weighted avg':<15} {report_dict['weighted avg']['precision']:>12.2f} "
          f"{report_dict['weighted avg']['recall']:>12.2f} {report_dict['weighted avg']['f1-score']:>12.2f} "
          f"{len(y_test):>12}")
    print("="*80)
    
    # Generate plots
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    # Confusion matrix
    cm_path = os.path.join(args.plots_dir, 'confusion_matrix.pdf')
    plot_confusion_matrix(y_test, y_pred, class_names, cm_path)
    
    # Training history
    if args.history_log and os.path.exists(args.history_log):
        history_path = os.path.join(args.plots_dir, 'training_history.pdf')
        plot_training_history(args.history_log, history_path)
    
    # Per-class metrics
    metrics_path = os.path.join(args.plots_dir, 'per_class_metrics.pdf')
    plot_per_class_metrics(y_test, y_pred, class_names, metrics_path)
    
    # Save evaluation results
    eval_results = {
        'model': 'Wav2Vec2_Baseline_A',
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'num_test_samples': len(y_test),
        'classification_report': classification_report(y_test, y_pred, 
                                                       target_names=class_names,
                                                       output_dict=True)
    }
    
    eval_file = os.path.join(args.results_dir, 'evaluation_results.json')
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print(f"\n✓ Evaluation results saved to {eval_file}")
    print("\nEvaluation completed successfully! ✓")

if __name__ == '__main__':
    main()
