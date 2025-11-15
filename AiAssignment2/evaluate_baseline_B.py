# evaluate_baseline_B.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from data_loader_B import SANDDatasetFast
from tensorflow.keras.models import load_model
import pickle

SAVE_DIR = "report_outputs_B"
DATASET_PATH = "SAND_Challenge_task1_dataset/task1/training"

# ---------------- Load Dataset ----------------
dataset = SANDDatasetFast(DATASET_PATH, augment=False)
X_val = tf.keras.preprocessing.sequence.pad_sequences(
    [dataset[i][0] for i in range(len(dataset))],
    padding='post', dtype='float32'
)
y_val = tf.keras.utils.to_categorical(
    [dataset[i][1] for i in range(len(dataset))],
    num_classes=len(dataset.encoder.classes_)
)

# ---------------- Load Model ----------------
model = load_model(os.path.join(SAVE_DIR, 'best_model.keras'))

# ---------------- Predictions & Metrics ----------------
y_true = np.argmax(y_val, axis=1)
y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=dataset.encoder.classes_,
            yticklabels=dataset.encoder.classes_, cmap='Blues')
plt.ylabel('True'); plt.xlabel('Predicted'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png')); plt.show()

# Classification Report
report = classification_report(y_true, y_pred, target_names=dataset.encoder.classes_,
                               output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(SAVE_DIR, 'classification_report.csv'))
print(report_df)

# Per-Class Metrics Plot
class_names = dataset.encoder.classes_
metrics = ['precision', 'recall', 'f1-score']
data = {metric: [report[cls][metric] for cls in class_names] for metric in metrics}

x = np.arange(len(class_names))
width = 0.25
fig, ax = plt.subplots(figsize=(15,7))
colors = ['#3498db', '#2ecc71', '#e74c3c']

for i, metric in enumerate(metrics):
    bars = ax.bar(x + i*width, data[metric], width, label=metric.capitalize(),
                  color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.7)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Class', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Per-Class Performance Metrics - LSTM Baseline (SAND)', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim([0, 1.1])
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'per_class_metrics.png')); plt.show()

# ---------------- Load Training History ----------------
with open(os.path.join(SAVE_DIR, 'history.pkl'), 'rb') as f:
    history = pickle.load(f)

# Training Accuracy & Loss
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.plot(history['accuracy'], label='Train Acc', linewidth=2, marker='o')
plt.plot(history['val_accuracy'], label='Val Acc', linewidth=2, marker='x')
plt.xlabel('Epoch', fontsize=14); plt.ylabel('Accuracy', fontsize=14)
plt.title('Accuracy over Epochs', fontsize=16); plt.grid(True, linestyle='--', alpha=0.6); plt.legend(fontsize=12)

plt.subplot(1,2,2)
plt.plot(history['loss'], label='Train Loss', linewidth=2, marker='o')
plt.plot(history['val_loss'], label='Val Loss', linewidth=2, marker='x')
plt.xlabel('Epoch', fontsize=14); plt.ylabel('Loss', fontsize=14)
plt.title('Loss over Epochs', fontsize=16); plt.grid(True, linestyle='--', alpha=0.6); plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_history.png')); plt.show()

# ---------------- Learning Rate Schedule ----------------
if 'lr' in history:
    lr_history = history['lr']
else:
    # Reconstruct LR if LearningRateScheduler used
    lr_history = []
    lr = 0.001
    for epoch in range(len(history['loss'])):
        if epoch > 0 and epoch % 10 == 0:
            lr *= 0.5
        lr_history.append(lr)

plt.figure(figsize=(12,5))
plt.plot(lr_history, label='Learning Rate', color='#8e44ad', linewidth=2)
plt.xlabel('Epoch', fontsize=13, fontweight='bold')
plt.ylabel('LR', fontsize=13, fontweight='bold')
plt.title('Learning Rate Schedule', fontsize=15, fontweight='bold', pad=15)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'learning_rate_schedule.png')); plt.show()

# ---------------- Overfitting Analysis ----------------
train_loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(train_loss)+1)
plt.figure(figsize=(12,5))
plt.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, alpha=0.7)
plt.plot(epochs, val_loss, 'r-x', label='Val Loss', linewidth=2, alpha=0.7)
plt.fill_between(epochs, train_loss, val_loss, color='gray', alpha=0.2)
plt.xlabel('Epoch', fontsize=13, fontweight='bold'); plt.ylabel('Loss', fontsize=13, fontweight='bold')
plt.title('Overfitting Analysis (Train vs Validation Loss)', fontsize=15, fontweight='bold', pad=15)
plt.legend(fontsize=12); plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'overfitting_analysis.png')); plt.show()
