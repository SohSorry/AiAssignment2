# Baseline Pipeline Implementation - SAND Challenge

## Project Overview
This project implements a baseline speech classification pipeline using **Wav2Vec2** pretrained model with TensorFlow for the SAND Challenge Task 1 dataset. The pipeline classifies audio recordings into 8 categories: phonation vowels (A, E, I, O, U) and rhythm syllables (KA, PA, TA).

## Architecture
**Baseline A: Wav2Vec2 Fine-tuning**
- Pretrained Model: `facebook/wav2vec2-base`
- Feature Extraction: Wav2Vec2 transformer encoder (768-dim)
- Pooling: Global Average Pooling
- Classification Head: Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.3) → Softmax(8)
- Optimizer: Adam (lr=1e-4)
- GPU-accelerated training with TensorFlow

## Project Structure
```
├── data_loader.py          # Load and split audio dataset
├── preprocess.py           # Wav2Vec2 feature extraction
├── model_baseline_A.py     # Wav2Vec2 baseline model
├── train.py               # Training pipeline
├── evaluate.py            # Evaluation and metrics
├── requirements.txt       # Dependencies
├── README.md             # This file
├── results/              # Training outputs and metrics
└── plots/                # Performance visualizations (PDF)
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify GPU Support
```bash
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

## Usage

### Training Baseline A (Wav2Vec2)
```bash
python train.py --epochs 20 --batch_size 8
```

### Training Baseline B (CNN + MFCC)
```bash
python train_baseline_B.py --epochs 50 --batch_size 32
```

### Training Baseline C (LSTM + Mel-Spectrogram)
```bash
python train_baseline_C.py --epochs 50 --batch_size 32
```

**Common Training Arguments:**
- `--data_path`: Path to training data (default: `SAND_Challenge_task1_dataset/task1/training`)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--results_dir`: Output directory (default: `results`)

### Evaluation
```bash
python evaluate.py --model_weights results/best_model_TIMESTAMP.h5 --history_log results/training_log_TIMESTAMP.csv
```

**Evaluation Arguments:**
- `--model_weights`: Path to saved model weights (required)
- `--history_log`: Path to training CSV log
- `--data_path`: Path to dataset
- `--batch_size`: Batch size (default: 8)
- `--results_dir`: Results directory (default: `results`)
- `--plots_dir`: Plots directory (default: `plots`)

## Pipeline Flow
```
Audio Data (.wav)
    ↓
Data Loader (data_loader.py)
    → Load audio files
    → Resample to 16kHz
    → Pad/truncate to fixed length
    → Split: Train (70%) / Val (10%) / Test (20%)
    ↓
Preprocessor (preprocess.py)
    → Wav2Vec2Processor normalization
    → Feature extraction preparation
    → TensorFlow dataset creation
    ↓
Model (model_baseline_A.py)
    → Wav2Vec2-base encoder (frozen)
    → Global Average Pooling
    → Dense classification layers
    → Softmax output (8 classes)
    ↓
Training (train.py)
    → GPU acceleration
    → Early stopping (patience=5)
    → Learning rate reduction
    → Model checkpointing
    ↓
Evaluation (evaluate.py)
    → Test accuracy & F1-scores
    → Confusion matrix (PDF)
    → Per-class metrics (PDF)
    → Training curves (PDF)
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | facebook/wav2vec2-base |
| Sample Rate | 16,000 Hz |
| Max Duration | 5.0 seconds |
| Batch Size | 8 |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Dense Layer 1 | 256 units, ReLU |
| Dense Layer 2 | 128 units, ReLU |
| Dropout | 0.3 |
| Epochs | 20 (with early stopping) |

## Output Files

### Results Directory
- `best_model_TIMESTAMP.h5` - Best model weights
- `training_log_TIMESTAMP.csv` - Training history
- `results_TIMESTAMP.json` - Training summary
- `evaluation_results.json` - Test metrics

### Plots Directory (PDF)
- `confusion_matrix.pdf` - Confusion matrix heatmap
- `training_history.pdf` - Accuracy & loss curves
- `per_class_metrics.pdf` - Precision, recall, F1 per class

## Expected Performance
- **Target Accuracy**: ~75-85% on test set
- **Training Time**: ~10-15 minutes per epoch (GPU)
- **Memory**: ~6-8 GB GPU RAM

## GPU Optimization
The pipeline automatically:
- Detects and utilizes available GPU
- Enables memory growth to prevent OOM errors
- Uses TensorFlow dataset prefetching
- Applies mixed precision if available

## Implementation Notes
- Wav2Vec2 base model is **frozen** during initial training for faster convergence
- Data augmentation not implemented in baseline (can be added)
- Uses stratified splitting to maintain class balance
- Supports early stopping to prevent overfitting

## Group Task Division
- **Student 1**: Wav2Vec2 baseline (this implementation)
- **Student 2**: [Add baseline B description]
- **Student 3**: [Add baseline C description]

## References
- Wav2Vec2: https://huggingface.co/facebook/wav2vec2-base
- TensorFlow: https://www.tensorflow.org/
- Transformers: https://huggingface.co/docs/transformers/

## Troubleshooting

**GPU not detected:**
```bash
pip install tensorflow-gpu==2.13.0
```

**Out of memory:**
- Reduce batch size: `--batch_size 4`
- Reduce max duration: `--max_duration 3.0`

**Import errors:**
```bash
pip install --upgrade -r requirements.txt
```
