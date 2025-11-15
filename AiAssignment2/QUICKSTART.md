# Quick Start Guide - Wav2Vec2 Baseline Pipeline

## Step-by-Step Setup

### 1. Install Dependencies
```bash
pip install tensorflow transformers librosa soundfile numpy pandas scikit-learn matplotlib seaborn tqdm
```

### 2. Verify Installation
```bash
python test_setup.py
```

This will check:
- ✓ All packages installed
- ✓ GPU availability
- ✓ Wav2Vec2 model loading
- ✓ Dataset accessibility

### 3. Train the Model
```bash
python train.py --epochs 20 --batch_size 8
```

**Quick Training Options:**
- Fast test run (2 epochs): `python train.py --epochs 2 --batch_size 4`
- Full training (20 epochs): `python train.py --epochs 20 --batch_size 8`
- High memory GPU: `python train.py --epochs 30 --batch_size 16`

### 4. Evaluate the Model
```bash
python evaluate.py --model_weights results/best_model_TIMESTAMP.h5 --history_log results/training_log_TIMESTAMP.csv
```

Replace `TIMESTAMP` with the actual timestamp from your training run (found in `results/` folder).

## Expected Output

### Training Output
```
==========================================
Setting up GPU...
==========================================
✓ GPU Available: 1 GPU(s)
  Device: /physical_device:GPU:0

==========================================
Loading dataset...
==========================================
Loading 267 files from phonationA
Loading 268 files from phonationE
...
Total samples loaded: 2127

Data split:
Train: 1489 samples
Validation: 212 samples
Test: 426 samples

==========================================
Building model...
==========================================
Model: "Wav2Vec2_Baseline_A"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_values (InputLayer)   [(None, 80000)]           0         
 
 tf_wav2vec2_model          [(None, 249, 768)]        95M       
 
 global_average_pooling1d   [(None, 768)]              0         
 
 dense1 (Dense)             [(None, 256)]              196,864   
 
 dropout (Dropout)          [(None, 256)]              0         
 
 dense2 (Dense)             [(None, 128)]              32,896    
 
 dropout_1 (Dropout)        [(None, 128)]              0         
 
 output (Dense)             [(None, 8)]                1,032     
=================================================================
Total params: 95,230,792
Trainable params: 230,792
Non-trainable params: 95,000,000
_________________________________________________________________

==========================================
Training model...
==========================================
Epoch 1/20
186/186 [======] - ETA: 0s - loss: 1.8234 - accuracy: 0.3567
Epoch 1: val_accuracy improved from -inf to 0.45283, saving model to results/best_model_...
186/186 [======] - 245s 1s/step - loss: 1.8234 - accuracy: 0.3567 - val_loss: 1.4567 - val_accuracy: 0.4528

...

Test Loss: 0.4523
Test Accuracy: 0.8357

✓ Training completed successfully!
```

### Files Generated
```
results/
├── best_model_20251111_143022.h5
├── training_log_20251111_143022.csv
├── results_20251111_143022.json
└── evaluation_results.json

plots/
├── confusion_matrix.pdf
├── training_history.pdf
└── per_class_metrics.pdf
```

## Troubleshooting

### Problem: No GPU detected
**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]
```

### Problem: Out of memory
**Solution:**
```bash
# Reduce batch size
python train.py --batch_size 4

# Reduce audio duration
python train.py --max_duration 3.0 --batch_size 4
```

### Problem: Module not found
**Solution:**
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

### Problem: Slow training (CPU)
**Solution:**
- Reduce batch size: `--batch_size 4`
- Reduce epochs: `--epochs 10`
- Use shorter audio: `--max_duration 3.0`

## Performance Tips

### For Fast Testing (2-3 minutes)
```bash
python train.py --epochs 2 --batch_size 4 --max_duration 3.0
```

### For Best Accuracy (15-20 minutes)
```bash
python train.py --epochs 30 --batch_size 16 --learning_rate 5e-5
```

### For Limited Memory
```bash
python train.py --epochs 20 --batch_size 4 --max_duration 3.0
```

## Understanding Results

### Training Log (CSV)
- Track accuracy and loss per epoch
- Identify overfitting (val_loss increases)
- Best epoch saved automatically

### Confusion Matrix (PDF)
- Shows which classes are confused
- Diagonal = correct predictions
- Off-diagonal = misclassifications

### Per-Class Metrics (PDF)
- Precision: How many predicted are correct
- Recall: How many actual are found
- F1-Score: Harmonic mean of precision/recall

## Next Steps

1. **Improve Baseline:**
   - Unfreeze Wav2Vec2 layers for fine-tuning
   - Add data augmentation (time stretch, pitch shift)
   - Try different pooling strategies (attention pooling)
   - Increase model capacity (more dense layers)

2. **Implement Other Baselines:**
   - Baseline B: CNN + MFCC features
   - Baseline C: LSTM + Mel-spectrograms
   - Compare all three in report

3. **Generate Report:**
   - Include pipeline diagram
   - Add results table comparing baselines
   - Document hyperparameters
   - Note training time and GPU usage

## Contact & Support
- Check README.md for detailed documentation
- Review code comments for implementation details
- Test with `test_setup.py` before training
