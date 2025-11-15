CNN-MFCC Baseline (Baseline C) - Saneha
📋 Overview
This is Baseline C for the SAND Challenge Task 1 - Audio Classification using a simplified CNN with MFCC features.
Student: Saneha
Model: Convolutional Neural Network (CNN) with MFCC spectrograms
Architecture: 3 Conv2D blocks + 1 Dense layer
Target Performance: ~50-55% accuracy (realistic baseline)

🎯 Model Architecture
Feature Extraction

MFCC (Mel-Frequency Cepstral Coefficients)

40 MFCC coefficients
2048 FFT window size
512 hop length
Normalized features



CNN Architecture
Input (time_steps, 40, 1)
    ↓
Conv2D(16) + BatchNorm + MaxPool + Dropout(0.3)
    ↓
Conv2D(32) + BatchNorm + MaxPool + Dropout(0.3)
    ↓
Conv2D(64) + BatchNorm + GlobalAvgPool + Dropout(0.5)
    ↓
Dense(128) + Dropout(0.5)
    ↓
Dense(8) [Softmax]
Model Parameters

Total Parameters: 33,096
Trainable Parameters: 32,872
Conv Filters: [16, 32, 64]
Dense Units: [128]
Regularization: L2 (0.01), Dropout, BatchNorm


📁 Files Structure
model_baseline_C.py       # CNN-MFCC model implementation
preprocess_mfcc.py        # MFCC feature extraction
train_baseline_C.py       # Training script
evaluate_baseline_C.py    # Evaluation script
data_loader.py            # Audio data loader (shared)
README_BASELINE_C.md      # This file

🚀 Usage
1. Install Dependencies
bash
pip install tensorflow==2.13.0 numpy librosa scikit-learn matplotlib seaborn pandas tqdm
2. Train Model
bash
 python AiAssignment2\train_baseline_C.py --data_path SAND_Challenge_task1_dataset/task1/training \ --epochs 38 \--batch_size 32 \ --learning_rate 0.0005 \ --n_mfcc 40 \    --results_dir results    
3. Evaluate Model
bash
python evaluate_baseline_C.py \ --data_path SAND_Challenge_task1_dataset/task1/training \ --model_weights results/best_model_C_YYYYMMDD_HHMMSS.h5 \ --history_log results/training_log_C_YYYYMMDD_HHMMSS.csv \ --results_dir results \--plots_dir plots

    run python AiAssignment2-main\AiAssignment2-main\train_baseline_C.py  --data_path \SAND_Challenge_task1_dataset\task1\training"