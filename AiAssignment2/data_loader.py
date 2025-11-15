"""
Data loader for SAND Challenge Task 1 Dataset
Loads audio files and prepares them for model training
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf

class AudioDataLoader:
    def __init__(self, data_path, sample_rate=16000):
        """
        Initialize data loader
        Args:
            data_path: Path to dataset root
            sample_rate: Target sample rate for audio files
        """
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.categories = ['phonationA', 'phonationE', 'phonationI', 'phonationO', 
                          'phonationU', 'rhythmKA', 'rhythmPA', 'rhythmTA']
        
    def load_audio_file(self, file_path):
        """Load and resample audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_dataset(self, max_duration=5.0):
        """
        Load all audio files and create labels
        Args:
            max_duration: Maximum audio duration in seconds
        Returns:
            audio_data: List of audio arrays
            labels: List of category labels
            file_paths: List of file paths
        """
        audio_data = []
        labels = []
        file_paths = []
        max_length = int(max_duration * self.sample_rate)
        
        print(f"Loading dataset from: {self.data_path}")
        
        for idx, category in enumerate(self.categories):
            category_path = os.path.join(self.data_path, category)
            if not os.path.exists(category_path):
                print(f"Warning: {category_path} not found")
                continue
                
            files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            print(f"Loading {len(files)} files from {category}")
            
            for file in files:
                file_path = os.path.join(category_path, file)
                audio = self.load_audio_file(file_path)
                
                if audio is not None:
                    # Pad or truncate to max_length
                    if len(audio) < max_length:
                        audio = np.pad(audio, (0, max_length - len(audio)))
                    else:
                        audio = audio[:max_length]
                    
                    audio_data.append(audio)
                    labels.append(idx)
                    file_paths.append(file_path)
        
        print(f"\nTotal samples loaded: {len(audio_data)}")
        print(f"Classes: {len(self.categories)}")
        
        return np.array(audio_data), np.array(labels), file_paths
    
    def split_data(self, audio_data, labels, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets
        Args:
            audio_data: Audio samples
            labels: Labels
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
        Returns:
            Train, validation, and test splits
        """
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            audio_data, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_names(self):
        """Return category names"""
        return self.categories
