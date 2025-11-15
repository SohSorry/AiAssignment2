"""
Audio preprocessing for Wav2Vec2 model
Extracts features and prepares data for PyTorch
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2Processor

class AudioPreprocessor:
    def __init__(self, model_name="facebook/wav2vec2-base", sample_rate=16000):
        """
        Initialize preprocessor with Wav2Vec2 processor
        Args:
            model_name: Pretrained Wav2Vec2 model name
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        print(f"Loaded Wav2Vec2 processor: {model_name}")
    
    def process_audio(self, audio_array):
        """
        Process single audio array using Wav2Vec2 processor
        Args:
            audio_array: Raw audio waveform
        Returns:
            Processed input values
        """
        # Normalize audio
        audio_array = audio_array.astype(np.float32)
        
        # Process with Wav2Vec2 processor
        inputs = self.processor(
            audio_array, 
            sampling_rate=self.sample_rate,
            return_tensors="np",
            padding=True
        )
        
        return inputs.input_values[0]
    
    def process_batch(self, audio_batch):
        """
        Process batch of audio arrays
        Args:
            audio_batch: Batch of audio waveforms
        Returns:
            Processed batch
        """
        processed = []
        for audio in audio_batch:
            processed.append(self.process_audio(audio))
        return np.array(processed)
    
    def create_dataloader(self, X, y, batch_size=8, shuffle=True):
        """
        Create PyTorch DataLoader with preprocessing
        Args:
            X: Audio data
            y: Labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
        Returns:
            TensorDataset and DataLoader
        """
        # Process all audio
        print("Processing audio features...")
        X_processed = self.process_batch(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_processed)
        y_tensor = torch.LongTensor(y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=torch.cuda.is_available()
        )
        
        return dataset, dataloader
    
    def get_feature_shape(self, audio_sample):
        """Get shape of processed features"""
        processed = self.process_audio(audio_sample)
        return processed.shape
