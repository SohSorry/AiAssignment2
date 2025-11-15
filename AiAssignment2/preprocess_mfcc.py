"""
Audio preprocessing for CNN-MFCC model
Extracts MFCC features from audio files
Student: Saneha
"""

import numpy as np
import librosa
from tqdm import tqdm

class MFCCPreprocessor:
    def __init__(self, sample_rate=16000, n_mfcc=40, n_fft=2048, hop_length=512):
        """
        Initialize MFCC preprocessor
        Args:
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        print(f"MFCC Preprocessor initialized:")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  N_MFCC: {n_mfcc}")
        print(f"  N_FFT: {n_fft}")
        print(f"  Hop Length: {hop_length}")
    
    def extract_mfcc(self, audio_array):
        """
        Extract MFCC features from audio
        Args:
            audio_array: Raw audio waveform
        Returns:
            MFCC features with shape (n_mfcc, time_steps)
        """
        try:
            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=audio_array,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Normalize MFCC
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
            
            return mfcc
            
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return None
    
    def process_batch(self, audio_batch, max_time_steps=None):
        """
        Process batch of audio arrays
        Args:
            audio_batch: List of audio waveforms
            max_time_steps: Maximum time steps (for padding/truncating)
        Returns:
            Processed batch with shape (batch_size, time_steps, n_mfcc, 1)
        """
        print("Extracting MFCC features...")
        
        mfcc_features = []
        
        for audio in tqdm(audio_batch, desc="Processing audio"):
            mfcc = self.extract_mfcc(audio)
            if mfcc is not None:
                mfcc_features.append(mfcc)
        
        # Determine max time steps if not provided
        if max_time_steps is None:
            max_time_steps = max([feat.shape[1] for feat in mfcc_features])
            print(f"Max time steps detected: {max_time_steps}")
        
        # Pad or truncate to same length
        processed = []
        for mfcc in mfcc_features:
            if mfcc.shape[1] < max_time_steps:
                # Pad
                pad_width = max_time_steps - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Truncate
                mfcc = mfcc[:, :max_time_steps]
            
            processed.append(mfcc)
        
        # Convert to numpy array and add channel dimension
        # Shape: (batch_size, n_mfcc, time_steps) -> (batch_size, time_steps, n_mfcc, 1)
        processed = np.array(processed)
        processed = np.transpose(processed, (0, 2, 1))  # (batch, time, n_mfcc)
        processed = np.expand_dims(processed, axis=-1)   # Add channel dim
        
        print(f"✓ Processed shape: {processed.shape}")
        return processed
    
    def get_feature_shape(self, audio_sample):
        """
        Get shape of MFCC features for a sample
        Args:
            audio_sample: Single audio waveform
        Returns:
            Feature shape tuple
        """
        mfcc = self.extract_mfcc(audio_sample)
        # Return shape as (time_steps, n_mfcc, 1)
        return (mfcc.shape[1], mfcc.shape[0], 1)