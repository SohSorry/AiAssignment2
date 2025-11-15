"""
Test script to verify Wav2Vec2 model can be loaded on CPU
"""

import tensorflow as tf
from transformers import TFWav2Vec2Model
import numpy as np

print("="*70)
print("Testing Wav2Vec2 Model Loading on CPU")
print("="*70)

# Check devices
gpus = tf.config.list_physical_devices('GPU')