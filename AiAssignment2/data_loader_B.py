# data_loader_B.py
import os
import glob
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

SR = 16000
DURATION = 3
N_MELS = 64
HOP_LENGTH = 512
NOISE_FACTOR = 0.03

NUM_CLASSES = None

class SANDDatasetFast:
    def __init__(self, root_dir, sr=SR, duration=DURATION, n_mels=N_MELS,
                 hop_length=HOP_LENGTH, augment=False, noise_factor=NOISE_FACTOR):
        self.root_dir = root_dir
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.augment = augment
        self.noise_factor = noise_factor
        
        self.audio_files = glob.glob(os.path.join(root_dir, '**/*.wav'), recursive=True)
        self.labels = [os.path.basename(os.path.dirname(f)) for f in self.audio_files]
        self.encoder = LabelEncoder()
        self.encoder.fit(self.labels)
        self.labels_encoded = self.encoder.transform(self.labels)
        
        global NUM_CLASSES
        NUM_CLASSES = len(self.encoder.classes_)
        
        print("Precomputing Mel-spectrograms...")
        self.mels = []
        for f in self.audio_files:
            y, sr = librosa.load(f, sr=self.sr)
            max_len = self.sr * self.duration
            if len(y) < max_len:
                y = np.pad(y, (0, max_len - len(y)))
            else:
                y = y[:max_len]
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
            mel_db = librosa.power_to_db(mel, ref=np.max).T
            self.mels.append(mel_db)
        print("Precomputation done.")

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        mel = self.mels[idx].copy()
        label = self.labels_encoded[idx]
        if self.augment:
            mel += self.noise_factor * np.random.randn(*mel.shape)
            shift = np.random.randint(-5,5)
            mel = np.roll(mel, shift, axis=0)
        return mel, label


def build_train_val_arrays(dataset, test_size=0.2, label_smoothing=0.1):
    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=test_size, random_state=42)

    X_train = [dataset[i][0] for i in train_idx]
    y_train = [dataset[i][1] for i in train_idx]
    X_val = [dataset[i][0] for i in val_idx]
    y_val = [dataset[i][1] for i in val_idx]

    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', dtype='float32')
    X_val = tf.keras.preprocessing.sequence.pad_sequences(X_val, padding='post', dtype='float32')

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(dataset.encoder.classes_))
    y_train = y_train * (1 - label_smoothing) + label_smoothing / len(dataset.encoder.classes_)

    y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(dataset.encoder.classes_))
    
    return X_train, y_train, X_val, y_val
