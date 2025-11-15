"""
Baseline C: CNN with MFCC features
Simplified CNN architecture with MFCC spectrograms for realistic baseline performance
Student: Saneha
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
from sklearn.metrics import f1_score
import os

class CNNMFCCBaseline:
    def __init__(self, num_classes=8, learning_rate=0.0005):
        """
        Initialize CNN-MFCC baseline model
        Args:
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        """
        Build simplified CNN architecture for MFCC features
        Args:
            input_shape: Shape of MFCC input (time_steps, n_mfcc, 1)
        """
        print(f"\nBuilding CNN-MFCC model...")
        print(f"Input shape: {input_shape}")
        
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # Convolutional Block 1
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Convolutional Block 2
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Convolutional Block 3
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Dense layer
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='CNN_MFCC_Baseline_C')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print summary
        print("\n" + "="*80)
        print('Model: "CNN_MFCC_Baseline_C"')
        print("="*80)
        self.model.summary()
        
        total_params = self.model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        
        print("="*80)
        print(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("="*80)
        
        return self
    
    def train(self, X_train, y_train, X_val, y_val, epochs=60, batch_size=32,
              checkpoint_path=None, log_path=None):
        """
        Train the model
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            checkpoint_path: Path to save best model
            log_path: Path to save training log
        Returns:
            Training history
        """
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)
        print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint
        if checkpoint_path:
            checkpoint_cb = keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint_cb)
        
        # Learning rate reduction
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # CSV Logger
        if log_path:
            csv_logger = keras.callbacks.CSVLogger(log_path, separator=',', append=False)
            callbacks.append(csv_logger)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Final metrics
        best_val_acc = max(self.history.history['val_accuracy'])
        best_epoch = self.history.history['val_accuracy'].index(best_val_acc) + 1
        
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)
        print(f"Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        # Calculate F1 on validation set
        y_val_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        print(f"Final validation F1-score: {val_f1:.4f}")
        
        return self.history
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Evaluate model on test set
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size
        Returns:
            Test loss, accuracy, F1-score, and predictions
        """
        print("\n" + "="*60)
        print("Evaluating on test set...")
        print("="*60)
        
        test_loss, test_acc = self.model.evaluate(X_test, y_test, 
                                                   batch_size=batch_size, 
                                                   verbose=1)
        y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        test_f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        
        return test_loss, test_acc, test_f1, y_pred
    
    def predict(self, X, batch_size=32):
        """
        Make predictions
        Args:
            X: Input features
            batch_size: Batch size
        Returns:
            Predicted labels
        """
        predictions = self.model.predict(X, batch_size=batch_size, verbose=1)
        return np.argmax(predictions, axis=1)
    
    def save_model(self, path):
        """Save model weights"""
        self.model.save(path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path):
        """Load model weights"""
        self.model = keras.models.load_model(path)
        print(f"✓ Model loaded from {path}")
    
    def get_hyperparameters(self):
        """Return hyperparameters dictionary"""
        return {
            'model_type': 'CNN-MFCC',
            'num_classes': self.num_classes,
            'learning_rate': self.learning_rate,
            'architecture': '3 Conv2D blocks + 1 Dense layer (Simplified)',
            'conv_filters': [16, 32, 64],
            'dense_units': [128],
            'dropout_rates': [0.3, 0.3, 0.5, 0.5],
            'regularization': 'L2 (0.01)',
            'pooling': 'MaxPooling2D + GlobalAveragePooling2D',
            'batch_normalization': True,
            'note': 'Simplified baseline for realistic performance (~50-55% accuracy)'
        }