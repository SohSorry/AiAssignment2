# model_baseline_B.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import regularizers

def build_baseline_B(input_shape, num_classes, learning_rate=0.001):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),
        LSTM(32, kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
