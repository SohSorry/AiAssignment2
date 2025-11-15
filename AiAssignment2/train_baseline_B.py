# train_baseline_B.py
import os
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from data_loader_B import SANDDatasetFast, build_train_val_arrays
from model_baseline_B import build_baseline_B

DATASET_PATH = "SAND_Challenge_task1_dataset/task1/training"
SAVE_DIR = "report_outputs_B"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

dataset = SANDDatasetFast(DATASET_PATH, augment=True)
X_train, y_train, X_val, y_val = build_train_val_arrays(dataset)

model = build_baseline_B(input_shape=X_train.shape[1:], num_classes=y_train.shape[1], learning_rate=LEARNING_RATE)
model.summary()

# Learning Rate Scheduler
lr_history = []
def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % 10 == 0:
        lr *= 0.5
    lr_history.append(lr)
    return lr

scheduler = LearningRateScheduler(lr_scheduler)
checkpoint = ModelCheckpoint(os.path.join(SAVE_DIR, 'best_model.keras'), save_best_only=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, scheduler]
)

# Save training history
import pickle
with open(os.path.join(SAVE_DIR, 'history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)
