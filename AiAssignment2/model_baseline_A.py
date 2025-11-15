"""
Baseline A: Wav2Vec2 with fine-tuning
Uses pretrained Wav2Vec2-base with custom classification head
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Wav2Vec2Model
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import csv

class Wav2Vec2Baseline(nn.Module):
    def __init__(self, num_classes=8, model_name="facebook/wav2vec2-base", learning_rate=1e-4):
        """
        Initialize Wav2Vec2 baseline model
        Args:
            num_classes: Number of output classes
            model_name: Pretrained model name
            learning_rate: Learning rate for optimizer
        """
        super(Wav2Vec2Baseline, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        
    def build_model(self, input_shape):
        """
        Build model architecture
        Args:
            input_shape: Shape of input features
        """
        print(f"Building Wav2Vec2 baseline model...")
        print(f"Input shape: {input_shape}")
        
        # Load pretrained Wav2Vec2
        print("Loading Wav2Vec2 model...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(self.model_name)
        
        # Freeze base model for initial training
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # Classification head
        hidden_size = self.wav2vec2.config.hidden_size  # 768 for base
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )
        
        print("\nModel Summary:")
        self._print_clean_summary()
        
        return self
    
    def forward(self, input_values):
        """Forward pass"""
        # Extract features from Wav2Vec2
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        
        # Global average pooling
        pooled = torch.mean(hidden_states, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def _print_clean_summary(self):
        """Print a clean, formatted model summary"""
        print("\n" + "="*80)
        print('Model: "Wav2Vec2_Baseline_A"')
        print("="*80)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Wav2Vec2 Base: {sum(p.numel() for p in self.wav2vec2.parameters()):,} parameters")
        print(f"Classifier Head: {sum(p.numel() for p in self.classifier.parameters()):,} parameters")
        print("="*80)
        print(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"Non-trainable params: {total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.2f}M)")
        print("="*80)
    
    def fit_model(self, train_loader, val_loader, epochs=20, device='cuda', checkpoint_path=None, log_path=None):
        """
        Train the model
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_path: Path to save best model
            log_path: Path to save training log
        Returns:
            Training history
        """
        print("\nStarting training...")
        
        history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [], 'val_f1': []
        }
        
        best_val_acc = 0.0
        
        # Open CSV log file
        if log_path:
            log_file = open(log_path, 'w', newline='')
            log_writer = csv.writer(log_file)
            log_writer.writerow(['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'val_f1', 'lr'])
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            # Training phase
            self.train()  # Set model to training mode
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc='Training')
            for batch_idx, (inputs, labels) in enumerate(train_pbar):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'loss': f'{train_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            val_loss, val_acc, val_f1 = self._validate(val_loader, device)
            
            # Update learning rate
            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr != prev_lr:
                print(f"  Learning rate reduced: {prev_lr:.2e} → {current_lr:.2e}")
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # Log to CSV
            if log_path:
                log_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, val_f1, current_lr])
                log_file.flush()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Val F1-Score: {val_f1:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if checkpoint_path:
                    self.save_model(checkpoint_path)
                    print(f"  ✓ New best model saved! (Val Acc: {best_val_acc:.4f})")
            
            # Early stopping check
            if epoch > 10 and val_acc < best_val_acc - 0.1:
                print("\nEarly stopping triggered!")
                break
        
        if log_path:
            log_file.close()
        
        print(f"\nTraining completed! Best Val Accuracy: {best_val_acc:.4f}")
        return history
    
    def _validate(self, val_loader, device):
        """Validate the model"""
        self.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation')
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return val_loss, val_acc, val_f1
    
    def evaluate(self, test_loader, device):
        """
        Evaluate model on test set
        Args:
            test_loader: Test DataLoader
            device: Device to evaluate on
        Returns:
            Test loss and accuracy
        """
        print("\nEvaluating model...")
        self.eval()  # Set model to evaluation mode
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Testing'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_total
        
        return test_loss, test_acc
    
    def predict(self, data_loader, device):
        """
        Make predictions
        Args:
            data_loader: Input DataLoader
            device: Device to predict on
        Returns:
            Predictions and true labels
        """
        self.eval()  # Set model to evaluation mode
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc='Predicting'):
                inputs = inputs.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def save_model(self, path):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path, device='cuda'):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Model loaded from {path}")
    
    def get_hyperparameters(self):
        """Return hyperparameters dictionary"""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'learning_rate': self.learning_rate,
            'architecture': 'Wav2Vec2-base + Dense(256) + Dense(128) + Softmax',
            'pooling': 'Global Average Pooling',
            'dropout': 0.3
        }
