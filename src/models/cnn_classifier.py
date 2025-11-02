"""CNN Classifier for supervised cardiac arrhythmia classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from sklearn.utils.class_weight import compute_class_weight


class CNNClassifier(nn.Module):
    """1D CNN for multi-class arrhythmia classification.
    
    Classifies ECG signals into:
    0: Normal
    1: Atrial Premature Contraction (APC)
    2: Premature Ventricular Contraction (PVC)
    3: Ventricular Fibrillation (VF)
    4: Fusion
    """
    
    def __init__(self, input_size: int = 360,
                 num_classes: int = 5,
                 channels: list = [32, 64, 128, 256],
                 kernel_size: int = 5,
                 dropout: float = 0.3):
        """Initialize CNN Classifier.
        
        Args:
            input_size: Length of input signal
            num_classes: Number of output classes
            channels: List of channel sizes for conv layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
        """
        super(CNNClassifier, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Convolutional layers
        conv_layers = []
        in_channels = 1
        
        for out_channels in channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate size after convolutions
        self.feature_size = self._calculate_feature_size()
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(channels[-1] * self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def _calculate_feature_size(self) -> int:
        """Calculate size after convolutional layers."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_size)
            x = self.conv_layers(x)
            return x.shape[2]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, 1, signal_length)
            
        Returns:
            Class logits (batch_size, num_classes)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class labels
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Class probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def train_classifier(model: CNNClassifier,
                    train_data: np.ndarray,
                    train_labels: np.ndarray,
                    val_data: Optional[np.ndarray] = None,
                    val_labels: Optional[np.ndarray] = None,
                    epochs: int = 50,
                    batch_size: int = 32,
                    learning_rate: float = 0.001,
                    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                    use_weighted_loss: bool = True,
                    early_stopping_patience: int = 10) -> Dict:
    """Train CNN Classifier.
    
    Args:
        model: CNN Classifier model
        train_data: Training data (n_samples, signal_length)
        train_labels: Training labels (n_samples,)
        val_data: Validation data
        val_labels: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        use_weighted_loss: Use class weights for imbalanced data
        early_stopping_patience: Patience for early stopping
        
    Returns:
        Dictionary of training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Compute class weights for imbalanced data
    if use_weighted_loss:
        class_weights = compute_class_weight('balanced', 
                                            classes=np.unique(train_labels),
                                            y=train_labels)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(train_data))
        
        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_data = train_data[batch_indices]
            batch_labels = train_labels[batch_indices]
            
            x = torch.FloatTensor(batch_data).unsqueeze(1).to(device)
            y = torch.LongTensor(batch_labels).to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        avg_train_loss = np.mean(train_losses)
        train_acc = 100 * train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        if val_data is not None and val_labels is not None:
            model.eval()
            val_losses = []
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch_data = val_data[i:i+batch_size]
                    batch_labels = val_labels[i:i+batch_size]
                    
                    x = torch.FloatTensor(batch_data).unsqueeze(1).to(device)
                    y = torch.LongTensor(batch_labels).to(device)
                    
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_losses.append(loss.item())
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y.size(0)
                    val_correct += (predicted == y).sum().item()
            
            avg_val_loss = np.mean(val_losses)
            val_acc = 100 * val_correct / val_total
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    
    return history


def evaluate_classifier(model: CNNClassifier,
                       test_data: np.ndarray,
                       test_labels: np.ndarray,
                       batch_size: int = 32,
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    """Evaluate classifier on test data.
    
    Args:
        model: Trained CNN Classifier
        test_data: Test data (n_samples, signal_length)
        test_labels: Test labels (n_samples,)
        batch_size: Batch size
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_data = test_data[i:i+batch_size]
            x = torch.FloatTensor(batch_data).unsqueeze(1).to(device)
            
            outputs = model(x)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(test_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, all_predictions, 
                                                               average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(test_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'probabilities': all_probabilities
    }
