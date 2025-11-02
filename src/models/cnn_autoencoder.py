"""CNN Autoencoder for unsupervised cardiac anomaly detection.

This module implements a 1D CNN-based autoencoder that learns to reconstruct
normal ECG patterns. Anomalies are detected based on high reconstruction error.

Why CNN is best for unsupervised anomaly detection:
1. Automatic hierarchical feature learning from raw signals
2. Translation invariance - detects patterns regardless of position
3. Local pattern recognition through convolutional filters
4. Efficient computation with fewer parameters than RNNs
5. Autoencoder learns compressed representation of normal patterns
6. High reconstruction error indicates deviation from normal
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class CNNAutoencoder(nn.Module):
    """1D CNN Autoencoder for ECG signal reconstruction.
    
    Architecture:
    - Encoder: Multiple 1D conv layers with downsampling
    - Bottleneck: Compressed latent representation
    - Decoder: Transposed conv layers for reconstruction
    """
    
    def __init__(self, input_size: int = 3600, 
                 encoder_channels: list = [16, 32, 64, 128],
                 latent_dim: int = 64,
                 kernel_size: int = 7,
                 stride: int = 2,
                 dropout: float = 0.2):
        """Initialize CNN Autoencoder.
        
        Args:
            input_size: Length of input signal (e.g., 3600 = 10s at 360Hz)
            encoder_channels: List of channel sizes for encoder layers
            latent_dim: Dimension of latent representation
            kernel_size: Kernel size for convolutions
            stride: Stride for downsampling
            dropout: Dropout rate
        """
        super(CNNAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.encoder_channels = encoder_channels
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        in_channels = 1
        
        for out_channels in encoder_channels:
            encoder_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         stride=stride, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate size after encoder
        self.encoded_size = self._calculate_encoded_size()
        
        # Bottleneck
        self.fc_encode = nn.Linear(encoder_channels[-1] * self.encoded_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, encoder_channels[-1] * self.encoded_size)
        
        # Decoder
        decoder_layers = []
        decoder_channels = encoder_channels[::-1]  # Reverse order
        
        for i in range(len(decoder_channels) - 1):
            decoder_layers.extend([
                nn.ConvTranspose1d(decoder_channels[i], decoder_channels[i+1],
                                  kernel_size, stride=stride, 
                                  padding=kernel_size//2, output_padding=1),
                nn.BatchNorm1d(decoder_channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
        
        # Final layer to reconstruct signal
        decoder_layers.append(
            nn.ConvTranspose1d(decoder_channels[-1], 1, kernel_size,
                              stride=stride, padding=kernel_size//2, output_padding=1)
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _calculate_encoded_size(self) -> int:
        """Calculate size of encoded representation."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_size)
            x = self.encoder(x)
            return x.shape[2]
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input tensor (batch_size, 1, signal_length)
            
        Returns:
            Latent representation (batch_size, latent_dim)
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encode(x)
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to signal.
        
        Args:
            z: Latent representation (batch_size, latent_dim)
            
        Returns:
            Reconstructed signal (batch_size, 1, signal_length)
        """
        x = self.fc_decode(z)
        x = x.view(x.size(0), self.encoder_channels[-1], self.encoded_size)
        x = self.decoder(x)
        
        # Adjust size to match input
        if x.size(2) != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='linear', align_corners=False)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder.
        
        Args:
            x: Input tensor (batch_size, 1, signal_length)
            
        Returns:
            Tuple of (reconstructed signal, latent representation)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def compute_reconstruction_error(self, x: torch.Tensor, 
                                    x_recon: torch.Tensor,
                                    reduction: str = 'none') -> torch.Tensor:
        """Compute reconstruction error (MSE).
        
        Args:
            x: Original signal
            x_recon: Reconstructed signal
            reduction: 'none', 'mean', or 'sum'
            
        Returns:
            Reconstruction error
        """
        if reduction == 'none':
            # Per-sample error
            error = torch.mean((x - x_recon) ** 2, dim=(1, 2))
        elif reduction == 'mean':
            error = torch.mean((x - x_recon) ** 2)
        else:
            error = torch.sum((x - x_recon) ** 2)
        
        return error


class AnomalyDetector:
    """Anomaly detector using trained CNN Autoencoder.
    
    Detects anomalies based on reconstruction error threshold.
    """
    
    def __init__(self, model: CNNAutoencoder, threshold: Optional[float] = None):
        """Initialize anomaly detector.
        
        Args:
            model: Trained CNN Autoencoder
            threshold: Reconstruction error threshold (computed if None)
        """
        self.model = model
        self.threshold = threshold
        self.device = next(model.parameters()).device
    
    def fit_threshold(self, normal_data: np.ndarray, percentile: float = 95):
        """Compute threshold from normal data.
        
        Args:
            normal_data: Normal ECG signals (n_samples, signal_length)
            percentile: Percentile for threshold (default: 95)
        """
        self.model.eval()
        
        # Compute reconstruction errors for normal data
        errors = []
        
        with torch.no_grad():
            for i in range(0, len(normal_data), 32):
                batch = normal_data[i:i+32]
                x = torch.FloatTensor(batch).unsqueeze(1).to(self.device)
                x_recon, _ = self.model(x)
                error = self.model.compute_reconstruction_error(x, x_recon, reduction='none')
                errors.extend(error.cpu().numpy())
        
        # Set threshold at specified percentile
        self.threshold = np.percentile(errors, percentile)
        print(f"Threshold set at {percentile}th percentile: {self.threshold:.6f}")
    
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies in data.
        
        Args:
            data: ECG signals (n_samples, signal_length)
            
        Returns:
            Tuple of (predictions, reconstruction_errors)
            predictions: 0 for normal, 1 for anomaly
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold first.")
        
        self.model.eval()
        
        errors = []
        
        with torch.no_grad():
            for i in range(0, len(data), 32):
                batch = data[i:i+32]
                x = torch.FloatTensor(batch).unsqueeze(1).to(self.device)
                x_recon, _ = self.model(x)
                error = self.model.compute_reconstruction_error(x, x_recon, reduction='none')
                errors.extend(error.cpu().numpy())
        
        errors = np.array(errors)
        predictions = (errors > self.threshold).astype(int)
        
        return predictions, errors
    
    def get_anomaly_score(self, signal: np.ndarray) -> float:
        """Get anomaly score for single signal.
        
        Args:
            signal: Single ECG signal (signal_length,)
            
        Returns:
            Anomaly score (reconstruction error)
        """
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(self.device)
            x_recon, _ = self.model(x)
            error = self.model.compute_reconstruction_error(x, x_recon, reduction='none')
        
        return float(error.cpu().numpy()[0])
    
    def reconstruct(self, signal: np.ndarray) -> np.ndarray:
        """Reconstruct signal using autoencoder.
        
        Args:
            signal: Input ECG signal (signal_length,)
            
        Returns:
            Reconstructed signal
        """
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(self.device)
            x_recon, _ = self.model(x)
        
        return x_recon.cpu().numpy()[0, 0, :]


def train_autoencoder(model: CNNAutoencoder, 
                     train_data: np.ndarray,
                     val_data: Optional[np.ndarray] = None,
                     epochs: int = 50,
                     batch_size: int = 32,
                     learning_rate: float = 0.001,
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                     early_stopping_patience: int = 10) -> Dict:
    """Train CNN Autoencoder.
    
    Args:
        model: CNN Autoencoder model
        train_data: Training data (n_samples, signal_length)
        val_data: Validation data
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        early_stopping_patience: Patience for early stopping
        
    Returns:
        Dictionary of training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        # Shuffle training data
        indices = np.random.permutation(len(train_data))
        
        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = train_data[batch_indices]
            
            x = torch.FloatTensor(batch).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            x_recon, _ = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        if val_data is not None:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch = val_data[i:i+batch_size]
                    x = torch.FloatTensor(batch).unsqueeze(1).to(device)
                    x_recon, _ = model(x)
                    loss = criterion(x_recon, x)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
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
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
    
    return history
