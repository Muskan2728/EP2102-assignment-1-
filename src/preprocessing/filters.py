"""Signal filtering utilities."""
import numpy as np
from scipy import signal
from typing import Optional


class BandpassFilter:
    """Bandpass filter for ECG signal preprocessing."""
    
    def __init__(self, lowcut: float = 0.5, highcut: float = 50.0, 
                 fs: int = 360, order: int = 4):
        """Initialize bandpass filter.
        
        Args:
            lowcut: Low cutoff frequency in Hz
            highcut: High cutoff frequency in Hz
            fs: Sampling frequency in Hz
            order: Filter order
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        
        # Design filter coefficients
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        self.b, self.a = signal.butter(order, [low, high], btype='band')
    
    def filter(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """Apply bandpass filter to signal.
        
        Args:
            data: Input signal
            axis: Axis along which to filter
            
        Returns:
            Filtered signal
        """
        # Use filtfilt for zero-phase filtering
        filtered = signal.filtfilt(self.b, self.a, data, axis=axis)
        return filtered
    
    def filter_batch(self, data: np.ndarray) -> np.ndarray:
        """Apply filter to batch of signals.
        
        Args:
            data: Batch of signals (batch_size, signal_length)
            
        Returns:
            Filtered signals
        """
        if data.ndim == 1:
            return self.filter(data)
        
        filtered_batch = np.zeros_like(data)
        for i in range(len(data)):
            filtered_batch[i] = self.filter(data[i])
        
        return filtered_batch


class NotchFilter:
    """Notch filter for powerline interference removal."""
    
    def __init__(self, freq: float = 60.0, fs: int = 360, quality: float = 30.0):
        """Initialize notch filter.
        
        Args:
            freq: Frequency to remove (Hz), typically 50 or 60 Hz
            fs: Sampling frequency in Hz
            quality: Quality factor
        """
        self.freq = freq
        self.fs = fs
        self.quality = quality
        
        # Design notch filter
        self.b, self.a = signal.iirnotch(freq, quality, fs)
    
    def filter(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter to signal.
        
        Args:
            data: Input signal
            
        Returns:
            Filtered signal
        """
        filtered = signal.filtfilt(self.b, self.a, data)
        return filtered


class HighpassFilter:
    """Highpass filter for baseline wander removal."""
    
    def __init__(self, cutoff: float = 0.5, fs: int = 360, order: int = 4):
        """Initialize highpass filter.
        
        Args:
            cutoff: Cutoff frequency in Hz
            fs: Sampling frequency in Hz
            order: Filter order
        """
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        
        # Design filter
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        self.b, self.a = signal.butter(order, normal_cutoff, btype='high')
    
    def filter(self, data: np.ndarray) -> np.ndarray:
        """Apply highpass filter to signal.
        
        Args:
            data: Input signal
            
        Returns:
            Filtered signal
        """
        filtered = signal.filtfilt(self.b, self.a, data)
        return filtered


class LowpassFilter:
    """Lowpass filter for high-frequency noise removal."""
    
    def __init__(self, cutoff: float = 50.0, fs: int = 360, order: int = 4):
        """Initialize lowpass filter.
        
        Args:
            cutoff: Cutoff frequency in Hz
            fs: Sampling frequency in Hz
            order: Filter order
        """
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        
        # Design filter
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low')
    
    def filter(self, data: np.ndarray) -> np.ndarray:
        """Apply lowpass filter to signal.
        
        Args:
            data: Input signal
            
        Returns:
            Filtered signal
        """
        filtered = signal.filtfilt(self.b, self.a, data)
        return filtered
