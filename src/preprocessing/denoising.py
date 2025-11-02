"""Signal denoising utilities using wavelet transform."""
import numpy as np
import pywt
from typing import Optional, Tuple
from .filters import BandpassFilter, NotchFilter, HighpassFilter


class SignalDenoiser:
    """Comprehensive signal denoising pipeline for ECG signals."""
    
    def __init__(self, wavelet: str = 'db4', level: int = 6, 
                 threshold_mode: str = 'soft', fs: int = 360):
        """Initialize signal denoiser.
        
        Args:
            wavelet: Wavelet type (e.g., 'db4', 'sym4', 'coif4')
            level: Decomposition level
            threshold_mode: 'soft' or 'hard' thresholding
            fs: Sampling frequency in Hz
        """
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode
        self.fs = fs
        
        # Initialize filters
        self.bandpass_filter = BandpassFilter(lowcut=0.5, highcut=50.0, fs=fs)
        self.notch_filter = NotchFilter(freq=60.0, fs=fs)
        self.highpass_filter = HighpassFilter(cutoff=0.5, fs=fs)
    
    def wavelet_denoise(self, signal: np.ndarray, 
                        threshold: Optional[float] = None) -> np.ndarray:
        """Denoise signal using wavelet transform.
        
        Args:
            signal: Input signal
            threshold: Threshold value (if None, computed automatically)
            
        Returns:
            Denoised signal
        """
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        # Calculate threshold if not provided (using universal threshold)
        if threshold is None:
            # Estimate noise standard deviation from finest scale coefficients
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Apply thresholding to detail coefficients
        coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
        for i in range(1, len(coeffs)):
            if self.threshold_mode == 'soft':
                coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            else:
                coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='hard'))
        
        # Reconstruct signal
        denoised = pywt.waverec(coeffs_thresh, self.wavelet)
        
        # Handle length mismatch due to wavelet decomposition
        if len(denoised) > len(signal):
            denoised = denoised[:len(signal)]
        elif len(denoised) < len(signal):
            denoised = np.pad(denoised, (0, len(signal) - len(denoised)), mode='edge')
        
        return denoised
    
    def remove_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Remove baseline wander using highpass filter.
        
        Args:
            signal: Input signal
            
        Returns:
            Signal with baseline removed
        """
        return self.highpass_filter.filter(signal)
    
    def remove_powerline_interference(self, signal: np.ndarray) -> np.ndarray:
        """Remove powerline interference (50/60 Hz).
        
        Args:
            signal: Input signal
            
        Returns:
            Signal with powerline interference removed
        """
        return self.notch_filter.filter(signal)
    
    def denoise_full_pipeline(self, signal: np.ndarray, 
                             apply_bandpass: bool = True,
                             apply_notch: bool = True,
                             apply_wavelet: bool = True) -> np.ndarray:
        """Apply complete denoising pipeline.
        
        Args:
            signal: Input signal
            apply_bandpass: Apply bandpass filter
            apply_notch: Apply notch filter for powerline interference
            apply_wavelet: Apply wavelet denoising
            
        Returns:
            Denoised signal
        """
        denoised = signal.copy()
        
        # Step 1: Remove baseline wander and high-frequency noise
        if apply_bandpass:
            denoised = self.bandpass_filter.filter(denoised)
        
        # Step 2: Remove powerline interference
        if apply_notch:
            denoised = self.remove_powerline_interference(denoised)
        
        # Step 3: Wavelet denoising
        if apply_wavelet:
            denoised = self.wavelet_denoise(denoised)
        
        return denoised
    
    def denoise_batch(self, signals: np.ndarray, **kwargs) -> np.ndarray:
        """Denoise batch of signals.
        
        Args:
            signals: Batch of signals (batch_size, signal_length)
            **kwargs: Arguments for denoise_full_pipeline
            
        Returns:
            Denoised signals
        """
        if signals.ndim == 1:
            return self.denoise_full_pipeline(signals, **kwargs)
        
        denoised_batch = np.zeros_like(signals)
        for i in range(len(signals)):
            denoised_batch[i] = self.denoise_full_pipeline(signals[i], **kwargs)
        
        return denoised_batch
    
    def compute_snr(self, original: np.ndarray, denoised: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio.
        
        Args:
            original: Original signal
            denoised: Denoised signal
            
        Returns:
            SNR in dB
        """
        noise = original - denoised
        signal_power = np.mean(denoised ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def get_wavelet_coefficients(self, signal: np.ndarray) -> Tuple[list, list]:
        """Get wavelet decomposition coefficients.
        
        Args:
            signal: Input signal
            
        Returns:
            Tuple of (coefficients, coefficient_slices)
        """
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        return coeffs, [len(c) for c in coeffs]


class AdaptiveDenoiser:
    """Adaptive denoising based on signal quality assessment."""
    
    def __init__(self, fs: int = 360):
        """Initialize adaptive denoiser.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        self.denoiser = SignalDenoiser(fs=fs)
    
    def assess_signal_quality(self, signal: np.ndarray) -> float:
        """Assess signal quality based on various metrics.
        
        Args:
            signal: Input signal
            
        Returns:
            Quality score (0-1, higher is better)
        """
        # Compute multiple quality metrics
        
        # 1. Signal-to-noise ratio estimate
        coeffs = pywt.wavedec(signal, 'db4', level=3)
        noise_std = np.std(coeffs[-1])
        signal_std = np.std(signal)
        snr_score = 1 / (1 + noise_std / signal_std)
        
        # 2. Baseline stability (low-frequency content)
        from scipy import signal as sp_signal
        f, psd = sp_signal.welch(signal, fs=self.fs, nperseg=min(256, len(signal)))
        low_freq_power = np.sum(psd[f < 1.0])
        total_power = np.sum(psd)
        baseline_score = 1 - (low_freq_power / total_power)
        
        # 3. High-frequency noise
        high_freq_power = np.sum(psd[f > 50.0])
        noise_score = 1 - (high_freq_power / total_power)
        
        # Combine scores
        quality = (snr_score + baseline_score + noise_score) / 3
        return np.clip(quality, 0, 1)
    
    def adaptive_denoise(self, signal: np.ndarray) -> np.ndarray:
        """Apply adaptive denoising based on signal quality.
        
        Args:
            signal: Input signal
            
        Returns:
            Denoised signal
        """
        quality = self.assess_signal_quality(signal)
        
        # Adjust denoising parameters based on quality
        if quality > 0.7:
            # High quality: minimal denoising
            denoised = self.denoiser.denoise_full_pipeline(
                signal, apply_bandpass=True, apply_notch=False, apply_wavelet=False
            )
        elif quality > 0.4:
            # Medium quality: standard denoising
            denoised = self.denoiser.denoise_full_pipeline(
                signal, apply_bandpass=True, apply_notch=True, apply_wavelet=True
            )
        else:
            # Low quality: aggressive denoising
            denoised = self.denoiser.denoise_full_pipeline(
                signal, apply_bandpass=True, apply_notch=True, apply_wavelet=True
            )
            # Apply additional smoothing
            from scipy.ndimage import gaussian_filter1d
            denoised = gaussian_filter1d(denoised, sigma=2)
        
        return denoised
