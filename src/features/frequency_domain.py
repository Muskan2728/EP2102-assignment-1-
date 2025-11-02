"""Frequency-domain feature extraction for ECG signals."""
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, Optional


class FrequencyDomainFeatures:
    """Extract frequency-domain features from ECG signals."""
    
    def __init__(self, fs: int = 360):
        """Initialize frequency-domain feature extractor.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        
        # HRV frequency bands (for RR intervals)
        self.vlf_band = (0.003, 0.04)  # Very low frequency
        self.lf_band = (0.04, 0.15)    # Low frequency
        self.hf_band = (0.15, 0.4)     # High frequency
    
    def compute_fft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Fast Fourier Transform.
        
        Args:
            signal_data: Input signal
            
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        n = len(signal_data)
        
        # Compute FFT
        fft_values = fft(signal_data)
        fft_freqs = fftfreq(n, 1/self.fs)
        
        # Take only positive frequencies
        positive_freqs = fft_freqs[:n//2]
        magnitudes = np.abs(fft_values[:n//2])
        
        return positive_freqs, magnitudes
    
    def compute_power_spectral_density(self, signal_data: np.ndarray,
                                      nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Power Spectral Density using Welch's method.
        
        Args:
            signal_data: Input signal
            nperseg: Length of each segment for Welch's method
            
        Returns:
            Tuple of (frequencies, psd)
        """
        if nperseg is None:
            nperseg = min(256, len(signal_data))
        
        freqs, psd = signal.welch(signal_data, fs=self.fs, nperseg=nperseg)
        return freqs, psd
    
    def compute_spectral_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Compute spectral features from signal.
        
        Args:
            signal_data: Input signal
            
        Returns:
            Dictionary of spectral features
        """
        freqs, psd = self.compute_power_spectral_density(signal_data)
        
        # Total power
        total_power = np.sum(psd)
        
        # Spectral centroid (center of mass of spectrum)
        spectral_centroid = np.sum(freqs * psd) / total_power if total_power > 0 else 0
        
        # Spectral spread (standard deviation around centroid)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power) \
                         if total_power > 0 else 0
        
        # Spectral entropy
        psd_norm = psd / total_power if total_power > 0 else psd
        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm)) if len(psd_norm) > 0 else 0
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum_psd = np.cumsum(psd)
        rolloff_idx = np.where(cumsum_psd >= 0.85 * total_power)[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        # Dominant frequency
        dominant_freq = freqs[np.argmax(psd)]
        
        return {
            'total_power': float(total_power),
            'spectral_centroid': float(spectral_centroid),
            'spectral_spread': float(spectral_spread),
            'spectral_entropy': float(spectral_entropy),
            'spectral_rolloff': float(spectral_rolloff),
            'dominant_frequency': float(dominant_freq)
        }
    
    def compute_band_power(self, signal_data: np.ndarray, 
                          band: Tuple[float, float]) -> float:
        """Compute power in specific frequency band.
        
        Args:
            signal_data: Input signal
            band: Tuple of (low_freq, high_freq) in Hz
            
        Returns:
            Power in the specified band
        """
        freqs, psd = self.compute_power_spectral_density(signal_data)
        
        # Find indices for frequency band
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        
        # Compute power in band
        band_power = np.trapz(psd[idx_band], freqs[idx_band])
        
        return float(band_power)
    
    def compute_hrv_frequency_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Compute frequency-domain HRV features from RR intervals.
        
        Args:
            rr_intervals: Array of RR intervals in ms
            
        Returns:
            Dictionary of HRV frequency features
        """
        if len(rr_intervals) < 10:
            return {
                'vlf_power': 0.0,
                'lf_power': 0.0,
                'hf_power': 0.0,
                'lf_hf_ratio': 0.0,
                'total_power': 0.0
            }
        
        # Resample RR intervals to uniform time series (4 Hz)
        fs_rr = 4  # Hz
        time_rr = np.cumsum(rr_intervals) / 1000  # Convert to seconds
        time_uniform = np.arange(0, time_rr[-1], 1/fs_rr)
        rr_uniform = np.interp(time_uniform, time_rr, rr_intervals)
        
        # Compute PSD
        freqs, psd = signal.welch(rr_uniform, fs=fs_rr, nperseg=min(256, len(rr_uniform)))
        
        # Compute power in each band
        vlf_power = self._integrate_band(freqs, psd, self.vlf_band)
        lf_power = self._integrate_band(freqs, psd, self.lf_band)
        hf_power = self._integrate_band(freqs, psd, self.hf_band)
        
        total_power = vlf_power + lf_power + hf_power
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        
        return {
            'vlf_power': float(vlf_power),
            'lf_power': float(lf_power),
            'hf_power': float(hf_power),
            'lf_hf_ratio': float(lf_hf_ratio),
            'total_power': float(total_power)
        }
    
    def _integrate_band(self, freqs: np.ndarray, psd: np.ndarray, 
                       band: Tuple[float, float]) -> float:
        """Integrate PSD over frequency band."""
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.trapz(psd[idx_band], freqs[idx_band])
    
    def compute_spectral_edge_frequency(self, signal_data: np.ndarray, 
                                       percentage: float = 0.95) -> float:
        """Compute spectral edge frequency.
        
        Args:
            signal_data: Input signal
            percentage: Percentage of total power (default: 0.95)
            
        Returns:
            Frequency below which percentage of power is contained
        """
        freqs, psd = self.compute_power_spectral_density(signal_data)
        
        total_power = np.sum(psd)
        cumsum_psd = np.cumsum(psd)
        
        idx = np.where(cumsum_psd >= percentage * total_power)[0]
        sef = freqs[idx[0]] if len(idx) > 0 else freqs[-1]
        
        return float(sef)
    
    def extract_all_features(self, signal_data: np.ndarray, 
                           rr_intervals: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Extract all frequency-domain features.
        
        Args:
            signal_data: Input ECG signal
            rr_intervals: Optional RR intervals for HRV analysis
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Spectral features from ECG signal
        spectral_features = self.compute_spectral_features(signal_data)
        features.update({f'freq_{k}': v for k, v in spectral_features.items()})
        
        # Spectral edge frequency
        features['freq_sef_95'] = self.compute_spectral_edge_frequency(signal_data, 0.95)
        
        # HRV frequency features if RR intervals provided
        if rr_intervals is not None and len(rr_intervals) > 0:
            hrv_freq_features = self.compute_hrv_frequency_features(rr_intervals)
            features.update({f'hrv_freq_{k}': v for k, v in hrv_freq_features.items()})
        
        return features
