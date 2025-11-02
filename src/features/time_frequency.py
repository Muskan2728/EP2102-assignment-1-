"""Time-frequency feature extraction for ECG signals."""
import numpy as np
import pywt
from scipy import signal
from typing import Dict, Tuple, Optional


class TimeFrequencyFeatures:
    """Extract time-frequency features from ECG signals."""
    
    def __init__(self, fs: int = 360):
        """Initialize time-frequency feature extractor.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
    
    def compute_stft(self, signal_data: np.ndarray, 
                    window: str = 'hann',
                    nperseg: Optional[int] = None,
                    noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Short-Time Fourier Transform.
        
        Args:
            signal_data: Input signal
            window: Window function
            nperseg: Length of each segment
            noverlap: Number of points to overlap
            
        Returns:
            Tuple of (frequencies, times, STFT magnitude)
        """
        if nperseg is None:
            nperseg = min(256, len(signal_data) // 4)
        if noverlap is None:
            noverlap = nperseg // 2
        
        freqs, times, stft = signal.stft(signal_data, fs=self.fs, 
                                         window=window, nperseg=nperseg, 
                                         noverlap=noverlap)
        
        return freqs, times, np.abs(stft)
    
    def compute_cwt(self, signal_data: np.ndarray, 
                   wavelet: str = 'morl',
                   scales: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Continuous Wavelet Transform.
        
        Args:
            signal_data: Input signal
            wavelet: Wavelet type (e.g., 'morl', 'mexh', 'gaus1')
            scales: Array of scales to use
            
        Returns:
            Tuple of (coefficients, frequencies)
        """
        if scales is None:
            # Default scales corresponding to frequencies of interest (0.5-50 Hz)
            scales = np.arange(1, 128)
        
        coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet, 
                                            sampling_period=1/self.fs)
        
        return np.abs(coefficients), frequencies
    
    def compute_dwt_features(self, signal_data: np.ndarray, 
                           wavelet: str = 'db4',
                           level: int = 6) -> Dict[str, np.ndarray]:
        """Compute Discrete Wavelet Transform features.
        
        Args:
            signal_data: Input signal
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Dictionary of wavelet coefficients
        """
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        features = {
            'approximation': coeffs[0]
        }
        
        for i, detail in enumerate(coeffs[1:], 1):
            features[f'detail_{i}'] = detail
        
        return features
    
    def compute_wavelet_energy(self, signal_data: np.ndarray,
                              wavelet: str = 'db4',
                              level: int = 6) -> Dict[str, float]:
        """Compute energy in each wavelet decomposition level.
        
        Args:
            signal_data: Input signal
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Dictionary of energy values
        """
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # Compute energy for each level
        energies = {}
        
        # Approximation energy
        energies['approx_energy'] = float(np.sum(coeffs[0] ** 2))
        
        # Detail energies
        for i, detail in enumerate(coeffs[1:], 1):
            energies[f'detail_{i}_energy'] = float(np.sum(detail ** 2))
        
        # Total energy
        total_energy = sum(energies.values())
        energies['total_energy'] = total_energy
        
        # Relative energies
        if total_energy > 0:
            for key in list(energies.keys()):
                if key != 'total_energy':
                    energies[f'{key}_relative'] = energies[key] / total_energy
        
        return energies
    
    def compute_wavelet_entropy(self, signal_data: np.ndarray,
                               wavelet: str = 'db4',
                               level: int = 6) -> float:
        """Compute wavelet entropy.
        
        Args:
            signal_data: Input signal
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Wavelet entropy value
        """
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # Compute energy for each level
        energies = []
        energies.append(np.sum(coeffs[0] ** 2))
        for detail in coeffs[1:]:
            energies.append(np.sum(detail ** 2))
        
        energies = np.array(energies)
        total_energy = np.sum(energies)
        
        if total_energy == 0:
            return 0.0
        
        # Normalize energies
        p = energies / total_energy
        
        # Remove zeros
        p = p[p > 0]
        
        # Compute entropy
        entropy = -np.sum(p * np.log2(p))
        
        return float(entropy)
    
    def compute_stft_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Compute features from STFT.
        
        Args:
            signal_data: Input signal
            
        Returns:
            Dictionary of STFT features
        """
        freqs, times, stft_mag = self.compute_stft(signal_data)
        
        # Time-averaged spectrum
        avg_spectrum = np.mean(stft_mag, axis=1)
        
        # Spectral flux (change in spectrum over time)
        spectral_flux = np.mean(np.diff(stft_mag, axis=1) ** 2)
        
        # Spectral centroid over time
        spectral_centroids = []
        for t in range(stft_mag.shape[1]):
            spectrum = stft_mag[:, t]
            total_power = np.sum(spectrum)
            if total_power > 0:
                centroid = np.sum(freqs * spectrum) / total_power
                spectral_centroids.append(centroid)
        
        mean_centroid = np.mean(spectral_centroids) if spectral_centroids else 0
        std_centroid = np.std(spectral_centroids) if spectral_centroids else 0
        
        # Spectral bandwidth over time
        spectral_bandwidths = []
        for t in range(stft_mag.shape[1]):
            spectrum = stft_mag[:, t]
            total_power = np.sum(spectrum)
            if total_power > 0:
                centroid = np.sum(freqs * spectrum) / total_power
                bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total_power)
                spectral_bandwidths.append(bandwidth)
        
        mean_bandwidth = np.mean(spectral_bandwidths) if spectral_bandwidths else 0
        
        return {
            'stft_spectral_flux': float(spectral_flux),
            'stft_mean_centroid': float(mean_centroid),
            'stft_std_centroid': float(std_centroid),
            'stft_mean_bandwidth': float(mean_bandwidth)
        }
    
    def compute_scalogram_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Compute features from wavelet scalogram (CWT).
        
        Args:
            signal_data: Input signal
            
        Returns:
            Dictionary of scalogram features
        """
        coeffs, freqs = self.compute_cwt(signal_data)
        
        # Mean and std of coefficients across time
        mean_coeffs = np.mean(coeffs, axis=1)
        std_coeffs = np.std(coeffs, axis=1)
        
        # Maximum coefficient at each scale
        max_coeffs = np.max(coeffs, axis=1)
        
        # Overall statistics
        features = {
            'cwt_mean_energy': float(np.mean(coeffs ** 2)),
            'cwt_max_energy': float(np.max(coeffs ** 2)),
            'cwt_mean_coeff': float(np.mean(mean_coeffs)),
            'cwt_std_coeff': float(np.mean(std_coeffs)),
            'cwt_max_coeff': float(np.max(max_coeffs))
        }
        
        return features
    
    def extract_all_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract all time-frequency features.
        
        Args:
            signal_data: Input ECG signal
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Wavelet energy features
        wavelet_energy = self.compute_wavelet_energy(signal_data)
        features.update({f'wavelet_{k}': v for k, v in wavelet_energy.items()})
        
        # Wavelet entropy
        features['wavelet_entropy'] = self.compute_wavelet_entropy(signal_data)
        
        # STFT features
        stft_features = self.compute_stft_features(signal_data)
        features.update(stft_features)
        
        # Scalogram features (CWT)
        try:
            scalogram_features = self.compute_scalogram_features(signal_data)
            features.update(scalogram_features)
        except Exception:
            # CWT might fail for very short signals
            pass
        
        return features
