"""Time-domain feature extraction for ECG signals."""
import numpy as np
from scipy import signal
from typing import Dict, List, Optional


class TimeDomainFeatures:
    """Extract time-domain features from ECG signals."""
    
    def __init__(self, fs: int = 360):
        """Initialize time-domain feature extractor.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
    
    def detect_r_peaks(self, ecg_signal: np.ndarray, 
                       distance: Optional[int] = None) -> np.ndarray:
        """Detect R-peaks in ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            distance: Minimum distance between peaks (samples)
            
        Returns:
            Array of R-peak indices
        """
        if distance is None:
            # Minimum distance = 0.6 * average RR interval (assume 60 bpm minimum)
            distance = int(0.6 * self.fs)
        
        # Find peaks
        peaks, _ = signal.find_peaks(ecg_signal, distance=distance, 
                                     prominence=np.std(ecg_signal) * 0.5)
        return peaks
    
    def compute_rr_intervals(self, r_peaks: np.ndarray) -> np.ndarray:
        """Compute RR intervals from R-peaks.
        
        Args:
            r_peaks: Array of R-peak indices
            
        Returns:
            Array of RR intervals in milliseconds
        """
        if len(r_peaks) < 2:
            return np.array([])
        
        rr_intervals = np.diff(r_peaks) / self.fs * 1000  # Convert to ms
        return rr_intervals
    
    def compute_heart_rate(self, rr_intervals: np.ndarray) -> float:
        """Compute average heart rate.
        
        Args:
            rr_intervals: Array of RR intervals in ms
            
        Returns:
            Heart rate in bpm
        """
        if len(rr_intervals) == 0:
            return 0.0
        
        mean_rr = np.mean(rr_intervals)
        heart_rate = 60000 / mean_rr  # Convert to bpm
        return heart_rate
    
    def compute_hrv_time_domain(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Compute time-domain HRV (Heart Rate Variability) features.
        
        Args:
            rr_intervals: Array of RR intervals in ms
            
        Returns:
            Dictionary of HRV features
        """
        if len(rr_intervals) < 2:
            return {
                'mean_rr': 0.0,
                'sdnn': 0.0,
                'rmssd': 0.0,
                'nn50': 0,
                'pnn50': 0.0,
                'cv': 0.0
            }
        
        # Mean RR interval
        mean_rr = np.mean(rr_intervals)
        
        # SDNN: Standard deviation of NN intervals
        sdnn = np.std(rr_intervals, ddof=1)
        
        # RMSSD: Root mean square of successive differences
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        
        # NN50: Number of successive differences > 50ms
        nn50 = np.sum(np.abs(successive_diffs) > 50)
        
        # pNN50: Percentage of NN50
        pnn50 = (nn50 / len(successive_diffs)) * 100 if len(successive_diffs) > 0 else 0
        
        # Coefficient of variation
        cv = (sdnn / mean_rr) * 100 if mean_rr > 0 else 0
        
        return {
            'mean_rr': float(mean_rr),
            'sdnn': float(sdnn),
            'rmssd': float(rmssd),
            'nn50': int(nn50),
            'pnn50': float(pnn50),
            'cv': float(cv)
        }
    
    def compute_statistical_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Compute statistical features of signal.
        
        Args:
            signal_data: Input signal
            
        Returns:
            Dictionary of statistical features
        """
        return {
            'mean': float(np.mean(signal_data)),
            'std': float(np.std(signal_data)),
            'var': float(np.var(signal_data)),
            'min': float(np.min(signal_data)),
            'max': float(np.max(signal_data)),
            'median': float(np.median(signal_data)),
            'q25': float(np.percentile(signal_data, 25)),
            'q75': float(np.percentile(signal_data, 75)),
            'iqr': float(np.percentile(signal_data, 75) - np.percentile(signal_data, 25)),
            'skewness': float(self._compute_skewness(signal_data)),
            'kurtosis': float(self._compute_kurtosis(signal_data))
        }
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def compute_morphological_features(self, ecg_signal: np.ndarray, 
                                      r_peaks: np.ndarray) -> Dict[str, float]:
        """Compute morphological features of ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            r_peaks: Array of R-peak indices
            
        Returns:
            Dictionary of morphological features
        """
        if len(r_peaks) == 0:
            return {
                'mean_amplitude': 0.0,
                'std_amplitude': 0.0,
                'mean_width': 0.0
            }
        
        # R-peak amplitudes
        amplitudes = ecg_signal[r_peaks]
        
        # QRS width estimation (simplified)
        qrs_widths = []
        for peak in r_peaks:
            # Look for QRS complex boundaries
            window = 50  # samples
            start = max(0, peak - window)
            end = min(len(ecg_signal), peak + window)
            segment = ecg_signal[start:end]
            
            # Estimate width at half maximum
            half_max = amplitudes[len(qrs_widths)] / 2
            above_half = segment > half_max
            if np.any(above_half):
                width = np.sum(above_half)
                qrs_widths.append(width)
        
        return {
            'mean_amplitude': float(np.mean(amplitudes)),
            'std_amplitude': float(np.std(amplitudes)),
            'mean_width': float(np.mean(qrs_widths)) if qrs_widths else 0.0
        }
    
    def extract_all_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """Extract all time-domain features.
        
        Args:
            ecg_signal: Input ECG signal
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Detect R-peaks
        r_peaks = self.detect_r_peaks(ecg_signal)
        
        # RR intervals and heart rate
        rr_intervals = self.compute_rr_intervals(r_peaks)
        if len(rr_intervals) > 0:
            features['heart_rate'] = self.compute_heart_rate(rr_intervals)
            
            # HRV features
            hrv_features = self.compute_hrv_time_domain(rr_intervals)
            features.update({f'hrv_{k}': v for k, v in hrv_features.items()})
        
        # Statistical features
        stat_features = self.compute_statistical_features(ecg_signal)
        features.update({f'stat_{k}': v for k, v in stat_features.items()})
        
        # Morphological features
        morph_features = self.compute_morphological_features(ecg_signal, r_peaks)
        features.update({f'morph_{k}': v for k, v in morph_features.items()})
        
        return features
