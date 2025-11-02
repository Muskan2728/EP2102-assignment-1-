"""MIT-BIH Arrhythmia Database loader."""
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings


class MITBIHLoader:
    """Loader for MIT-BIH Arrhythmia Database.
    
    The MIT-BIH database contains 48 half-hour excerpts of two-channel 
    ambulatory ECG recordings, sampled at 360 Hz.
    """
    
    # Annotation mapping for arrhythmia types
    ANNOTATION_MAP = {
        'N': 'Normal',
        'L': 'Left bundle branch block',
        'R': 'Right bundle branch block',
        'A': 'Atrial premature contraction',
        'a': 'Aberrated atrial premature contraction',
        'J': 'Nodal (junctional) premature contraction',
        'S': 'Supraventricular premature contraction',
        'V': 'Premature ventricular contraction',
        'F': 'Fusion of ventricular and normal',
        '[': 'Start of ventricular flutter/fibrillation',
        '!': 'Ventricular flutter wave',
        ']': 'End of ventricular flutter/fibrillation',
        'e': 'Atrial escape beat',
        'j': 'Nodal (junctional) escape beat',
        'E': 'Ventricular escape beat',
        '/': 'Paced beat',
        'f': 'Fusion of paced and normal beat',
        'x': 'Non-conducted P-wave (blocked APB)',
        'Q': 'Unclassifiable beat',
        '|': 'Isolated QRS-like artifact',
    }
    
    # Simplified classification (5 classes)
    SIMPLIFIED_CLASSES = {
        'N': 0,  # Normal
        'L': 0,
        'R': 0,
        'A': 1,  # Atrial Premature Contraction
        'a': 1,
        'S': 1,
        'V': 2,  # Premature Ventricular Contraction
        '[': 3,  # Ventricular Fibrillation
        '!': 3,
        ']': 3,
        'F': 4,  # Fusion
        'f': 4,
    }
    
    def __init__(self, data_path: str = "data/raw/", sampling_rate: int = 360):
        """Initialize MIT-BIH loader.
        
        Args:
            data_path: Path to store/load MIT-BIH data
            sampling_rate: Sampling rate in Hz (default: 360)
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.sampling_rate = sampling_rate
        
    def download_record(self, record_name: str, database: str = "mitdb") -> bool:
        """Download a single record from PhysioNet.
        
        Args:
            record_name: Record identifier (e.g., "100")
            database: Database name (default: "mitdb")
            
        Returns:
            bool: True if successful
        """
        try:
            wfdb.dl_database(database, str(self.data_path), records=[record_name])
            print(f"Downloaded record {record_name}")
            return True
        except Exception as e:
            warnings.warn(f"Failed to download record {record_name}: {e}")
            return False
    
    def load_record(self, record_name: str, channel: int = 0) -> Tuple[np.ndarray, Dict]:
        """Load ECG signal and annotations for a record.
        
        Args:
            record_name: Record identifier (e.g., "100")
            channel: Channel to load (0 or 1)
            
        Returns:
            Tuple of (signal, metadata)
        """
        record_path = self.data_path / record_name
        
        # Load signal
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal[:, channel]
        
        # Load annotations
        annotation = wfdb.rdann(str(record_path), 'atr')
        
        metadata = {
            'record_name': record_name,
            'sampling_rate': record.fs,
            'signal_length': len(signal),
            'duration': len(signal) / record.fs,
            'annotations': annotation.symbol,
            'annotation_samples': annotation.sample,
            'num_beats': len(annotation.sample)
        }
        
        return signal, metadata
    
    def segment_signal(self, signal: np.ndarray, segment_length: int, 
                       overlap: float = 0.5) -> List[np.ndarray]:
        """Segment signal into fixed-length windows.
        
        Args:
            signal: Input ECG signal
            segment_length: Length of each segment in samples
            overlap: Overlap ratio between segments (0-1)
            
        Returns:
            List of signal segments
        """
        step = int(segment_length * (1 - overlap))
        segments = []
        
        for start in range(0, len(signal) - segment_length + 1, step):
            segment = signal[start:start + segment_length]
            segments.append(segment)
        
        return segments
    
    def extract_beats(self, signal: np.ndarray, annotation_samples: np.ndarray,
                      window_size: int = 180) -> Tuple[List[np.ndarray], List[int]]:
        """Extract individual heartbeats centered on R-peaks.
        
        Args:
            signal: Input ECG signal
            annotation_samples: Sample indices of R-peaks
            window_size: Number of samples before and after R-peak
            
        Returns:
            Tuple of (beat_segments, valid_indices)
        """
        beats = []
        valid_indices = []
        
        for idx, r_peak in enumerate(annotation_samples):
            start = r_peak - window_size
            end = r_peak + window_size
            
            # Check boundaries
            if start >= 0 and end < len(signal):
                beat = signal[start:end]
                beats.append(beat)
                valid_indices.append(idx)
        
        return beats, valid_indices
    
    def create_dataset(self, record_names: List[str], segment_length: int = 3600,
                       extract_beats_mode: bool = False) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Create dataset from multiple records.
        
        Args:
            record_names: List of record identifiers
            segment_length: Length of segments in samples (default: 3600 = 10s at 360Hz)
            extract_beats_mode: If True, extract individual beats instead of segments
            
        Returns:
            Tuple of (signals, labels, metadata_df)
        """
        all_signals = []
        all_labels = []
        metadata_list = []
        
        for record_name in record_names:
            try:
                signal, metadata = self.load_record(record_name)
                
                if extract_beats_mode:
                    # Extract individual beats
                    beats, valid_indices = self.extract_beats(
                        signal, metadata['annotation_samples']
                    )
                    
                    # Get labels for valid beats
                    annotations = np.array(metadata['annotations'])
                    beat_labels = [
                        self.SIMPLIFIED_CLASSES.get(annotations[i], -1)
                        for i in valid_indices
                    ]
                    
                    # Filter out unknown labels
                    valid_beats = [
                        beat for beat, label in zip(beats, beat_labels) 
                        if label != -1
                    ]
                    valid_labels = [label for label in beat_labels if label != -1]
                    
                    all_signals.extend(valid_beats)
                    all_labels.extend(valid_labels)
                    
                    for i, (beat, label) in enumerate(zip(valid_beats, valid_labels)):
                        metadata_list.append({
                            'record': record_name,
                            'segment_id': i,
                            'label': label,
                            'length': len(beat)
                        })
                else:
                    # Segment signal
                    segments = self.segment_signal(signal, segment_length)
                    
                    # For segments, use majority voting or first annotation
                    # Here we'll use a simple approach: label as normal (0) by default
                    segment_labels = [0] * len(segments)
                    
                    all_signals.extend(segments)
                    all_labels.extend(segment_labels)
                    
                    for i, segment in enumerate(segments):
                        metadata_list.append({
                            'record': record_name,
                            'segment_id': i,
                            'label': 0,
                            'length': len(segment)
                        })
                
                print(f"Processed record {record_name}")
                
            except Exception as e:
                warnings.warn(f"Failed to process record {record_name}: {e}")
                continue
        
        # Convert to numpy arrays
        signals_array = np.array(all_signals)
        labels_array = np.array(all_labels)
        metadata_df = pd.DataFrame(metadata_list)
        
        return signals_array, labels_array, metadata_df
    
    def get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """Get distribution of classes in dataset.
        
        Args:
            labels: Array of labels
            
        Returns:
            Dictionary of class counts
        """
        class_names = ['Normal', 'APC', 'PVC', 'VF', 'Fusion']
        unique, counts = np.unique(labels, return_counts=True)
        
        distribution = {}
        for label, count in zip(unique, counts):
            if 0 <= label < len(class_names):
                distribution[class_names[label]] = int(count)
        
        return distribution
