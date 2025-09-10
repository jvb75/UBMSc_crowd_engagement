import os
import argparse
import numpy as np
import librosa
import librosa.display
import pandas as pd
from scipy import signal
from scipy.io import wavfile
import scipy.stats as stats
from tqdm import tqdm
import warnings
import h5py  # For saving large arrays efficiently
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, frame_size=2048, hop_length=512, max_frames=500):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.max_frames = max_frames  # Maximum number of frames to keep for consistent sizing
    
    def load_audio(self, file_path):
        """Load audio file using librosa"""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            return y, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None, None
    
    def pad_or_truncate(self, feature_matrix):
        """Ensure feature matrix has consistent temporal dimension"""
        if feature_matrix.shape[1] > self.max_frames:
            # Truncate if too long
            return feature_matrix[:, :self.max_frames]
        elif feature_matrix.shape[1] < self.max_frames:
            # Pad if too short
            pad_width = self.max_frames - feature_matrix.shape[1]
            return np.pad(feature_matrix, ((0, 0), (0, pad_width)), mode='constant')
        return feature_matrix
    
    def extract_spectrogram_features(self, y):
        """Extract spectrogram features for CNN input"""
        # STFT Spectrogram
        stft = np.abs(librosa.stft(y, n_fft=self.frame_size, hop_length=self.hop_length))
        stft = self.pad_or_truncate(stft)
        
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, 
                                                 n_fft=self.frame_size, hop_length=self.hop_length)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = self.pad_or_truncate(mel_spec)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=40, 
                                    n_fft=self.frame_size, hop_length=self.hop_length)
        mfccs = self.pad_or_truncate(mfccs)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate, 
                                           n_fft=self.frame_size, hop_length=self.hop_length)
        chroma = self.pad_or_truncate(chroma)
        
        return {
            'stft': stft,
            'mel_spectrogram': mel_spec,
            'mfcc': mfccs,
            'chroma': chroma
        }
    
    def extract_time_series_features(self, y):
        """Extract time-series features for LSTM input"""
        features = {}
        
        # Zero Crossing Rate (time series)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_size, 
                                                hop_length=self.hop_length)
        features['zcr'] = self.pad_or_truncate(zcr)
        
        # RMS Energy (time series)
        rms = librosa.feature.rms(y=y, frame_length=self.frame_size, 
                                 hop_length=self.hop_length)
        features['rms'] = self.pad_or_truncate(rms)
        
        # Spectral Centroid (time series)
        S = np.abs(librosa.stft(y, n_fft=self.frame_size, hop_length=self.hop_length))
        spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=self.sample_rate)
        features['spectral_centroid'] = self.pad_or_truncate(spectral_centroid)
        
        # Spectral Bandwidth (time series)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=self.sample_rate)
        features['spectral_bandwidth'] = self.pad_or_truncate(spectral_bandwidth)
        
        # Spectral Rolloff (time series)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=self.sample_rate)
        features['spectral_rolloff'] = self.pad_or_truncate(spectral_rolloff)
        
        return features
    
    def extract_multi_scale_features(self, y):
        """Extract features at multiple time scales"""
        features = {}
        
        # Short-term features (frame-level)
        stft = np.abs(librosa.stft(y, n_fft=self.frame_size, hop_length=self.hop_length))
        features['stft'] = self.pad_or_truncate(stft)
        
        # Medium-term features (segment-level)
        segment_length = 3 * self.sample_rate  # 3-second segments
        if len(y) > segment_length:
            segments = []
            for i in range(0, len(y) - segment_length, segment_length // 2):
                segment = y[i:i + segment_length]
                segment_features = self.extract_segment_features(segment)
                segments.append(segment_features)
            
            if segments:
                # Stack segment features
                segment_features = np.stack(segments, axis=0)
                features['segment_features'] = segment_features
        
        return features
    
    def extract_segment_features(self, segment):
        """Extract features for a single audio segment"""
        # MFCCs for the segment
        mfccs = librosa.feature.mfcc(y=segment, sr=self.sample_rate, n_mfcc=13, 
                                   n_fft=self.frame_size, hop_length=self.hop_length)
        return np.mean(mfccs, axis=1)  # Mean across time for segment-level features
    
    def extract_deep_learning_features(self, y):
        """Extract all features suitable for CNN+LSTM models"""
        features = {}
        
        # Spectrogram-based features (for CNN)
        spectrogram_features = self.extract_spectrogram_features(y)
        features.update(spectrogram_features)
        
        # Time-series features (for LSTM)
        time_series_features = self.extract_time_series_features(y)
        features.update(time_series_features)
        
        # Multi-scale features
        multi_scale_features = self.extract_multi_scale_features(y)
        features.update(multi_scale_features)
        
        # Combined feature matrix (for hybrid models)
        combined_features = []
        for key in ['mfcc', 'chroma', 'zcr', 'rms', 'spectral_centroid']:
            if key in features:
                combined_features.append(features[key])
        
        if combined_features:
            features['combined'] = np.vstack(combined_features)
        
        return features
    
    def extract_all_dl_features(self, file_path):
        """Extract all deep learning features from audio file"""
        y, sr = self.load_audio(file_path)
        if y is None:
            return None
        
        # Extract features
        features = self.extract_deep_learning_features(y)
        features['filename'] = os.path.basename(file_path)
        features['audio_length'] = len(y) / sr  # Audio length in seconds
        
        return features

def find_audio_files(root_dirs):
    """Find all WAV files in the specified directories"""
    audio_files = []
    for root_dir in root_dirs:
        if os.path.exists(root_dir):
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.lower().endswith(('.wav', '.WAV')):
                        full_path = os.path.join(dirpath, filename)
                        audio_files.append(full_path)
        else:
            print(f"Directory not found: {root_dir}")
    return audio_files

def save_features_hdf5(features_list, output_path):
    """Save features in HDF5 format for efficient storage"""
    with h5py.File(output_path, 'w') as hf:
        # Create datasets for each feature type
        for i, features in enumerate(features_list):
            grp = hf.create_group(f'sample_{i}')
            grp.attrs['filename'] = features['filename']
            grp.attrs['audio_length'] = features['audio_length']
            
            for key, value in features.items():
                if key not in ['filename', 'audio_length'] and isinstance(value, np.ndarray):
                    grp.create_dataset(key, data=value)

def save_features_numpy(features_list, output_dir):
    """Save features as numpy arrays"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, features in enumerate(features_list):
        filename = features['filename']
        base_name = os.path.splitext(filename)[0]
        
        # Save metadata
        metadata = {
            'filename': filename,
            'audio_length': features['audio_length']
        }
        np.savez(os.path.join(output_dir, f'{base_name}_meta.npz'), **metadata)
        
        # Save feature arrays
        for key, value in features.items():
            if key not in ['filename', 'audio_length'] and isinstance(value, np.ndarray):
                np.save(os.path.join(output_dir, f'{base_name}_{key}.npy'), value)

def main():
    parser = argparse.ArgumentParser(description='Audio Feature Extractor for CNN+LSTM Models')
    parser.add_argument('--output', '-o', default='dl_audio_features.h5', 
                       help='Output file name (HDF5 format)')
    parser.add_argument('--output_dir', '-od', default='dl_features', 
                       help='Output directory for numpy arrays')
    parser.add_argument('--output_format', '-of', choices=['hdf5', 'numpy', 'both'], 
                       default='both', help='Output format')
    parser.add_argument('--sample_rate', '-sr', type=int, default=22050,
                       help='Sample rate for audio processing')
    parser.add_argument('--frame_size', '-fs', type=int, default=2048,
                       help='Frame size for analysis')
    parser.add_argument('--hop_length', '-hl', type=int, default=512,
                       help='Hop length for analysis')
    parser.add_argument('--max_frames', '-mf', type=int, default=500,
                       help='Maximum number of time frames')
    
    args = parser.parse_args()
    
    # Define root directories
    ROOT_DIRS = [
        'datasets/bradford25/audio17-18_annotations/firstOne',
        'datasets/bradford25/audio17-18_annotations/SecondOne',
        'datasets/bradford25/AudioMM',
        'datasets/bradford25/AudioTO'
    ]
    
    # Find all audio files
    print("Searching for audio files...")
    audio_files = find_audio_files(ROOT_DIRS)
    
    if not audio_files:
        print("No WAV files found in the specified directories.")
        return
    
    print(f"Found {len(audio_files)} audio files.")
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor(
        sample_rate=args.sample_rate,
        frame_size=args.frame_size,
        hop_length=args.hop_length,
        max_frames=args.max_frames
    )
    
    # Extract features from each file
    all_features = []
    for audio_file in tqdm(audio_files, desc="Extracting DL features"):
        features = extractor.extract_all_dl_features(audio_file)
        if features:
            all_features.append(features)
    
    # Save features
    if all_features:
        if args.output_format in ['hdf5', 'both']:
            print(f"Saving features to HDF5: {args.output}")
            save_features_hdf5(all_features, args.output)
        
        if args.output_format in ['numpy', 'both']:
            print(f"Saving features to numpy arrays: {args.output_dir}")
            save_features_numpy(all_features, args.output_dir)
        
        # Print feature shapes for reference
        print("\nFeature shapes for first sample:")
        for key, value in all_features[0].items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape}")
        
        print(f"\nExtracted {len(all_features)} files with deep learning features.")
    else:
        print("No features were extracted.")

if __name__ == "__main__":
    main()