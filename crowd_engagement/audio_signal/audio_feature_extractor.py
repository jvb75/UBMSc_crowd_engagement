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
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, frame_size=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_length = hop_length
        
    def load_audio(self, file_path):
        """Load audio file using librosa"""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            return y, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None, None
    
    def extract_time_domain_features(self, y):
        """Extract time domain features"""
        features = {}
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_size, hop_length=self.hop_length)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['zcr_max'] = np.max(zcr)
        
        # Root Mean Square Energy
        rms = librosa.feature.rms(y=y, frame_length=self.frame_size, hop_length=self.hop_length)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)
        
        # Signal Entropy
        energy = np.array([np.sum(np.abs(y[i:i+self.frame_size]**2)) 
                          for i in range(0, len(y)-self.frame_size, self.hop_length)])
        energy = energy / (np.sum(energy) + 1e-10)  # Normalize to get probability distribution
        entropy = -np.sum(energy * np.log2(energy + 1e-10))
        features['signal_entropy'] = entropy
        
        return features
    
    def extract_frequency_domain_features(self, y):
        """Extract frequency domain features"""
        features = {}
        
        # Compute Short-Time Fourier Transform
        stft = np.abs(librosa.stft(y, n_fft=self.frame_size, hop_length=self.hop_length))
        
        # FFT Magnitude statistics
        magnitude = np.mean(stft, axis=1)
        features['fft_mean'] = np.mean(magnitude)
        features['fft_std'] = np.std(magnitude)
        features['fft_max'] = np.max(magnitude)
        
        # Dominant Frequency
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_size)
        dominant_freq = freqs[np.argmax(magnitude)]
        features['dominant_freq'] = dominant_freq
        
        # Band Energy Ratios
        # Define frequency bands (Hz): low (0-250), mid (250-2000), high (2000+)
        low_band = np.where(freqs <= 250)[0]
        mid_band = np.where((freqs > 250) & (freqs <= 2000))[0]
        high_band = np.where(freqs > 2000)[0]
        
        low_energy = np.sum(magnitude[low_band])
        mid_energy = np.sum(magnitude[mid_band])
        high_energy = np.sum(magnitude[high_band])
        total_energy = low_energy + mid_energy + high_energy + 1e-10
        
        features['low_band_energy_ratio'] = low_energy / total_energy
        features['mid_band_energy_ratio'] = mid_energy / total_energy
        features['high_band_energy_ratio'] = high_energy / total_energy
        
        return features
    
    def extract_spectral_features(self, y):
        """Extract spectral features"""
        features = {}
        
        # Compute mel-scaled spectrogram
        S = np.abs(librosa.stft(y, n_fft=self.frame_size, hop_length=self.hop_length))
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=self.sample_rate)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=self.sample_rate)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Spectral Flux
        spectral_flux = np.sum(np.diff(S, axis=1)**2, axis=0)
        features['spectral_flux_mean'] = np.mean(spectral_flux)
        features['spectral_flux_std'] = np.std(spectral_flux)
        
        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=self.sample_rate)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral Flatness
        spectral_flatness = librosa.feature.spectral_flatness(S=S)
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)
        
        return features
    
    def extract_cepstral_features(self, y):
        """Extract cepstral features"""
        features = {}
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=13, 
                                    n_fft=self.frame_size, hop_length=self.hop_length)
        
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        # Delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        for i in range(13):
            features[f'delta_mfcc_{i+1}_mean'] = np.mean(delta_mfccs[i])
            features[f'delta_mfcc_{i+1}_std'] = np.std(delta_mfccs[i])
        
        # Delta-Delta MFCCs
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        for i in range(13):
            features[f'delta2_mfcc_{i+1}_mean'] = np.mean(delta2_mfccs[i])
            features[f'delta2_mfcc_{i+1}_std'] = np.std(delta2_mfccs[i])
            
        return features
    
    def extract_perceptual_features(self, y):
        """Extract perceptual features"""
        features = {}
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate, 
                                           n_fft=self.frame_size, hop_length=self.hop_length)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
            features[f'chroma_{i}_std'] = np.std(chroma[i])
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=y, sr=self.sample_rate, 
                                        chroma=chroma)
        for i in range(6):
            features[f'tonnetz_{i}_mean'] = np.mean(tonnetz[i])
            features[f'tonnetz_{i}_std'] = np.std(tonnetz[i])
            
        # Loudness
        S = np.abs(librosa.stft(y, n_fft=self.frame_size, hop_length=self.hop_length))
        loudness = librosa.feature.rms(S=S)
        features['loudness_mean'] = np.mean(loudness)
        features['loudness_std'] = np.std(loudness)
        
        return features
    
    def extract_pitch_features(self, y):
        """Extract pitch-based features"""
        features = {}
        
        # Fundamental frequency (F0) using PYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                    fmin=librosa.note_to_hz('C2'),
                                                    fmax=librosa.note_to_hz('C7'),
                                                    sr=self.sample_rate,
                                                    frame_length=self.frame_size,
                                                    hop_length=self.hop_length)
        
        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]
        voiced_probs_clean = voiced_probs[~np.isnan(voiced_probs)]
        
        if len(f0_clean) > 0:
            features['f0_mean'] = np.mean(f0_clean)
            features['f0_std'] = np.std(f0_clean)
            features['f0_min'] = np.min(f0_clean)
            features['f0_max'] = np.max(f0_clean)
        else:
            features['f0_mean'] = 0
            features['f0_std'] = 0
            features['f0_min'] = 0
            features['f0_max'] = 0
            
        if len(voiced_probs_clean) > 0:
            features['voicing_prob_mean'] = np.mean(voiced_probs_clean)
            features['voicing_prob_std'] = np.std(voiced_probs_clean)
        else:
            features['voicing_prob_mean'] = 0
            features['voicing_prob_std'] = 0
            
        return features
    
    def extract_temporal_dynamics(self, y):
        """Extract temporal dynamics features"""
        features = {}
        
        # We'll use MFCCs to compute deltas as an example
        mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=13, 
                                    n_fft=self.frame_size, hop_length=self.hop_length)
        
        # Delta features (first derivative)
        delta_mfccs = librosa.feature.delta(mfccs)
        for i in range(13):
            features[f'mfcc_delta_{i+1}_mean'] = np.mean(delta_mfccs[i])
            features[f'mfcc_delta_{i+1}_std'] = np.std(delta_mfccs[i])
        
        # Delta-delta features (second derivative)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        for i in range(13):
            features[f'mfcc_delta2_{i+1}_mean'] = np.mean(delta2_mfccs[i])
            features[f'mfcc_delta2_{i+1}_std'] = np.std(delta2_mfccs[i])
            
        return features
    
    def extract_all_features(self, file_path, categories=None):
        """Extract all features or specified categories from audio file"""
        y, sr = self.load_audio(file_path)
        if y is None:
            return None
            
        all_features = {'filename': os.path.basename(file_path)}
        
        # Define extraction functions mapping
        extractors = {
            'time_domain': self.extract_time_domain_features,
            'frequency_domain': self.extract_frequency_domain_features,
            'spectral': self.extract_spectral_features,
            'cepstral': self.extract_cepstral_features,
            'perceptual': self.extract_perceptual_features,
            'pitch': self.extract_pitch_features,
            'temporal': self.extract_temporal_dynamics
        }
        
        # Extract all categories if none specified
        if categories is None:
            categories = list(extractors.keys())
        
        # Extract features for each requested category
        for category in categories:
            if category in extractors:
                try:
                    features = extractors[category](y)
                    all_features.update(features)
                except Exception as e:
                    print(f"Error extracting {category} features: {e}")
        
        return all_features

def find_audio_files(root_dirs):
    """Find all WAV files in the specified directories"""
    audio_files = []
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.wav', '.WAV')):
                    full_path = os.path.join(dirpath, filename)
                    audio_files.append(full_path)
    return audio_files

def main():
    parser = argparse.ArgumentParser(description='Audio Feature Extractor for Crowd Engagement Analysis')
    parser.add_argument('--output', '-o', default='audio_features.csv', 
                       help='Output CSV file name')
    parser.add_argument('--categories', '-c', nargs='+', 
                       choices=['time_domain', 'frequency_domain', 'spectral', 
                               'cepstral', 'perceptual', 'pitch', 'temporal'],
                       help='Specific feature categories to extract')
    parser.add_argument('--sample_rate', '-sr', type=int, default=22050,
                       help='Sample rate for audio processing')
    parser.add_argument('--frame_size', '-fs', type=int, default=2048,
                       help='Frame size for analysis')
    parser.add_argument('--hop_length', '-hl', type=int, default=512,
                       help='Hop length for analysis')
    
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
        hop_length=args.hop_length
    )
    
    # Extract features from each file
    all_features = []
    for audio_file in tqdm(audio_files, desc="Extracting features"):
        features = extractor.extract_all_features(audio_file, args.categories)
        if features:
            all_features.append(features)
    
    # Save to CSV
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(args.output, index=False)
        print(f"Features saved to {args.output}")
        print(f"Extracted {len(df)} files with {len(df.columns)} features each.")
    else:
        print("No features were extracted.")

if __name__ == "__main__":
    main()